import torch
from timm.utils import accuracy
from timm import create_model
from lightning import LightningModule
import utils
import torchvision


class VisionClassifier(LightningModule):
    def __init__(
            self,
            config,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.save_hyperparameters()

        # self.model = create_model(
        #     model_name=config.model,
        #     pretrained=False,
        #     num_classes=config.nb_classes,
        #     drop_rate=config.drop,
        #     drop_path_rate=config.drop_path,
        #     drop_block_rate=None,
        #     img_size=config.input_size
        # )

        print(f"Creating model: {config.model}")
        self.model = create_model(
            model_name=config.model,
            pretrained=False,
            num_classes=config.num_classes,
            img_size=config.train_crop_size
        )

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # model EMA
        model_ema = None
        if config.model_ema:
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = config.world_size * config.batch_size * config.model_ema_steps / config.epochs
            alpha = 1.0 - config.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(self.model, device=self.device, decay=1.0 - alpha)
        self.model_ema = model_ema

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        loss = self.criterion(output, target)
        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log('val/loss', loss, sync_dist=True)
        self.log('val/acc1', acc1, sync_dist=True)
        self.log('val/acc5', acc5, sync_dist=True)

        return loss

    def configure_optimizers(self):
        config = self.config
        custom_keys_weight_decay = []
        if config.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", config.bias_weight_decay))
        if config.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, config.transformer_embedding_decay))
        parameters = utils.set_weight_decay(
            self.model,
            config.weight_decay,
            norm_weight_decay=config.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )

        opt_name = config.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, eps=0.0316,
                alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {config.opt}. Only SGD, RMSprop and AdamW are supported.")

        config.lr_scheduler = config.lr_scheduler.lower()
        if config.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size,
                                                                gamma=config.lr_gamma)
        elif config.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs - config.lr_warmup_epochs, eta_min=config.lr_min
            )
        elif config.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{config.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if config.lr_warmup_epochs > 0:
            if config.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=config.lr_warmup_decay, total_iters=config.lr_warmup_epochs
                )
            elif config.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=config.lr_warmup_decay, total_iters=config.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{config.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def lr_scheduler_step(self, scheduler, metric):
        # scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        scheduler.step()  # timm's scheduler need the epoch value

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update_parameters(self.model)
            if self.current_epoch < self.config.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                self.model_ema.n_averaged.fill_(0)

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     print("**************on_before_opt enter*********")
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #
    #     print("***************on_before_opt exit*********")
