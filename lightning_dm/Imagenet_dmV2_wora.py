import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from lightning import LightningDataModule
import presets
import utils
from sampler import RASampler
from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


class CustomDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, args):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.args = args
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None

    def setup(self, stage=None):
        val_resize_size, val_crop_size, train_crop_size = (
            self.args.val_resize_size,
            self.args.val_crop_size,
            self.args.train_crop_size,
        )
        interpolation = InterpolationMode(self.args.interpolation)

        print("Loading training data")
        cache_path_train = _get_cache_path(self.train_dir)
        if self.args.cache_dataset and os.path.exists(cache_path_train):
            print(f"Loading dataset_train from {cache_path_train}")
            self.train_dataset, _ = torch.load(cache_path_train, weights_only=False)
        else:
            auto_augment_policy = getattr(self.args, "auto_augment", None)
            random_erase_prob = getattr(self.args, "random_erase", 0.0)
            ra_magnitude = getattr(self.args, "ra_magnitude", None)
            augmix_severity = getattr(self.args, "augmix_severity", None)
            self.train_dataset = torchvision.datasets.ImageFolder(
                self.train_dir,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                    backend=self.args.backend,
                    use_v2=self.args.use_v2,
                ),
            )
            if self.args.cache_dataset:
                print(f"Saving dataset_train to {cache_path_train}")
                utils.mkdir(os.path.dirname(cache_path_train))
                utils.save_on_master((self.train_dataset, self.train_dir), cache_path_train)

        if hasattr(self.args, "ra_sampler") and self.args.ra_sampler:
            self.train_sampler = RASampler(self.train_dataset, shuffle=True, repetitions=self.args.ra_reps)
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)

        print("Loading validation data")
        cache_path_val = _get_cache_path(self.val_dir)
        if self.args.cache_dataset and os.path.exists(cache_path_val):
            print(f"Loading dataset_test from {cache_path_val}")
            self.val_dataset, _ = torch.load(cache_path_val, weights_only=False)
        else:
            if self.args.weights and self.args.test_only:
                weights = torchvision.models.get_weight(self.args.weights)
                preprocessing = weights.transforms(antialias=True)
                if self.args.backend == "tensor":
                    preprocessing = torchvision.transforms.Compose(
                        [torchvision.transforms.PILToTensor(), preprocessing])
            else:
                preprocessing = presets.ClassificationPresetEval(
                    crop_size=val_crop_size,
                    resize_size=val_resize_size,
                    interpolation=interpolation,
                    backend=self.args.backend,
                    use_v2=self.args.use_v2,
                )
            self.val_dataset = torchvision.datasets.ImageFolder(
                self.val_dir,
                preprocessing,
            )
            if self.args.cache_dataset:
                print(f"Saving dataset_test to {cache_path_val}")
                utils.mkdir(os.path.dirname(cache_path_val))
                utils.save_on_master((self.val_dataset, self.val_dir), cache_path_val)

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)

    def train_dataloader(self):
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=self.args.mixup_alpha,
            cutmix_alpha=self.args.cutmix_alpha,
            num_classes=len(self.train_dataset.classes),
            use_v2=self.args.use_v2
        )

        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            sampler=self.val_sampler,
            num_workers=self.args.workers,
            pin_memory=True,
        )
