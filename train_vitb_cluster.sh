MODEL_NAME=vit_b_16

python main.py \
    --model ${MODEL_NAME} \
    --epochs 300 \
    --batch-size 512 \
    --opt adamw \
    --lr 0.003 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 30 \
    --lr-warmup-decay 0.033 \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --ra-sampler \
    --cutmix-alpha 1.0 \
    --model-ema \
    --project lightning-vision \
    --name ${MODEL_NAME} \
    --offline \
    --trainer.accelerator npu \
    --trainer.num_nodes ${NNODES} \
    --trainer.precision 16 \
    --trainer.gradient_clip_val 1 \
    --trainer.log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /opt/huawei/dataset/all/torch_ds/imagenet