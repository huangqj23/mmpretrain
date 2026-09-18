# Converted from configs/spark/spark_sparse-resnet50_8xb512-amp-coslr-800e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs512_mae import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop

from mmpretrain.engine import CosineAnnealingWeightDecay, Lamb
from mmpretrain.models import (PixelReconstructionLoss, SparK,
                               SparKLightDecoder, SparKPretrainHead,
                               SparseResNet, SparseSyncBatchNorm2d)

# dataset 8 x 512
train_dataloader.merge(dict(batch_size=512, num_workers=8))

# model settings
model = dict(
    type=SparK,
    input_size=224,
    downsample_raito=32,
    mask_ratio=0.6,
    enc_dec_norm_cfg=dict(type=SparseSyncBatchNorm2d),
    enc_dec_norm_dim=2048,
    backbone=dict(
        type=SparseResNet,
        depth=50,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.05),
    neck=dict(
        type=SparKLightDecoder,
        feature_dim=512,
        upsample_ratio=32,  # equal to downsample_raito
        mid_channels=0,
        last_act=False),
    head=dict(
        type=SparKPretrainHead,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')))

# optimizer wrapper
optimizer = dict(
    type=Lamb, lr=2e-4 * 4096 / 512, betas=(0.9, 0.95), weight_decay=0.04)
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            'mask_token': dict(decay_mult=0.),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=760,
        by_epoch=True,
        begin=40,
        end=800,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingWeightDecay,
        eta_min=0.2,
        T_max=800,
        by_epoch=True,
        begin=0,
        end=800,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=800)
default_hooks.merge(dict(
    logger=dict(type=LoggerHook, interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=2)))

# randomness
randomness.merge(dict(seed=0, diff_rank_seed=True))
