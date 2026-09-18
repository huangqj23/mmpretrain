# Converted from configs/simmim/simmim_swin-large-w12_16xb128-amp-coslr-800e_in1k-192px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs256_simmim_192 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.models import (PixelReconstructionLoss, SimMIM, SimMIMHead,
                               SimMIMLinearDecoder, SimMIMSwinTransformer)

# model settings
model = dict(
    type=SimMIM,
    backbone=dict(
        type=SimMIMSwinTransformer,
        arch='large',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        pad_small_map=True),
    neck=dict(
        type=SimMIMLinearDecoder, in_channels=192 * 2**3, encoder_stride=32),
    head=dict(
        type=SimMIMHead,
        patch_size=4,
        loss=dict(type=PixelReconstructionLoss, criterion='L1', channel=3)))

# optimizer wrapper
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1e-4 * 2048 / 512,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=5e-7 / 1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type=MultiStepLR,
        milestones=[700],
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# runtime
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=800)
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
