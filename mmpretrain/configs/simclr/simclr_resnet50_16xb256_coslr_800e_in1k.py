# Converted from configs/simclr/simclr_resnet50_16xb256-coslr-800e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32_simclr import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_lars_coslr_200e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmpretrain.engine import LARS
from mmpretrain.models import (ContrastiveHead, CrossEntropyLoss,
                               NonLinearNeck, ResNet, SimCLR)

# model settings
model = dict(
    type=SimCLR,
    backbone=dict(
        type=ResNet,
        depth=50,
        norm_cfg=dict(type=SyncBatchNorm),
        zero_init_residual=True),
    neck=dict(
        type=NonLinearNeck,  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type=ContrastiveHead,
        loss=dict(type=CrossEntropyLoss),
        temperature=0.1),
)

# optimizer
optim_wrapper.merge(dict(
    type=OptimWrapper,
    optimizer=dict(type=LARS, lr=4.8, momentum=0.9, weight_decay=1e-6),
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        })))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR, T_max=790, by_epoch=True, begin=10, end=800)
]

# runtime settings
train_cfg.merge(dict(max_epochs=800))
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
