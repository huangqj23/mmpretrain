# Converted from configs/mae/benchmarks/vit-base-p16_8xb2048-linear-coslr-90e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.model import PretrainedInit, TruncNormalInit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR

from mmpretrain.engine import LARS
from mmpretrain.models import (ClsBatchNormNeck, CrossEntropyLoss,
                               ImageClassifier, VisionTransformer,
                               VisionTransformerClsHead)

# dataset settings
train_dataloader.merge(dict(batch_size=2048, drop_last=True))
val_dataloader.merge(dict(drop_last=False))
test_dataloader.merge(dict(drop_last=False))

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=VisionTransformer,
        arch='base',
        img_size=224,
        patch_size=16,
        frozen_stages=12,
        out_type='cls_token',
        final_norm=True,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.')),
    neck=dict(type=ClsBatchNormNeck, input_features=768),
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=768,
        loss=dict(type=CrossEntropyLoss),
        init_cfg=[dict(type=TruncNormalInit, layer='Linear', std=0.01)]))

# optimizer
optim_wrapper.merge(dict(
    _delete_=True,
    type=AmpOptimWrapper,
    optimizer=dict(type=LARS, lr=6.4, weight_decay=0.0, momentum=0.9)))

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
        type=CosineAnnealingLR,
        T_max=80,
        by_epoch=True,
        begin=10,
        end=90,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg.merge(dict(by_epoch=True, max_epochs=90))

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3),
    logger=dict(type=LoggerHook, interval=10)))

randomness.merge(dict(seed=0, diff_rank_seed=True))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
