# Converted from configs/mocov3/benchmarks/vit-large-p16_8xb64-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import ConstantInit, PretrainedInit, TruncNormalInit
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.engine import EMAHook
from mmpretrain.models import (CutMix, ImageClassifier, LabelSmoothLoss, Mixup,
                               VisionTransformer, VisionTransformerClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=VisionTransformer,
        arch='large',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.5,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.')),
    neck=None,
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        init_cfg=[
            dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
            dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
        ]),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0)
    ]))

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=5e-4, eps=1e-8, betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-3,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=95,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=100)
val_cfg = dict()
test_cfg = dict()

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
custom_hooks = [dict(type=EMAHook, momentum=4e-5, priority='ABOVE_NORMAL')]

randomness.merge(dict(seed=0))
