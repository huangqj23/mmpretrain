# Converted from configs/eva02/eva02-large-p14_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs16_eva_448 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs2048_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, ImageClassifier, LabelSmoothLoss,
                               LinearClsHead, Mixup, ViTEVA02)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ViTEVA02,
        arch='l',
        img_size=448,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap'),
    neck=None,
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=.02),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0)
    ]))
