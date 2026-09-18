# Converted from configs/_base_/models/revvit/revvit-base.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, ImageClassifier, LabelSmoothLoss,
                               LinearClsHead, Mixup, RevVisionTransformer)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=RevVisionTransformer,
        arch='deit-base',
        img_size=224,
        patch_size=16,
        out_type='avg_featmap',
    ),
    neck=None,
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1536,
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
    ]),
)
