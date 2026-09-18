# Converted from configs/_base_/models/deit3/deit3-base-p16-384.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, DeiT3, ImageClassifier, LabelSmoothLoss,
                               Mixup, VisionTransformerClsHead)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=DeiT3,
        arch='b',
        img_size=384,
        patch_size=16,
        drop_path_rate=0.15),
    neck=None,
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=768,
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
