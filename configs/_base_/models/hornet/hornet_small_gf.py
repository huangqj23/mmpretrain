# Converted from configs/_base_/models/hornet/hornet-small-gf.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, HorNet, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead, Mixup)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=HorNet, arch='small-gf', drop_path_rate=0.4),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
        dict(type=ConstantInit, layer=['LayerScale'], val=1e-6)
    ],
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0)
    ]))
