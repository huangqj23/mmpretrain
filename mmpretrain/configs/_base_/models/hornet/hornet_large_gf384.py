# Converted from configs/_base_/models/hornet/hornet-large-gf384.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (HorNet, ImageClassifier, LabelSmoothLoss,
                               LinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=HorNet, arch='large-gf384', drop_path_rate=0.4),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1536,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
        dict(type=ConstantInit, layer=['LayerScale'], val=1e-6)
    ])
