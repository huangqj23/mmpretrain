# Converted from configs/_base_/models/van/van_large.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (VAN, GlobalAveragePooling, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=VAN, arch='large', drop_path_rate=0.2),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=512,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        cal_acc=False))
