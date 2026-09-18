# Converted from configs/_base_/models/replknet-XL_in1k.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, RepLKNet)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=RepLKNet,
        arch='XL',
        out_indices=(3, ),
    ),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
