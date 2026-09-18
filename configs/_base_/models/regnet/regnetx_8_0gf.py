# Converted from configs/_base_/models/regnet/regnetx_8.0gf.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, RegNet)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=RegNet, arch='regnetx_8.0gf'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1920,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
