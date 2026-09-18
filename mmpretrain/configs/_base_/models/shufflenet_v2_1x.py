# Converted from configs/_base_/models/shufflenet_v2_1x.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, ShuffleNetV2)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=ShuffleNetV2, widen_factor=1.0),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
