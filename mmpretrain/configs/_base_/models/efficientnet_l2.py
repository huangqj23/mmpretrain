# Converted from configs/_base_/models/efficientnet_l2.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, EfficientNet,
                               GlobalAveragePooling, ImageClassifier,
                               LinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=EfficientNet, arch='l2'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=5504,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
