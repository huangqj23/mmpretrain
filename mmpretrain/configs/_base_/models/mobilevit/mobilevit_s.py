# Converted from configs/_base_/models/mobilevit/mobilevit_s.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, MobileViT)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=MobileViT, arch='small'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=640,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
