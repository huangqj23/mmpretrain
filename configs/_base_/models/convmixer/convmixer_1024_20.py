# Converted from configs/_base_/models/convmixer/convmixer-1024-20.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (ConvMixer, CrossEntropyLoss,
                               GlobalAveragePooling, ImageClassifier,
                               LinearClsHead)

# Model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=ConvMixer, arch='1024/20'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))
