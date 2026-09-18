# Converted from configs/_base_/models/resnet18_cifar.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, ResNet_CIFAR)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet_CIFAR,
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=10,
        in_channels=512,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))
