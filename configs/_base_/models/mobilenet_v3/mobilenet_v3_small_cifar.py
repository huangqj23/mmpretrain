# Converted from configs/_base_/models/mobilenet_v3/mobilenet_v3_small_cifar.py by industrial-vision tools/convert_configs.py
from torch.nn import Hardswish

from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, MobileNetV3,
                               StackedLinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=MobileNetV3, arch='small'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=StackedLinearClsHead,
        num_classes=10,
        in_channels=576,
        mid_channels=[1280],
        act_cfg=dict(type=Hardswish),
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)))
