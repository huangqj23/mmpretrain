# Converted from configs/_base_/models/mobilenet_v3/mobilenet_v3_small_075_imagenet.py by industrial-vision tools/convert_configs.py
from mmengine.model import NormalInit
from torch.nn import Hardswish

from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, MobileNetV3,
                               StackedLinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=MobileNetV3, arch='small_075'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=StackedLinearClsHead,
        num_classes=1000,
        in_channels=432,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type=Hardswish),
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        init_cfg=dict(
            type=NormalInit, layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))
