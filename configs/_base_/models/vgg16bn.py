# Converted from configs/_base_/models/vgg16bn.py by industrial-vision tools/convert_configs.py
from torch.nn import BatchNorm2d

from mmpretrain.models import VGG, ClsHead, CrossEntropyLoss, ImageClassifier

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=VGG, depth=16, norm_cfg=dict(type=BatchNorm2d), num_classes=1000),
    neck=None,
    head=dict(
        type=ClsHead,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))
