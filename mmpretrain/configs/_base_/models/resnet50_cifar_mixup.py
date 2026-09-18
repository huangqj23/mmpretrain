# Converted from configs/_base_/models/resnet50_cifar_mixup.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, Mixup, MultiLabelLinearClsHead,
                               ResNet_CIFAR)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet_CIFAR,
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=MultiLabelLinearClsHead,
        num_classes=10,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0, use_soft=True)),
    train_cfg=dict(augments=dict(type=Mixup, alpha=1.)),
)
