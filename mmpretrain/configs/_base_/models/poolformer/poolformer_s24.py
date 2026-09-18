# Converted from configs/_base_/models/poolformer/poolformer_s24.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, PoolFormer)

# Model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=PoolFormer,
        arch='s24',
        drop_path_rate=0.1,
        init_cfg=[
            dict(
                type=TruncNormalInit,
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type=ConstantInit, layer=['GroupNorm'], val=1., bias=0.),
        ]),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=512,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))
