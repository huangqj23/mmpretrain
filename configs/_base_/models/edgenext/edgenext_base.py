# Converted from configs/_base_/models/edgenext/edgenext-base.py by industrial-vision tools/convert_configs.py
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CrossEntropyLoss, EdgeNeXt, ImageClassifier,
                               LinearClsHead)

# Model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=EdgeNeXt,
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type=TruncNormalInit,
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type=ConstantInit, layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=584,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))
