# Converted from configs/_base_/models/inception_v3.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (ClsHead, CrossEntropyLoss, ImageClassifier,
                               InceptionV3)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=InceptionV3, num_classes=1000, aux_logits=False),
    neck=None,
    head=dict(
        type=ClsHead,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)),
)
