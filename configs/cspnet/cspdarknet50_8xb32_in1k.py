# Converted from configs/cspnet/cspdarknet50_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip

from mmpretrain.datasets import PackInputs, RandomResizedCrop, ResizeEdge
from mmpretrain.models import (CrossEntropyLoss, CSPDarkNet,
                               GlobalAveragePooling, ImageClassifier,
                               LinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=CSPDarkNet, depth=53),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ))

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=288,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=256),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
