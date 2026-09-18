# Converted from configs/efficientnet_v2/efficientnetv2-b2_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .efficientnetv2_b0_8xb32_in1k import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomFlip

from mmpretrain.datasets import (EfficientNetCenterCrop,
                                 EfficientNetRandomCrop, PackInputs)

# model setting
model.merge(dict(backbone=dict(arch='b2'), head=dict(in_channels=1408, )))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=EfficientNetRandomCrop, scale=208),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=EfficientNetCenterCrop, crop_size=260, crop_padding=0),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
