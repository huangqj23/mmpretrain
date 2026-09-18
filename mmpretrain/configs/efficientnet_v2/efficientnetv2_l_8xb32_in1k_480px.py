# Converted from configs/efficientnet_v2/efficientnetv2-l_8xb32_in1k-480px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .efficientnetv2_s_8xb32_in1k_384px import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomFlip

from mmpretrain.datasets import (EfficientNetCenterCrop,
                                 EfficientNetRandomCrop, PackInputs)

# model setting
model.merge(dict(backbone=dict(arch='l'), ))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=EfficientNetRandomCrop, scale=384, crop_padding=0),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=EfficientNetCenterCrop, crop_size=480, crop_padding=0),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
