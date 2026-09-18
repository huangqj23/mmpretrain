# Converted from configs/efficientnet_v2/efficientnetv2-s_8xb32_in21k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.efficientnet_v2.efficientnetv2_s import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model setting
model.merge(dict(head=dict(num_classes=21843)))

# dataset settings
dataset_type = 'ImageNet21k'
data_preprocessor.merge(dict(
    num_classes=21843,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224, crop_padding=0),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
))
if isinstance(dataset_type, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    dataset_type.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
