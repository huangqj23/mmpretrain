# Converted from configs/vision_transformer/vit-base-p16_64xb64_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.vit_base_p16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model setting
model.merge(dict(backbone=dict(img_size=384)))

# dataset setting
data_preprocessor.merge(dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule setting
optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
