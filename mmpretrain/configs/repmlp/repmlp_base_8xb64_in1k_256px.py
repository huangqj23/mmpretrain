# Converted from configs/repmlp/repmlp-base_8xb64_in1k-256px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.repmlp_base_224 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(backbone=dict(img_size=256)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=256),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=292, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule settings
optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=512))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
