# Converted from configs/convnext_v2/convnext-v2-huge_32xb32_in1k-512px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.convnext_v2.huge import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset setting
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=512,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(batch_size=32, dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=2.5e-3),
    clip_grad=None,
))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

# train, val, test setting
train_cfg.merge(dict(by_epoch=True, max_epochs=100, val_interval=1))

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
