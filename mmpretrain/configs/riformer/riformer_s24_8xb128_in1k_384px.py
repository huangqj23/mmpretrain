# Converted from configs/riformer/riformer-s24_8xb128_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs128_riformer_small_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RIFormer',
        arch='s24',
        drop_path_rate=0.1,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
        ]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# schedule settings
optim_wrapper.merge(dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=4096))
