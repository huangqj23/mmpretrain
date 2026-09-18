# Converted from configs/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32_mocov2 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_sgd_coslr_200e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='BN'),
        zero_init_residual=False),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2),
)
find_unused_parameters = True

# runtime settings
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
