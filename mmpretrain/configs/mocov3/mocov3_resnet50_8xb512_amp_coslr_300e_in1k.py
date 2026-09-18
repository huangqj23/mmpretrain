# Converted from configs/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs512_mocov3 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
temperature = 1.0
model = dict(
    type='MoCoV3',
    base_momentum=0.01,  # 0.01 for 100e and 300e, 0.004 for 1000e
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=False),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=True),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=False,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        loss=dict(type='CrossEntropyLoss', loss_weight=2 * temperature),
        temperature=temperature))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9),
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
# only keeps the latest 3 checkpoints
default_hooks.merge(dict(checkpoint=dict(max_keep_ckpts=3)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
