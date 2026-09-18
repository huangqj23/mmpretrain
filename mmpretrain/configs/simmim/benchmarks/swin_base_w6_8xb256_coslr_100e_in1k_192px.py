# Converted from configs/simmim/benchmarks/swin-base-w6_8xb256-coslr-100e_in1k-192px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.models.swin_transformer.base_224 import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs256_swin_192 import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(
    backbone=dict(
        img_size=192,
        drop_path_rate=0.1,
        stage_cfgs=dict(block_cfgs=dict(window_size=6)),
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.'))))

# optimizer settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.9,
        custom_keys={
            '.norm': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2.5e-7 / 1.25e-3,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=2.5e-7 * 2048 / 512,
        by_epoch=True,
        begin=20,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
val_cfg = dict()
test_cfg = dict()

default_hooks.merge(dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100)))

randomness.merge(dict(seed=0))
