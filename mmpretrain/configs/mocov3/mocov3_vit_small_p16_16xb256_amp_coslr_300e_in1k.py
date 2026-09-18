# Converted from configs/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs512_mocov3 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset settings
# the difference between ResNet50 and ViT pipeline is the `scale` in
# `RandomResizedCrop`, `scale=(0.08, 1.)` in ViT pipeline
view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.08, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=1.),
    dict(type='Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.08, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.1),
    dict(type='Solarize', thr=128, prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackInputs')
]

train_dataloader.merge(dict(batch_size=256, dataset=dict(pipeline=train_pipeline)))

# model settings
temperature = 0.2
model = dict(
    type='MoCoV3',
    base_momentum=0.01,
    backbone=dict(
        type='MoCoV3ViT',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=384,
        hid_channels=4096,
        out_channels=256,
        num_layers=3,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        loss=dict(type='CrossEntropyLoss', loss_weight=2 * temperature),
        temperature=temperature))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2.4e-3, weight_decay=0.1))
find_unused_parameters = True

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
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
if isinstance(view_pipeline1, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    view_pipeline1.pop('_delete_', None)
if isinstance(view_pipeline2, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    view_pipeline2.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
