# Converted from configs/swav/swav_resnet50_8xb32-mcrop-coslr-200e_in1k-224px-96px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.schedules.imagenet_lars_coslr_200e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import (LoadImageFromFile, RandomApply, RandomFlip,
                             RandomGrayscale)
from mmengine.dataset import DefaultSampler, default_collate
from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, OptimWrapper
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmpretrain.datasets import (ColorJitter, GaussianBlur, ImageNet,
                                 MultiView, PackInputs, RandomResizedCrop)
from mmpretrain.engine import LARS, SwAVHook
from mmpretrain.models import (ResNet, SelfSupDataPreprocessor, SwAV, SwAVHead,
                               SwAVLoss, SwAVNeck)

# dataset settings
dataset_type = ImageNet
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type=SelfSupDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

num_crops = [2, 6]
color_distort_strength = 1.0
view_pipeline1 = [
    dict(
        type=RandomResizedCrop,
        scale=224,
        crop_ratio_range=(0.14, 1.),
        backend='pillow'),
    dict(
        type=RandomApply,
        transforms=[
            dict(
                type=ColorJitter,
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        prob=0.8),
    dict(
        type=RandomGrayscale,
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type=GaussianBlur,
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type=RandomFlip, prob=0.5),
]
view_pipeline2 = [
    dict(
        type=RandomResizedCrop,
        scale=96,
        crop_ratio_range=(0.05, 0.14),
        backend='pillow'),
    dict(
        type=RandomApply,
        transforms=[
            dict(
                type=ColorJitter,
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        prob=0.8),
    dict(
        type=RandomGrayscale,
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type=GaussianBlur,
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type=RandomFlip, prob=0.5),
]
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=MultiView,
        num_views=num_crops,
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type=PackInputs)
]

batch_size = 32
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# model settings
model = dict(
    type=SwAV,
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        to_rgb=True),
    backbone=dict(
        type=ResNet,
        depth=50,
        norm_cfg=dict(type=SyncBatchNorm),
        zero_init_residual=True),
    neck=dict(
        type=SwAVNeck,
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type=SwAVHead,
        loss=dict(
            type=SwAVLoss,
            feat_dim=128,  # equal to neck['out_channels']
            epsilon=0.05,
            temperature=0.1,
            num_crops=num_crops,
        )))

# optimizer
optim_wrapper.merge(dict(type=OptimWrapper, optimizer=dict(type=LARS, lr=0.6)))
find_unused_parameters = True

# learning policy
param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        T_max=200,
        eta_min=6e-4,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))

# additional hooks
custom_hooks = [
    dict(
        type=SwAVHook,
        priority='VERY_HIGH',
        batch_size=batch_size,
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=3840,
        frozen_layers_cfg=dict(prototypes=5005))
]
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
