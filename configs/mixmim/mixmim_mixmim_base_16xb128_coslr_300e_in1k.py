# Converted from configs/mixmim/mixmim_mixmim-base_16xb128-coslr-300e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomFlip
from mmengine.dataset import DefaultSampler, default_collate
from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.datasets import ImageNet, PackInputs, RandomResizedCrop
from mmpretrain.models import (MixMIM, MixMIMPretrainDecoder,
                               MixMIMPretrainHead, MixMIMPretrainTransformer,
                               PixelReconstructionLoss,
                               SelfSupDataPreprocessor)

# dataset settings
dataset_type = ImageNet
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type=SelfSupDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackInputs)
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
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
    type=MixMIM,
    backbone=dict(
        type=MixMIMPretrainTransformer,
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.0,  # drop_path_rate=0.0 during pretraining
        mask_ratio=0.5),
    neck=dict(
        type=MixMIMPretrainDecoder,
        num_patches=49,
        encoder_stride=32,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16),
    head=dict(
        type=MixMIMPretrainHead,
        norm_pix=True,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')))

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1.5e-4 * (2048 / 256),
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'ln': dict(decay_mult=0.0),
        'bias': dict(decay_mult=0.0)
    }))

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=300)
default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=1)))

randomness.merge(dict(seed=0, diff_rank_seed=True))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
