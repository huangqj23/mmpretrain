# Converted from configs/mixmim/benchmarks/mixmim-base_8xb128-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from ..._base_.models.mixmim.mixmim_base import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip
from mmengine.dataset import DefaultSampler, default_collate
from mmengine.hooks import CheckpointHook
from mmengine.model import PretrainedInit
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from torch.optim import AdamW

from mmpretrain.datasets import (ImageNet, PackInputs, RandAugment,
                                 RandomErasing, RandomResizedCrop, ResizeEdge)


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# dataset settings
dataset_type = ImageNet
data_root = 'data/imagenet/'

data_preprocessor.merge(dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
))

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(
        type=RandAugment,
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type=RandomErasing,
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
    persistent_workers=True,
))

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

val_dataloader.merge(dict(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True,
))
test_dataloader = val_dataloader

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.'))))

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=5e-4 * (8 * 128 / 256),
        betas=(0.9, 0.999),
        weight_decay=0.05),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.7,
        custom_keys={
            '.ln': dict(decay_mult=0.0),  # do not decay on ln and bias
            '.bias': dict(decay_mult=0.0)
        }))

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=95,
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict()

default_hooks.merge(dict(
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=1)))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(dataset_type, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    dataset_type.pop('_delete_', None)
if isinstance(bgr_mean, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    bgr_mean.pop('_delete_', None)
if isinstance(bgr_std, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    bgr_std.pop('_delete_', None)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
