# Converted from configs/simmim/benchmarks/swin-large-w14_8xb256-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from ..._base_.models.swin_transformer.base_224 import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs256_swin_192 import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip
from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.model import PretrainedInit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.datasets import (PackInputs, RandAugment, RandomErasing,
                                 RandomResizedCrop, ResizeEdge)


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
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type=RandomErasing,
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type=PackInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# model settings
model.merge(dict(
    backbone=dict(
        arch='large',
        img_size=224,
        drop_path_rate=0.2,
        stage_cfgs=dict(block_cfgs=dict(window_size=14)),
        pad_small_map=True,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.')),
    head=dict(in_channels=1536)))

# optimizer settings
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=5e-3, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.7,
        custom_keys={
            '.norm': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=2.5e-7 / 1.25e-3,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=100,
        eta_min=1e-6,
        by_epoch=True,
        begin=20,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=100)
val_cfg = dict()
test_cfg = dict()

default_hooks.merge(dict(
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3),
    logger=dict(type=LoggerHook, interval=100)))

randomness.merge(dict(seed=0))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
