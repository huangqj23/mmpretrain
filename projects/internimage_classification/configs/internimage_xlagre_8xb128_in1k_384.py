# Converted from projects/internimage_classification/configs/internimage-xlagre_8xb128_in1k-384.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from ._base_ import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip
from mmengine.optim import CosineAnnealingLR, LinearLR

from mmpretrain.datasets import PackInputs, RandomResizedCrop, ResizeEdge


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

model.merge(dict(
    backbone=dict(
        stem_channels=192,
        drop_path_rate=0.2,
        stage_blocks=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        layer_scale=1e-5,
        offset_scale=2.0,
        post_norm=True),
    head=dict(in_channels=2304)))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=384),
    dict(type=PackInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

optim_wrapper.merge(dict(optimizer=dict(lr=5e-6)))
param_scheduler = [
    dict(
        type=LinearLR,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, T_max=18, by_epoch=True, begin=2, end=20)
]
train_cfg.merge(dict(by_epoch=True, max_epochs=20, val_interval=1))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
