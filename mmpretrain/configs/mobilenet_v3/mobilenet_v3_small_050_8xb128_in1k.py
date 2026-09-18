# Converted from configs/mobilenet_v3/mobilenet-v3-small-050_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.models.mobilenet_v3.mobilenet_v3_small_050_imagenet import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs128_mbv3 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# model settings
model.merge(dict(backbone=dict(norm_cfg=dict(type='BN', eps=1e-5, momentum=0.1))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in [103.53, 116.28, 123.675]])),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=0.064,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-5))

param_scheduler = dict(type='StepLR', by_epoch=True, step_size=2, gamma=0.973)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
