# Converted from configs/mff/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from ..._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
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
    dict(type='PackInputs')
]

train_dataloader.merge(dict(batch_size=128, dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(batch_size=128, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='avg_featmap',
        final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

# optimizer wrapper
optim_wrapper.merge(dict(
    optimizer=dict(
        type='AdamW', lr=2e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.65,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks.merge(dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)))

train_cfg.merge(dict(by_epoch=True, max_epochs=100))

randomness.merge(dict(seed=0, diff_rank_seed=True))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
