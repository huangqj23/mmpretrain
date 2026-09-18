# Converted from projects/internimage_classification/configs/_base_.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from mmpretrain.configs._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip
from mmengine.dataset import DefaultSampler
from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from mmpretrain.datasets import (ImageNet, PackInputs, RandomResizedCrop,
                                 ResizeEdge)
from mmpretrain.evaluation import Accuracy
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead)

def __iv_scope(cfg, scope, has_scope=True):
    """复刻旧式跨库继承（scope::path）时 mmengine 给 base 变量加 _scope_ 的逻辑：
    每条路径上最外层带 type 的 dict 加 _scope_。"""
    if isinstance(cfg, dict):
        if has_scope and 'type' in cfg:
            has_scope = False
            if cfg.get('_scope_') is None:
                cfg['_scope_'] = scope
        for k in list(cfg):
            cfg[k] = __iv_scope(cfg[k], scope, has_scope)
    elif isinstance(cfg, list):
        cfg = [__iv_scope(v, scope, has_scope) for v in cfg]
    elif isinstance(cfg, tuple):
        cfg = tuple(__iv_scope(v, scope, has_scope) for v in cfg)
    return cfg

default_hooks = __iv_scope(default_hooks, 'mmpretrain')
default_scope = __iv_scope(default_scope, 'mmpretrain')
env_cfg = __iv_scope(env_cfg, 'mmpretrain')
load_from = __iv_scope(load_from, 'mmpretrain')
log_level = __iv_scope(log_level, 'mmpretrain')
randomness = __iv_scope(randomness, 'mmpretrain')
resume = __iv_scope(resume, 'mmpretrain')
vis_backends = __iv_scope(vis_backends, 'mmpretrain')
visualizer = __iv_scope(visualizer, 'mmpretrain')

# dataset settings
dataset_type = ImageNet
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=224,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/imagenet',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/imagenet',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy, topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# model setting
custom_imports = dict(imports='models')

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type='InternImage',
        stem_channels=64,
        drop_path_rate=0.1,
        stage_blocks=[4, 4, 18, 4],
        groups=[4, 8, 16, 32]),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type=AdamW, lr=1.25e-04, eps=1e-8, betas=(0.9, 0.999)),
    weight_decay=0.05)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=280,
        by_epoch=True,
        begin=20,
        end=300,
        eta_min=1.25e-06)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128 * 8)
