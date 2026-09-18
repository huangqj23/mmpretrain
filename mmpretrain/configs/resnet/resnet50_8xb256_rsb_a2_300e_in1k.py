# Converted from configs/resnet/resnet50_8xb256-rsb-a2-300e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs256_rsb_a12 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs2048_rsb import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
    ),
    head=dict(loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ])))

# dataset settings
train_dataloader.merge(dict(sampler=dict(type='RepeatAugSampler', shuffle=True)))

# schedule settings
optim_wrapper.merge(dict(
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]
train_cfg.merge(dict(by_epoch=True, max_epochs=300))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
