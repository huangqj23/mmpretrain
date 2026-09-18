# Converted from configs/arcface/resnet50-arcface_8xb32_inshop.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.inshop_bs32_448 import *  # noqa: F401,F403
    from .._base_.schedules.cub_bs64 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook, LoggerHook, SyncBuffersHook
from mmengine.model import PretrainedInit
from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import SGD

from mmpretrain.engine import PrepareProtoBeforeValLoopHook
from mmpretrain.models import (ArcFaceClsHead, CrossEntropyLoss,
                               GlobalAveragePooling, ImageToImageRetriever,
                               ResNet)

pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa
model = dict(
    type=ImageToImageRetriever,
    image_encoder=[
        dict(
            type=ResNet,
            depth=50,
            init_cfg=dict(
                type=PretrainedInit, checkpoint=pretrained, prefix='backbone')),
        dict(type=GlobalAveragePooling),
    ],
    head=dict(
        type=ArcFaceClsHead,
        num_classes=3997,
        in_channels=2048,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        init_cfg=None),
    prototype=gallery_dataloader)

# runtime settings
default_hooks.merge(dict(
    # log every 20 intervals
    logger=dict(type=LoggerHook, interval=20),
    # save last three checkpoints
    checkpoint=dict(
        type=CheckpointHook,
        save_best='auto',
        interval=1,
        max_keep_ckpts=3,
        rule='greater')))

# optimizer
optim_wrapper.merge(dict(
    optimizer=dict(
        type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0005, nesterov=True)))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
    )
]

train_cfg.merge(dict(by_epoch=True, max_epochs=50, val_interval=1))

auto_scale_lr.merge(dict(enable=True, base_batch_size=256))

custom_hooks = [
    dict(type=PrepareProtoBeforeValLoopHook),
    dict(type=SyncBuffersHook)
]
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
