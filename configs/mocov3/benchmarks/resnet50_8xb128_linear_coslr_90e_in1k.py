# Converted from configs/mocov3/benchmarks/resnet50_8xb128-linear-coslr-90e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.models.resnet50 import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_sgd_coslr_100e import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import PretrainedInit
from mmengine.optim import CosineAnnealingLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import SGD

# dataset settings
train_dataloader.merge(dict(batch_size=128))

model.merge(dict(
    backbone=dict(
        frozen_stages=4,
        norm_eval=True,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.'))))

# optimizer
optim_wrapper.merge(dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.4, momentum=0.9, weight_decay=0.)))

# learning rate scheduler
param_scheduler = [
    dict(type=CosineAnnealingLR, T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg.merge(dict(type=EpochBasedTrainLoop, max_epochs=90))

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
