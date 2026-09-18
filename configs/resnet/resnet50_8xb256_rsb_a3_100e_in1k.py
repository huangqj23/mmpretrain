# Converted from configs/resnet/resnet50_8xb256-rsb-a3-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs256_rsb_a3 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs2048_rsb import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmpretrain.models import CutMix, Mixup

# model settings
model.merge(dict(
    backbone=dict(norm_cfg=dict(type=SyncBatchNorm, requires_grad=True)),
    head=dict(loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.1),
        dict(type=CutMix, alpha=1.0)
    ]),
))

# schedule settings
optim_wrapper.merge(dict(
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
))
