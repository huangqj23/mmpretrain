# Converted from configs/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32_mocov2 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_sgd_coslr_200e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
from torch.optim import SGD

from mmpretrain.engine import SimSiamHook
from mmpretrain.models import (CosineSimilarityLoss, LatentPredictHead,
                               NonLinearNeck, ResNet, SimSiam)

# model settings
model = dict(
    type=SimSiam,
    backbone=dict(
        type=ResNet,
        depth=50,
        norm_cfg=dict(type=SyncBatchNorm),
        zero_init_residual=True),
    neck=dict(
        type=NonLinearNeck,
        in_channels=2048,
        hid_channels=2048,
        out_channels=2048,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type=LatentPredictHead,
        loss=dict(type=CosineSimilarityLoss),
        predictor=dict(
            type=NonLinearNeck,
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
)

# optimizer
# set base learning rate
lr = 0.05
optim_wrapper.merge(dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=lr, weight_decay=1e-4, momentum=0.9),
    paramwise_cfg=dict(custom_keys={'predictor': dict(fix_lr=True)})))

# learning rate scheduler
param_scheduler = [
    dict(type=CosineAnnealingLR, T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
train_cfg.merge(dict(type=EpochBasedTrainLoop, max_epochs=100))
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))

# additional hooks
custom_hooks = [
    dict(type=SimSiamHook, priority='HIGH', fix_pred_lr=True, lr=lr)
]
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
