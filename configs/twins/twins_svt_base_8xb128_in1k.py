# Converted from configs/twins/twins-svt-base_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.twins_svt_base import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import AdamW

# dataset settings
train_dataloader.merge(dict(batch_size=128))

# schedule settings
optim_wrapper.merge(dict(
    optimizer=dict(
        type=AdamW,
        lr=5e-4 * 128 * 8 / 512,  # learning rate for 128 batch size, 8 gpu.
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=5.0),
))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=295,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=300)
]
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
