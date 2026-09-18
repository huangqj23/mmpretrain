# Converted from configs/convnext_v2/convnext-v2-nano_32xb32_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.convnext_v2.nano import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import CosineAnnealingLR

from mmpretrain.engine import EMAHook

# dataset setting
train_dataloader.merge(dict(batch_size=32))

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=8e-4, weight_decay=0.3),
    clip_grad=None,
))

# learning policy
param_scheduler = [dict(type=CosineAnnealingLR, eta_min=1e-5, by_epoch=True)]

# train, val, test setting
train_cfg.merge(dict(by_epoch=True, max_epochs=600, val_interval=1))

# runtime setting
custom_hooks = [dict(type=EMAHook, momentum=1e-4, priority='ABOVE_NORMAL')]
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
