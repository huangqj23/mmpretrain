# Converted from configs/convmixer/convmixer-1536-20_10xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.convmixer.convmixer_1536_20 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_convmixer_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import CosineAnnealingLR, LinearLR

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=0.01),
    clip_grad=dict(max_norm=5.0),
))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=130,
        eta_min=1e-5,
        by_epoch=True,
        begin=20,
        end=150)
]

train_cfg.merge(dict(by_epoch=True, max_epochs=150))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (10 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=640))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
