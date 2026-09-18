# Converted from configs/replknet/replknet-XL_32xb64_in1k-320px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.replknet_XL_in1k import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs8_pil_bicubic_320 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_coslr import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import CosineAnnealingLR

# schedule settings
param_scheduler.merge(dict(
    type=CosineAnnealingLR, T_max=300, by_epoch=True, begin=0, end=300))

train_cfg.merge(dict(by_epoch=True, max_epochs=300))
