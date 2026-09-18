# Converted from configs/levit/levit-128_8xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.levit_256_p16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs2048_adamw_levit import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(backbone=dict(arch='128'), head=dict(in_channels=384)))

# dataset settings
train_dataloader.merge(dict(batch_size=256))
