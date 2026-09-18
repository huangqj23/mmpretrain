# Converted from configs/davit/davit-base_4xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.davit.davit_base import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs256_davit_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# data settings
train_dataloader.merge(dict(batch_size=256))
