# Converted from configs/eva/eva-g-p14_8xb16_in1k-336px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.eva.eva_g import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs16_eva_336 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(backbone=dict(img_size=336)))
