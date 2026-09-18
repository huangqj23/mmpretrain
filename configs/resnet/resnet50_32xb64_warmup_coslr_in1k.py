# Converted from configs/resnet/resnet50_32xb64-warmup-coslr_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs2048_coslr import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
