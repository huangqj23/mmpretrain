# Converted from configs/revvit/revvit-base_8xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.revvit.revvit_base import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs128_revvit_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_revvit import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
