# Converted from configs/tinyvit/tinyvit-11m_8xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32_pil_bicubic import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.models.tinyvit.tinyvit_11m import *  # noqa: F401,F403
