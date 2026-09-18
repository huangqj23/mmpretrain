# Converted from configs/res2net/res2net50-w14-s8_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.res2net50_w14_s8 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
