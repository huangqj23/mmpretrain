# Converted from configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobilenet_v2_1x import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_epochstep import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
