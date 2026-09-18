# Converted from configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.shufflenet_v1_1x import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_linearlr_bn_nowd import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
