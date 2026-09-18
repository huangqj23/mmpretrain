# Converted from configs/mixmim/benchmarks/mixmim-base_8xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.models.mixmim.mixmim_base import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403
