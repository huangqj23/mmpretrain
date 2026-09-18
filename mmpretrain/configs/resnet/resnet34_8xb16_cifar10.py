# Converted from configs/resnet/resnet34_8xb16_cifar10.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet34_cifar import *  # noqa: F401,F403
    from .._base_.datasets.cifar10_bs16 import *  # noqa: F401,F403
    from .._base_.schedules.cifar10_bs128 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
