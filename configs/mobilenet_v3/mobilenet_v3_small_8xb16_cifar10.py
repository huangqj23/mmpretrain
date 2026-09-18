# Converted from configs/mobilenet_v3/mobilenet-v3-small_8xb16_cifar10.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobilenet_v3.mobilenet_v3_small_cifar import *  # noqa: F401,F403
    from .._base_.datasets.cifar10_bs16 import *  # noqa: F401,F403
    from .._base_.schedules.cifar10_bs128 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import MultiStepLR

# schedule settings
param_scheduler.merge(dict(
    type=MultiStepLR,
    by_epoch=True,
    milestones=[120, 170],
    gamma=0.1,
))

train_cfg.merge(dict(by_epoch=True, max_epochs=200))
