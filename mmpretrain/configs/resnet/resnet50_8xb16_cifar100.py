# Converted from configs/resnet/resnet50_8xb16_cifar100.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50_cifar import *  # noqa: F401,F403
    from .._base_.datasets.cifar100_bs16 import *  # noqa: F401,F403
    from .._base_.schedules.cifar10_bs128 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.optim import MultiStepLR

# model settings
model.merge(dict(head=dict(num_classes=100)))

# schedule settings
optim_wrapper.merge(dict(optimizer=dict(weight_decay=0.0005)))

param_scheduler.merge(dict(
    type=MultiStepLR,
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
))
