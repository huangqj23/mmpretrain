# Converted from configs/resnet/resnet50_8xb128_coslr-90e_in21k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet21k_bs128 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_coslr import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(head=dict(num_classes=21843)))

# runtime settings
train_cfg.merge(dict(by_epoch=True, max_epochs=90))
