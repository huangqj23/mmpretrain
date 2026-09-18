# Converted from configs/hornet/hornet-small_8xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.hornet.hornet_small import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

data = dict(samples_per_gpu=64)

optim_wrapper.merge(dict(optimizer=dict(lr=4e-3), clip_grad=dict(max_norm=5.0)))

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
