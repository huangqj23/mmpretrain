# Converted from configs/swin_transformer/swin-small_16xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.swin_transformer.small_224 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# schedule settings
optim_wrapper.merge(dict(clip_grad=dict(max_norm=5.0)))
