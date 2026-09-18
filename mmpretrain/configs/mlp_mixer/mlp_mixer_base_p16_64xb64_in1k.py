# Converted from configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mlp_mixer_base_patch16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_mixer_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))
