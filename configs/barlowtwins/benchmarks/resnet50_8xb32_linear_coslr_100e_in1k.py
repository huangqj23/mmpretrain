# Converted from configs/barlowtwins/benchmarks/resnet50_8xb32-linear-coslr-100e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.models.resnet50 import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_sgd_coslr_100e import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.'))))

# runtime settings
default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
