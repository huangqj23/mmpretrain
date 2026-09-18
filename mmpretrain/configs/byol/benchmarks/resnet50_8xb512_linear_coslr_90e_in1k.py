# Converted from configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.models.resnet50 import *  # noqa: F401,F403
    from ..._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_lars_coslr_90e import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.'))))

# dataset summary
train_dataloader.merge(dict(batch_size=512))

# runtime settings
default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
