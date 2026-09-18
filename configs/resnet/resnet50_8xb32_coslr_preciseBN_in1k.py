# Converted from configs/resnet/resnet50_8xb32-coslr-preciseBN_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .resnet50_8xb32_coslr_in1k import *  # noqa: F401,F403

from mmpretrain.engine import PreciseBNHook

# Precise BN hook will update the bn stats, so this hook should be executed
# before CheckpointHook(priority of 'VERY_LOW') and
# EMAHook(priority of 'NORMAL') So set the priority of PreciseBNHook to
# 'ABOVENORMAL' here.
custom_hooks = [
    dict(
        type=PreciseBNHook,
        num_samples=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]
