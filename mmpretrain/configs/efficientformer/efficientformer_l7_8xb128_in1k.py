# Converted from configs/efficientformer/efficientformer-l7_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .efficientformer_l1_8xb128_in1k import *  # noqa: F401,F403

model.merge(dict(backbone=dict(arch='l7'), head=dict(in_channels=768)))
