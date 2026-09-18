# Converted from configs/repvgg/repvgg-B1g4_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .repvgg_A0_8xb32_in1k import *  # noqa: F401,F403

model.merge(dict(backbone=dict(arch='B1g4'), head=dict(in_channels=2048)))
