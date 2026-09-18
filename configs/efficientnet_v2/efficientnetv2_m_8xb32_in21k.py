# Converted from configs/efficientnet_v2/efficientnetv2-m_8xb32_in21k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .efficientnetv2_s_8xb32_in21k import *  # noqa: F401,F403

# model setting
model.merge(dict(backbone=dict(arch='m'), ))
