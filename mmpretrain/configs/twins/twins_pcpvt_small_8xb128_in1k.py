# Converted from configs/twins/twins-pcpvt-small_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .twins_pcpvt_base_8xb128_in1k import *  # noqa: F401,F403

# model settings
model.merge(dict(backbone=dict(arch='small'), head=dict(in_channels=512)))
