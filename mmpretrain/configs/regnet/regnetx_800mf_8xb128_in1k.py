# Converted from configs/regnet/regnetx-800mf_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .regnetx_400mf_8xb128_in1k import *  # noqa: F401,F403

from mmpretrain.models import RegNet

# model settings
model.merge(dict(
    backbone=dict(type=RegNet, arch='regnetx_800mf'),
    head=dict(in_channels=672, )))
