# Converted from configs/twins/twins-pcpvt-large_16xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .twins_pcpvt_base_8xb128_in1k import *  # noqa: F401,F403

# model settings
model.merge(dict(backbone=dict(arch='large'), head=dict(in_channels=512)))

# dataset settings
train_dataloader.merge(dict(batch_size=64))
