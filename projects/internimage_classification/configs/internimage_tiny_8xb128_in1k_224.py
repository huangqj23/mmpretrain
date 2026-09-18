# Converted from projects/internimage_classification/configs/internimage-tiny_8xb128_in1k-224.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ._base_ import *  # noqa: F401,F403

model.merge(dict(
    backbone=dict(
        stem_channels=64,
        drop_path_rate=0.1,
        stage_blocks=[4, 4, 18, 4],
        groups=[4, 8, 16, 32])))
