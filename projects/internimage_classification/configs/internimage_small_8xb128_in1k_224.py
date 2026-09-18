# Converted from projects/internimage_classification/configs/internimage-small_8xb128_in1k-224.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ._base_ import *  # noqa: F401,F403

model.merge(dict(
    backbone=dict(
        stem_channels=80,
        drop_path_rate=0.4,
        stage_blocks=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        layer_scale=1e-5,
        post_norm=True),
    head=dict(in_channels=960)))
