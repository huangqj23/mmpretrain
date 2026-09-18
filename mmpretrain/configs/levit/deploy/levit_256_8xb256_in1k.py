# Converted from configs/levit/deploy/levit-256_8xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..levit_256_8xb256_in1k import *  # noqa: F401,F403

model.merge(dict(backbone=dict(deploy=True), head=dict(deploy=True)))
