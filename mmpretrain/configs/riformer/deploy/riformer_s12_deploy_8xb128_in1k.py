# Converted from configs/riformer/deploy/riformer-s12-deploy_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..riformer_s12_8xb128_in1k import *  # noqa: F401,F403

model.merge(dict(backbone=dict(deploy=True)))
