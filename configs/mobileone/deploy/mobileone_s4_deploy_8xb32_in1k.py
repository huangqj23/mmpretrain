# Converted from configs/mobileone/deploy/mobileone-s4_deploy_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..mobileone_s4_8xb32_in1k import *  # noqa: F401,F403

model.merge(dict(backbone=dict(deploy=True)))
