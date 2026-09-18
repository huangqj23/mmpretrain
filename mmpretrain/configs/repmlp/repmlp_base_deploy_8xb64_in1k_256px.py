# Converted from configs/repmlp/repmlp-base_deploy_8xb64_in1k-256px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .repmlp_base_8xb64_in1k_256px import *  # noqa: F401,F403

model.merge(dict(backbone=dict(deploy=True)))
