# Converted from configs/replknet/deploy/replknet-XL-deploy_32xb64_in1k-320px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..replknet_XL_32xb64_in1k_320px import *  # noqa: F401,F403

model.merge(dict(backbone=dict(small_kernel_merged=True)))
