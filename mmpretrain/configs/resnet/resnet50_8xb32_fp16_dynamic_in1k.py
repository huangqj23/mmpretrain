# Converted from configs/resnet/resnet50_8xb32-fp16-dynamic_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .resnet50_8xb32_in1k import *  # noqa: F401,F403

# schedule settings
optim_wrapper.merge(dict(type='AmpOptimWrapper', loss_scale='dynamic'))
