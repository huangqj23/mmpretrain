# Converted from configs/simmim/simmim_swin-base-w6_16xb128-amp-coslr-100e_in1k-192px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .simmim_swin_base_w6_8xb256_amp_coslr_100e_in1k_192px import *  # noqa: F401,F403

# dataset 16 GPUs x 128
train_dataloader.merge(dict(batch_size=128))
