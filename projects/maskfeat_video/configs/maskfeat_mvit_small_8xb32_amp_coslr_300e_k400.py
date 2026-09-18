# Converted from projects/maskfeat_video/configs/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .maskfeat_mvit_small_16xb32_amp_coslr_300e_k400 import *  # noqa: F401,F403

from torch.optim import AdamW

optim_wrapper.merge(dict(
    optimizer=dict(
        type=AdamW, lr=8e-4, betas=(0.9, 0.999), weight_decay=0.05)))
