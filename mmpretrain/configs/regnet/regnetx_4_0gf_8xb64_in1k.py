# Converted from configs/regnet/regnetx-4.0gf_8xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .regnetx_400mf_8xb128_in1k import *  # noqa: F401,F403

from mmpretrain.models import RegNet

# model settings
model.merge(dict(
    backbone=dict(type=RegNet, arch='regnetx_4.0gf'),
    head=dict(in_channels=1360, )))

# dataset settings
train_dataloader.merge(dict(batch_size=64))

# schedule settings
# for batch_size 512, use lr = 0.4
optim_wrapper.merge(dict(optimizer=dict(lr=0.4)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=512))
