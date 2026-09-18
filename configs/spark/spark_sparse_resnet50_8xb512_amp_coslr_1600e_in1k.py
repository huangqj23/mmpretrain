# Converted from configs/spark/spark_sparse-resnet50_8xb512-amp-coslr-1600e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .spark_sparse_resnet50_8xb512_amp_coslr_800e_in1k import *  # noqa: F401,F403

from mmengine.optim import CosineAnnealingLR, LinearLR

from mmpretrain.engine import CosineAnnealingWeightDecay

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=1560,
        by_epoch=True,
        begin=40,
        end=1600,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingWeightDecay,
        eta_min=0.2,
        T_max=1600,
        by_epoch=True,
        begin=0,
        end=1600,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg.merge(dict(max_epochs=1600))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
