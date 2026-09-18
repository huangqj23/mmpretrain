# Converted from configs/repvgg/repvgg-D2se_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .repvgg_B3_8xb32_in1k import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, LinearLR


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_param_scheduler = param_scheduler

model.merge(dict(backbone=dict(arch='D2se'), head=dict(in_channels=2560)))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type=CosineAnnealingLR,
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]

train_cfg.merge(dict(by_epoch=True, max_epochs=300))

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3)))

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(param_scheduler, dict):
    param_scheduler = __iv_merge(__base_param_scheduler, param_scheduler)
