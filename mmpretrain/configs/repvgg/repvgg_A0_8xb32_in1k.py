# Converted from configs/repvgg/repvgg-A0_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.repvgg_A0_in1k import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_coslr import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR

val_dataloader.merge(dict(batch_size=256))
test_dataloader.merge(dict(batch_size=256))

# schedule settings
optim_wrapper.merge(dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys={
            'branch_3x3.norm': dict(decay_mult=0.0),
            'branch_1x1.norm': dict(decay_mult=0.0),
            'branch_norm.bias': dict(decay_mult=0.0),
        })))

# schedule settings
param_scheduler.merge(dict(
    type=CosineAnnealingLR,
    T_max=120,
    by_epoch=True,
    begin=0,
    end=120,
    convert_to_iter_based=True))

train_cfg.merge(dict(by_epoch=True, max_epochs=120))

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3)))
