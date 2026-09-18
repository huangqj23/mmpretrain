# Converted from configs/mvit/mvitv2-base_8xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mvit.mvitv2_base import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset settings
train_dataloader.merge(dict(batch_size=256))
val_dataloader.merge(dict(batch_size=256))
test_dataloader.merge(dict(batch_size=256))

# schedule settings
optim_wrapper.merge(dict(
    optimizer=dict(lr=2.5e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.0),
            '.rel_pos_h': dict(decay_mult=0.0),
            '.rel_pos_w': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=1.0),
))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=70,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=70)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr.merge(dict(base_batch_size=2048))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
