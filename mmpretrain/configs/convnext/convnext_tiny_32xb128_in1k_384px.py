# Converted from configs/convnext/convnext-tiny_32xb128_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.convnext.convnext_tiny import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset setting
train_dataloader.merge(dict(batch_size=128))

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
))

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=4096))
