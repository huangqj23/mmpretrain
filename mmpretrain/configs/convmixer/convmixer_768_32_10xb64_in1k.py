# Converted from configs/convmixer/convmixer-768-32_10xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.convmixer.convmixer_768_32 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_convmixer_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# schedule setting
optim_wrapper.merge(dict(
    optimizer=dict(lr=0.01),
    clip_grad=dict(max_norm=5.0),
))

train_cfg.merge(dict(by_epoch=True, max_epochs=300))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (10 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=640))
