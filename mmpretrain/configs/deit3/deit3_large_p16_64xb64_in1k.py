# Converted from configs/deit3/deit3-large-p16_64xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.deit3.deit3_large_p16_224 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_deit3_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset setting
train_dataloader.merge(dict(batch_size=64))

# schedule settings
optim_wrapper.merge(dict(optimizer=dict(lr=1e-5, weight_decay=0.1)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=4096))
