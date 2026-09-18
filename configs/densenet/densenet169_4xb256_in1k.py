# Converted from configs/densenet/densenet169_4xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.densenet.densenet169 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset settings
train_dataloader.merge(dict(batch_size=256))

# schedule settings
train_cfg.merge(dict(by_epoch=True, max_epochs=90))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (256 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=1024))
