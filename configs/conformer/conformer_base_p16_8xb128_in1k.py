# Converted from configs/conformer/conformer-base-p16_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.conformer.base_p16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_conformer import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

train_dataloader.merge(dict(batch_size=128))
