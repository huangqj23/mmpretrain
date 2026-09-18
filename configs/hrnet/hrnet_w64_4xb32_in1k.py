# Converted from configs/hrnet/hrnet-w64_4xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.hrnet.hrnet_w64 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_coslr import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (32 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=128))
