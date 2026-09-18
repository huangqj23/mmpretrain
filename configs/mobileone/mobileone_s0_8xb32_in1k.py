# Converted from configs/mobileone/mobileone-s0_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobileone.mobileone_s0 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_coslr_coswd_300e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmpretrain.engine import EMAHook

# schedule settings
optim_wrapper.merge(dict(paramwise_cfg=dict(norm_decay_mult=0.)))

val_dataloader.merge(dict(batch_size=256))
test_dataloader.merge(dict(batch_size=256))

custom_hooks = [
    dict(
        type=EMAHook,
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]
