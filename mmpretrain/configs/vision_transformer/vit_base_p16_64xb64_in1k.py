# Converted from configs/vision_transformer/vit-base-p16_64xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.vit_base_p16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_pil_resize_autoaug import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model setting
model.merge(dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
))

# schedule setting
optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))
