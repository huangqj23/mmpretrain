# Converted from configs/swin_transformer_v2/swinv2-base-w24_in21k-pre_16xb64_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.swin_transformer_v2.base_384 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmpretrain.models import ImageClassifier

model.merge(dict(
    type=ImageClassifier,
    backbone=dict(
        img_size=384,
        window_size=[24, 24, 24, 12],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6])))
