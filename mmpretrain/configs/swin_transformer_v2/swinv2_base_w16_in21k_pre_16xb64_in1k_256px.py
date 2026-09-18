# Converted from configs/swin_transformer_v2/swinv2-base-w16_in21k-pre_16xb64_in1k-256px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.swin_transformer_v2.base_256 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_swin_256 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmpretrain.models import ImageClassifier

model.merge(dict(
    type=ImageClassifier,
    backbone=dict(
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6])))
