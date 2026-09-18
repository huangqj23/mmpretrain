# Converted from configs/swin_transformer_v2/swinv2-large-w12_8xb128_in21k-192px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.swin_transformer_v2.base_256 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet21k_bs128 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model.merge(dict(
    backbone=dict(img_size=192, window_size=[12, 12, 12, 6]),
    head=dict(num_classes=21841),
))

# dataset settings
data_preprocessor.merge(dict(num_classes=21841))

_base_['train_pipeline'][1]['scale'] = 192  # RandomResizedCrop
_base_['test_pipeline'][1]['scale'] = 219  # ResizeEdge
_base_['test_pipeline'][2]['crop_size'] = 192  # CenterCrop
