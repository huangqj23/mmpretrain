# Converted from configs/vig/pvig-base_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.vig.pyramid_vig_base import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs128_vig_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=235,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
