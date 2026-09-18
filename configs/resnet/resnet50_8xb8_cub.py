# Converted from configs/resnet/resnet50_8xb8_cub.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.resnet50 import *  # noqa: F401,F403
    from .._base_.datasets.cub_bs8_448 import *  # noqa: F401,F403
    from .._base_.schedules.cub_bs64 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import LoggerHook
from mmengine.model import PretrainedInit

from mmpretrain.models import ImageClassifier

# model settings
# use pre-train weight converted from https://github.com/Alibaba-MIIL/ImageNet21K # noqa
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

model.merge(dict(
    type=ImageClassifier,
    backbone=dict(
        init_cfg=dict(
            type=PretrainedInit, checkpoint=pretrained, prefix='backbone')),
    head=dict(num_classes=200, )))

# runtime settings
default_hooks.merge(dict(logger=dict(type=LoggerHook, interval=20)))
