# Converted from configs/deit/deit-small-distilled_4xb256_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, DeiTClsHead, DistilledVisionTransformer,
                               ImageClassifier, LabelSmoothLoss, Mixup)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=DistilledVisionTransformer,
        arch='deit-small',
        img_size=224,
        patch_size=16),
    neck=None,
    head=dict(
        type=DeiTClsHead,
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=.02),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0)
    ]),
)

# data settings
train_dataloader.merge(dict(batch_size=256))

# schedule settings
optim_wrapper.merge(dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
))
