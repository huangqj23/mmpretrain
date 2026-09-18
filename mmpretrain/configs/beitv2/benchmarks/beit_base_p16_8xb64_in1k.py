# Converted from configs/beitv2/benchmarks/beit-base-p16_8xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from ..._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (BEiTViT, CutMix, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead, Mixup)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=BEiTViT,
        arch='base',
        img_size=224,
        patch_size=16,
        out_type='avg_featmap',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
    ),
    neck=None,
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
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
    ]))
