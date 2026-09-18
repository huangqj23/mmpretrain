# Converted from configs/xcit/xcit-tiny-24-p16_8xb128_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmpretrain.models import (CrossEntropyLoss, CutMix, ImageClassifier,
                               LinearClsHead, Mixup, XCiT)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=XCiT,
        patch_size=16,
        embed_dims=192,
        depth=24,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        layer_scale_init_value=1e-5,
        tokens_norm=True,
        out_type='cls_token',
    ),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=192,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0),
    ]),
)

# dataset settings
train_dataloader.merge(dict(batch_size=128))
