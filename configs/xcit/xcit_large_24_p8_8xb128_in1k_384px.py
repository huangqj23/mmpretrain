# Converted from configs/xcit/xcit-large-24-p8_8xb128_in1k-384px.py by industrial-vision tools/convert_configs.py
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
        patch_size=8,
        embed_dims=768,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        layer_scale_init_value=1e-5,
        tokens_norm=True,
        out_type='cls_token',
    ),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0),
    ]),
)

# dataset settings
train_dataloader.merge(dict(batch_size=128))
