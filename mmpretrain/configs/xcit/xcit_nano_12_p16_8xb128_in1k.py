# Converted from configs/xcit/xcit-nano-12-p16_8xb128_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dims=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        layer_scale_init_value=1.0,
        tokens_norm=False,
        out_type='cls_token',
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)

# dataset settings
train_dataloader.merge(dict(batch_size=128))
