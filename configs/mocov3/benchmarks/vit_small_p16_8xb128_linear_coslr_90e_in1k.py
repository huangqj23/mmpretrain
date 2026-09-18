# Converted from configs/mocov3/benchmarks/vit-small-p16_8xb128-linear-coslr-90e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from ..._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import NormalInit, PretrainedInit
from mmengine.optim import CosineAnnealingLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import SGD

from mmpretrain.models import (CrossEntropyLoss, ImageClassifier, MoCoV3ViT,
                               VisionTransformerClsHead)

# dataset settings
train_dataloader.merge(dict(batch_size=128))

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=MoCoV3ViT,
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True,
        frozen_stages=12,
        norm_eval=True,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.')),
    head=dict(
        type=VisionTransformerClsHead,
        num_classes=1000,
        in_channels=384,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        init_cfg=dict(type=NormalInit, std=0.01, layer='Linear'),
    ))

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=12, momentum=0.9, weight_decay=0.))

# learning rate scheduler
param_scheduler = [
    dict(type=CosineAnnealingLR, T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=90)
val_cfg = dict()
test_cfg = dict()

default_hooks.merge(dict(
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
