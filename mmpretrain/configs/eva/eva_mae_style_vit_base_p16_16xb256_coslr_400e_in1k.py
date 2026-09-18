# Converted from configs/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mae_vit_base_p16 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs512_mae import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmengine.hooks import CheckpointHook
from mmengine.model import ConstantInit, XavierInit
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.models import (EVA, CLIPGenerator, CosineSimilarityLoss,
                               MAEPretrainDecoder, MIMHead)

# dataset settings
train_dataloader.merge(dict(batch_size=256))

# model settings
model.merge(dict(
    type=EVA,
    backbone=dict(init_cfg=[
        dict(type=XavierInit, distribution='uniform', layer='Linear'),
        dict(type=ConstantInit, layer='LayerNorm', val=1.0, bias=0.0)
    ]),
    neck=dict(
        type=MAEPretrainDecoder,
        predict_feature_dim=512,
        init_cfg=[
            dict(type=XavierInit, distribution='uniform', layer='Linear'),
            dict(type=ConstantInit, layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    head=dict(
        _delete_=True,
        type=MIMHead,
        loss=dict(
            type=CosineSimilarityLoss, shift_factor=2.0, scale_factor=2.0),
    ),
    target_generator=dict(
        type=CLIPGenerator,
        tokenizer_path=  # noqa
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
    ),
    init_cfg=None))

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))
find_unused_parameters = True

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=400)
default_hooks.merge(dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3)))

randomness.merge(dict(seed=0, diff_rank_seed=True))

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
if isinstance(resume, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    resume.pop('_delete_', None)
