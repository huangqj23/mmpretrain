# Converted from configs/vision_transformer/vit-base-p16_8xb64-lora_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model setting
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='LoRAModel',
        module=dict(
            type='VisionTransformer',
            arch='b',
            img_size=384,
            patch_size=16,
            drop_rate=0.1,
            init_cfg=dict(type='Pretrained', checkpoint='',
                          prefix='backbone')),
        alpha=16,
        rank=16,
        drop_rate=0.1,
        targets=[dict(type='qkv')]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)],
    ))

# dataset setting
data_preprocessor.merge(dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=384, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg.merge(dict(by_epoch=True, max_epochs=50))
default_hooks.merge(dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)))

# schedule setting
optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
