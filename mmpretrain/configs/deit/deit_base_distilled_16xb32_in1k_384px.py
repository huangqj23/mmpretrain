# Converted from configs/deit/deit-base-distilled_16xb32_in1k-384px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_384 import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs4096_AdamW import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DistilledVisionTransformer',
        arch='deit-base',
        img_size=384,
        patch_size=16,
    ),
    neck=None,
    head=dict(
        type='DeiTClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    # Change to the path of the pretrained model
    # init_cfg=dict(type='Pretrained', checkpoint=''),
)

# dataset settings
train_dataloader.merge(dict(batch_size=32))

# schedule settings
optim_wrapper.merge(dict(clip_grad=dict(max_norm=1.0)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (16 GPUs) x (32 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=512))
