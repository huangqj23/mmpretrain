# Converted from configs/tnt/tnt-s-p16_16xb64_in1k.py by industrial-vision tools/convert_configs.py
# accuracy_top-1 : 81.52 accuracy_top-5 : 95.73
from mmengine.config import read_base

with read_base():
    from .._base_.models.tnt_s_patch16_224 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import CenterCrop, LoadImageFromFile
from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from mmpretrain.datasets import PackInputs, ResizeEdge

# dataset settings
data_preprocessor.merge(dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
))

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=248,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(batch_size=64))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule settings
optim_wrapper = dict(optimizer=dict(type=AdamW, lr=1e-3, weight_decay=0.05))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type=CosineAnnealingLR, T_max=295, by_epoch=True, begin=5, end=300)
]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (16 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
