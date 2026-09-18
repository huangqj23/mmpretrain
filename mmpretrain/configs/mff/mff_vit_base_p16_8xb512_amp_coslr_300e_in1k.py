# Converted from configs/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..mae.mae_vit_base_p16_8xb512_amp_coslr_300e_in1k import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile

from mmpretrain.datasets import NumpyToPIL, PackInputs, PILToNumpy
from mmpretrain.models import MFF, MFFViT

randomness.merge(dict(seed=2, diff_rank_seed=True))

# dataset config
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=NumpyToPIL, to_rgb=True),
    dict(type='torchvision/Resize', size=224),
    dict(
        type='torchvision/RandomCrop',
        size=224,
        padding=4,
        padding_mode='reflect'),
    dict(type='torchvision/RandomHorizontalFlip', p=0.5),
    dict(type=PILToNumpy, to_bgr=True),
    dict(type=PackInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

# model config
model.merge(dict(
    type=MFF, backbone=dict(type=MFFViT, out_indices=[0, 2, 4, 6, 8, 11])))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
