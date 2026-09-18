# Converted from configs/csra/resnet101-csra_1xb16_voc07-448px.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.datasets.voc_bs16 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, RandomFlip, Resize
from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, StepLR
from torch.optim import SGD

from mmpretrain.datasets import PackInputs, RandomResizedCrop
from mmpretrain.models import (CrossEntropyLoss, CSRAClsHead, ImageClassifier,
                               ResNet)


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# Pre-trained Checkpoint Path
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth'  # noqa
# If you want to use the pre-trained weight of ResNet101-CutMix from the
# originary repo(https://github.com/Kevinz-code/CSRA). Script of
# 'tools/model_converters/torchvision_to_mmpretrain.py' can help you convert
# weight into mmpretrain format. The mAP result would hit 95.5 by using the
# weight. checkpoint = 'PATH/TO/PRE-TRAINED_WEIGHT'

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet,
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type=PretrainedInit, checkpoint=checkpoint, prefix='backbone')),
    neck=None,
    head=dict(
        type=CSRAClsHead,
        num_classes=20,
        in_channels=2048,
        num_heads=1,
        lam=0.1,
        loss=dict(type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0)))

# dataset setting
data_preprocessor.merge(dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255]))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=448, crop_ratio_range=(0.7, 1.0)),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=448),
    dict(
        type=PackInputs,
        # `gt_label_difficult` is needed for VOC evaluation
        meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'gt_label_difficult')),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# optimizer
# the lr of classifier.head is 10 * base_lr, which help convergence.
optim_wrapper = dict(
    optimizer=dict(type=SGD, lr=0.0002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10)}))

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-7,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(type=StepLR, by_epoch=True, step_size=6, gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
