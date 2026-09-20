# Converted from configs/blip2/blip2-opt2.7b_8xb16_vqa.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.datasets.coco_vqa import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, Resize
from mmengine.optim import CosineAnnealingLR
from torch.optim import AdamW

from mmpretrain.datasets import CleanCaption, PackInputs, RandomResizedCrop
from mmpretrain.models import BEiTViT, Blip2VQA, LinearClsHead, Qformer


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
__base_train_dataloader = train_dataloader
__base_val_dataloader = val_dataloader

# model settings
model = dict(
    type=Blip2VQA,
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type=BEiTViT,
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    text_backbone=dict(
        type='OPTForCausalLM', name_or_path='facebook/opt-2.7b'),
    multimodal_backbone=dict(
        type=Qformer,
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type=LinearClsHead,
        in_channels=768,
        num_classes=2560,
    ),
    prompt='Question: {} Answer:',
    max_txt_len=10)

# data settings
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=224),
    dict(
        type=PackInputs,
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=Resize,
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type=CleanCaption,
        keys=['question'],
    ),
    dict(
        type=PackInputs,
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(optimizer=dict(type=AdamW, lr=1e-5, weight_decay=0.05))

param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        by_epoch=True,
        begin=0,
        end=10,
    )
]

train_cfg = dict(max_epochs=10)
val_cfg = dict()
test_cfg = dict()

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
train_dataloader = __iv_merge(__base_train_dataloader, train_dataloader)
val_dataloader = __iv_merge(__base_val_dataloader, val_dataloader)
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
