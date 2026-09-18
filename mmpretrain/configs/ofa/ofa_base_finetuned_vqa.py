# Converted from configs/ofa/ofa-base_finetuned_vqa.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.datasets.coco_vqa import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, Resize

from mmpretrain.datasets import PackInputs
from mmpretrain.datasets.transforms.processing import OFAAddObjects
from mmpretrain.models import MultiModalDataPreprocessor


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_train_dataloader = train_dataloader

ANS2LABEL = 'https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/vqa_data/trainval_ans2label.pkl'  # noqa: E501

# model settings
model = dict(
    type='OFA',
    task='vqa',
    vocab_size=59457,
    embedding_dim=768,
    ans2label=ANS2LABEL,
    encoder_cfg=dict(
        embed_images=dict(type='OFAResNet', depth=101),
        num_layers=6,
        num_heads=12,
    ),
    decoder_cfg=dict(
        num_layers=6,
        num_heads=12,
    ),
    generation_cfg=dict(
        num_beams=5,
        max_new_tokens=200,
        length_penalty=0.,  # VQA doesn't require longer answer.
        use_cache=True,
    ),
    tokenizer=dict(type='OFATokenizer', name_or_path='OFA-Sys/OFA-base'),
)

# data settings
data_preprocessor.merge(dict(
    type=MultiModalDataPreprocessor,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
))

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=Resize,
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(type=OFAAddObjects),
    dict(
        type=PackInputs,
        algorithm_keys=[
            'question', 'gt_answer', 'gt_answer_weight', 'decoder_prompt'
        ],
        meta_keys=['question_id', 'image_id'],
    ),
]

train_dataloader = None  # Eval only
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule settings
train_cfg = None
val_cfg = dict()
test_cfg = dict()

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(train_dataloader, dict):
    train_dataloader = __iv_merge(__base_train_dataloader, train_dataloader)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
