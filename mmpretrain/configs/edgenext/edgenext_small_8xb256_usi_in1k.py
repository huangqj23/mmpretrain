# Converted from configs/edgenext/edgenext-small_8xb256-usi_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .edgenext_small_8xb256_in1k import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# dataset setting

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=269,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs')
]

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
