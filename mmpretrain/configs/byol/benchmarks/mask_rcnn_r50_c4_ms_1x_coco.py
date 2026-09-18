# Converted from configs/byol/benchmarks/mask-rcnn_r50-c4_ms-1x_coco.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from mmdet.configs.mask_rcnn.mask_rcnn_r50_caffe_c4_1x_coco import *  # noqa: F401,F403

def __iv_scope(cfg, scope, has_scope=True):
    """复刻旧式跨库继承（scope::path）时 mmengine 给 base 变量加 _scope_ 的逻辑：
    每条路径上最外层带 type 的 dict 加 _scope_。"""
    if isinstance(cfg, dict):
        if has_scope and 'type' in cfg:
            has_scope = False
            if cfg.get('_scope_') is None:
                cfg['_scope_'] = scope
        for k in list(cfg):
            cfg[k] = __iv_scope(cfg[k], scope, has_scope)
    elif isinstance(cfg, list):
        cfg = [__iv_scope(v, scope, has_scope) for v in cfg]
    elif isinstance(cfg, tuple):
        cfg = tuple(__iv_scope(v, scope, has_scope) for v in cfg)
    return cfg

auto_scale_lr = __iv_scope(auto_scale_lr, 'mmdet')
backend_args = __iv_scope(backend_args, 'mmdet')
data_root = __iv_scope(data_root, 'mmdet')
dataset_type = __iv_scope(dataset_type, 'mmdet')
default_hooks = __iv_scope(default_hooks, 'mmdet')
default_scope = __iv_scope(default_scope, 'mmdet')
env_cfg = __iv_scope(env_cfg, 'mmdet')
load_from = __iv_scope(load_from, 'mmdet')
log_level = __iv_scope(log_level, 'mmdet')
log_processor = __iv_scope(log_processor, 'mmdet')
model = __iv_scope(model, 'mmdet')
norm_cfg = __iv_scope(norm_cfg, 'mmdet')
optim_wrapper = __iv_scope(optim_wrapper, 'mmdet')
param_scheduler = __iv_scope(param_scheduler, 'mmdet')
resume = __iv_scope(resume, 'mmdet')
test_cfg = __iv_scope(test_cfg, 'mmdet')
test_dataloader = __iv_scope(test_dataloader, 'mmdet')
test_evaluator = __iv_scope(test_evaluator, 'mmdet')
test_pipeline = __iv_scope(test_pipeline, 'mmdet')
train_cfg = __iv_scope(train_cfg, 'mmdet')
train_dataloader = __iv_scope(train_dataloader, 'mmdet')
train_pipeline = __iv_scope(train_pipeline, 'mmdet')
val_cfg = __iv_scope(val_cfg, 'mmdet')
val_dataloader = __iv_scope(val_dataloader, 'mmdet')
val_evaluator = __iv_scope(val_evaluator, 'mmdet')
vis_backends = __iv_scope(vis_backends, 'mmdet')
visualizer = __iv_scope(visualizer, 'mmdet')


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_model = model
__base_norm_cfg = norm_cfg
__base_train_cfg = train_cfg
__base_train_dataloader = train_dataloader
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm',
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch')))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

custom_imports = dict(
    imports=['mmpretrain.models.utils.res_layer_extra_norm'],
    allow_failed_imports=False)

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
norm_cfg = __iv_merge(__base_norm_cfg, norm_cfg)
model = __iv_merge(__base_model, model)
train_dataloader = __iv_merge(__base_train_dataloader, train_dataloader)
train_cfg = __iv_merge(__base_train_cfg, train_cfg)
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
