# Converted from configs/byol/benchmarks/mask-rcnn_r50_fpn_ms-1x_coco.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from mmdet.configs.mask_rcnn.mask_rcnn_r50_fpn_1x_coco import *  # noqa: F401,F403

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
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py

norm_cfg = dict(type='SyncBN', requires_grad=True)
model.merge(dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg))))

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

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
if isinstance(train_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    train_pipeline.pop('_delete_', None)
