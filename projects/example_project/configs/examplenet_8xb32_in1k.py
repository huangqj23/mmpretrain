# Converted from projects/example_project/configs/examplenet_8xb32_in1k.py by industrial-vision tools/convert_configs.py
# Directly inherit the entire recipe you want to use.
from mmengine.config import read_base

with read_base():
    from mmpretrain.configs.resnet.resnet50_8xb32_in1k import *  # noqa: F401,F403

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

auto_scale_lr = __iv_scope(auto_scale_lr, 'mmpretrain')
data_preprocessor = __iv_scope(data_preprocessor, 'mmpretrain')
dataset_type = __iv_scope(dataset_type, 'mmpretrain')
default_hooks = __iv_scope(default_hooks, 'mmpretrain')
default_scope = __iv_scope(default_scope, 'mmpretrain')
env_cfg = __iv_scope(env_cfg, 'mmpretrain')
load_from = __iv_scope(load_from, 'mmpretrain')
log_level = __iv_scope(log_level, 'mmpretrain')
model = __iv_scope(model, 'mmpretrain')
optim_wrapper = __iv_scope(optim_wrapper, 'mmpretrain')
param_scheduler = __iv_scope(param_scheduler, 'mmpretrain')
randomness = __iv_scope(randomness, 'mmpretrain')
resume = __iv_scope(resume, 'mmpretrain')
test_cfg = __iv_scope(test_cfg, 'mmpretrain')
test_dataloader = __iv_scope(test_dataloader, 'mmpretrain')
test_evaluator = __iv_scope(test_evaluator, 'mmpretrain')
test_pipeline = __iv_scope(test_pipeline, 'mmpretrain')
train_cfg = __iv_scope(train_cfg, 'mmpretrain')
train_dataloader = __iv_scope(train_dataloader, 'mmpretrain')
train_pipeline = __iv_scope(train_pipeline, 'mmpretrain')
val_cfg = __iv_scope(val_cfg, 'mmpretrain')
val_dataloader = __iv_scope(val_dataloader, 'mmpretrain')
val_evaluator = __iv_scope(val_evaluator, 'mmpretrain')
vis_backends = __iv_scope(vis_backends, 'mmpretrain')
visualizer = __iv_scope(visualizer, 'mmpretrain')

# This line is to import your own modules.
custom_imports = dict(imports='models')

# Modify the backbone to use your own backbone.
model['backbone'] = dict(type='ExampleNet', depth=18)
# Modify the in_channels of classifier head to fit your backbone.
model['head']['in_channels'] = 512
