# Converted from configs/repmlp/repmlp-base_8xb64_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.repmlp_base_224 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs64_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # resizing to (256, 256) here, different from resizing shorter edge to 256
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))

# schedule settings
optim_wrapper.merge(dict(clip_grad=dict(max_norm=5.0)))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=512))
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
