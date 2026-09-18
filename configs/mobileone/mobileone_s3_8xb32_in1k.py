# Converted from configs/mobileone/mobileone-s3_8xb32_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobileone.mobileone_s3 import *  # noqa: F401,F403
    from .._base_.datasets.imagenet_bs32_pil_resize import *  # noqa: F401,F403
    from .._base_.schedules.imagenet_bs256_coslr_coswd_300e import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

# schedule settings
optim_wrapper.merge(dict(paramwise_cfg=dict(norm_decay_mult=0.)))

val_dataloader.merge(dict(batch_size=256))
test_dataloader.merge(dict(batch_size=256))

import copy  # noqa: E402

from mmcv.transforms import LoadImageFromFile, RandomFlip

from mmpretrain.datasets import PackInputs, RandAugment, RandomResizedCrop
from mmpretrain.engine import EMAHook, SwitchRecipeHook

bgr_mean = data_preprocessor['mean'][::-1]
base_train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=224, backend='pillow'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(
        type=RandAugment,
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type=PackInputs)
]

# modify start epoch RandomResizedCrop.scale to 160
# and RA.magnitude_level * 0.3
train_pipeline_1e = copy.deepcopy(base_train_pipeline)
train_pipeline_1e[1]['scale'] = 160
train_pipeline_1e[3]['magnitude_level'] *= 0.3
train_dataloader.dataset.pipeline = train_pipeline_1e

import copy  # noqa: E402

# modify 137 epoch's RandomResizedCrop.scale to 192
# and RA.magnitude_level * 0.7
train_pipeline_37e = copy.deepcopy(base_train_pipeline)
train_pipeline_37e[1]['scale'] = 192
train_pipeline_37e[3]['magnitude_level'] *= 0.7

# modify 112 epoch's RandomResizedCrop.scale to 224
# and RA.magnitude_level * 1.0
train_pipeline_112e = copy.deepcopy(base_train_pipeline)
train_pipeline_112e[1]['scale'] = 224
train_pipeline_112e[3]['magnitude_level'] *= 1.0

custom_hooks = [
    dict(
        type=SwitchRecipeHook,
        schedule=[
            dict(action_epoch=37, pipeline=train_pipeline_37e),
            dict(action_epoch=112, pipeline=train_pipeline_112e),
        ]),
    dict(
        type=EMAHook,
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]

# 旧式加载不会把模块/函数放进配置，lazy 模式会，这里清掉以保持等价
del copy
