# Converted from configs/_base_/schedules/imagenet_sgd_coslr_200e.py by industrial-vision tools/convert_configs.py
from mmengine.optim import CosineAnnealingLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import SGD

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.03, weight_decay=1e-4, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(type=CosineAnnealingLR, T_max=200, by_epoch=True, begin=0, end=200)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=200)
