# Converted from configs/_base_/schedules/imagenet_sgd_steplr_100e.py by industrial-vision tools/convert_configs.py
from mmengine.optim import MultiStepLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import SGD

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=1e-4))

# learning rate scheduler
param_scheduler = [
    dict(type=MultiStepLR, by_epoch=True, milestones=[60, 80], gamma=0.1)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=100)
val_cfg = dict()
test_cfg = dict()
