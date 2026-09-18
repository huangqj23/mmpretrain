# Converted from configs/resnet/resnet50_32xb64-warmup-lbs_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .resnet50_32xb64_warmup_in1k import *  # noqa: F401,F403

from mmpretrain.models import LabelSmoothLoss, LinearClsHead

model.merge(dict(
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type=LabelSmoothLoss,
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    )))
