# Converted from configs/_base_/models/seresnet101.py by industrial-vision tools/convert_configs.py
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
