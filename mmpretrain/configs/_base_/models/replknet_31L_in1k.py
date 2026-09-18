# Converted from configs/_base_/models/replknet-31L_in1k.py by industrial-vision tools/convert_configs.py
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepLKNet',
        arch='31L',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
