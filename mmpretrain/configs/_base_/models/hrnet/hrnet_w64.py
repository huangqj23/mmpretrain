# Converted from configs/_base_/models/hrnet/hrnet-w64.py by industrial-vision tools/convert_configs.py
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w64'),
    neck=[
        dict(type='HRFuseScales', in_channels=(64, 128, 256, 512)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=1000,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
