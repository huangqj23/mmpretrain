# Converted from configs/_base_/models/efficientnet_v2/efficientnetv2_b0.py by industrial-vision tools/convert_configs.py
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='b0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
