# Converted from configs/_base_/models/mobileone/mobileone_s1.py by industrial-vision tools/convert_configs.py
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s1',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
    ))
