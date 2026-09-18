# Converted from configs/_base_/models/res2net50-w26-s6.py by industrial-vision tools/convert_configs.py
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=6,
        base_width=26,
        deep_stem=False,
        avg_down=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
