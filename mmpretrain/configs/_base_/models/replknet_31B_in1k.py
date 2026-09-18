# Converted from configs/_base_/models/replknet-31B_in1k.py by industrial-vision tools/convert_configs.py
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, RepLKNet,
                               build_classifier)

model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=RepLKNet,
        arch='31B',
        out_indices=(3, ),
    ),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, 5),
    ))

if __name__ == '__main__':
    # model.pop('type')
    model = build_classifier(model)
    model.eval()
    print('------------------- training-time model -------------')
    for i in model.state_dict().keys():
        print(i)
