# Copyright (c) OpenMMLab. All rights reserved.
"""``type`` written as a class must behave exactly like the registered name."""
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.transforms import LoadImageFromFile, Resize
from mmengine.config import Config
from mmengine.model.weight_init import PretrainedInit
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmpretrain.apis import NLVRInferencer
from mmpretrain.datasets.transforms import (ApplyToList, AutoAugment,
                                            AutoContrast, PackInputs,
                                            Posterize, RandAugment, Rotate)
from mmpretrain.datasets.transforms.utils import get_transform_idx
from mmpretrain.models.backbones import (MobileOne, RepVGG, ResNet,
                                         SparseResNet, VisionTransformer, XCiT)
from mmpretrain.models.utils import LayerNorm2d, build_norm_layer
from mmpretrain.models.utils.sparse_modules import (SparseBatchNorm2d,
                                                    SparseSyncBatchNorm2d)
from mmpretrain.utils import register_all_modules

register_all_modules()


def _built_same(make, a_arg, b_arg, seed=0):
    torch.manual_seed(seed)
    a = make(a_arg)
    torch.manual_seed(seed)
    b = make(b_arg)
    assert repr(a) == repr(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    return a, b


@pytest.mark.parametrize('s,c,extra', [
    ('BN', nn.BatchNorm2d, {}),
    ('BN1d', nn.BatchNorm1d, {}),
    ('GN', nn.GroupNorm, dict(num_groups=4)),
    ('LN', nn.LayerNorm, {}),
    ('LN2d', LayerNorm2d, {}),
    ('IN', nn.InstanceNorm2d, {}),
    ('SyncBN', SyncBatchNorm, {}),
])
def test_build_norm_layer(s, c, extra):
    a, b = _built_same(lambda t: build_norm_layer(dict(type=t, **extra), 8),
                       s, c)
    assert type(a) is type(b)


def _load_back(make, t, perturb=True):
    """Save ``make(None)`` and load it back via ``init_cfg`` of type ``t``."""
    torch.manual_seed(0)
    src = make(None)
    src.init_weights()
    if perturb:
        with torch.no_grad():
            for p in src.parameters():
                p.uniform_(0.5, 1.5)
    with tempfile.TemporaryDirectory() as d:
        ckpt = osp.join(d, 'src.pth')
        torch.save(src.state_dict(), ckpt)
        dst = make(dict(type=t, checkpoint=ckpt))
        dst.init_weights()
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


@pytest.mark.parametrize('t', ['Pretrained', PretrainedInit])
def test_resnet_pretrained_suppresses_zero_init(t):
    # With a pretrained init_cfg, ``zero_init_residual`` must not zero the
    # loaded BN weights afterwards.
    _load_back(
        lambda init: ResNet(depth=18, zero_init_residual=init is not None,
                            init_cfg=init), t)


@pytest.mark.parametrize('t', ['Pretrained', PretrainedInit])
def test_vit_pretrained_keeps_pos_embed(t):
    arch = dict(embed_dims=32, num_layers=1, num_heads=2,
                feedforward_channels=64)
    _load_back(
        lambda init: VisionTransformer(arch=arch, img_size=32, patch_size=16,
                                       init_cfg=init), t)


@pytest.mark.parametrize('t', ['Pretrained', PretrainedInit])
def test_xcit_pretrained_skips_reinit(t):
    _load_back(
        lambda init: XCiT(img_size=32, patch_size=16, embed_dims=32, depth=1,
                          cls_attn_layers=1, num_heads=2,
                          **(dict(init_cfg=init) if init else {})), t)


@pytest.mark.parametrize('make', [
    lambda t: MobileOne('s0', norm_cfg=dict(type=t)),
    lambda t: RepVGG('A0', norm_cfg=dict(type=t)),
])
def test_reparam_switch_to_deploy(make):
    a, b = _built_same(make, 'BN', nn.BatchNorm2d)
    x = torch.randn(1, 3, 32, 32)
    for m in (a, b):
        m.eval()
        m.switch_to_deploy()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    for oa, ob in zip(a(x), b(x)):
        assert torch.equal(oa, ob)


@pytest.mark.parametrize('t', ['SyncBN', SyncBatchNorm])
def test_reparam_rejects_non_bn(t):
    model = RepVGG('A0', norm_cfg=dict(type=t))
    with pytest.raises(AssertionError):
        model.switch_to_deploy()


@pytest.mark.parametrize('s,c,sparse_bn', [
    ('SparseSyncBatchNorm2d', SparseSyncBatchNorm2d, SparseSyncBatchNorm2d),
    ('SparseBatchNorm2d', SparseBatchNorm2d, SparseBatchNorm2d),
    ('SyncBN', SyncBatchNorm, SparseSyncBatchNorm2d),
    ('BN', nn.BatchNorm2d, SparseBatchNorm2d),
])
def test_sparse_resnet_sync_bn(s, c, sparse_bn):
    a, b = _built_same(lambda t: SparseResNet(depth=18, norm_cfg=dict(type=t)),
                       s, c)
    assert type(b.norm1) is sparse_bn


STR = dict(Posterize='Posterize', Rotate='Rotate', AutoContrast='AutoContrast')
CLS = dict(Posterize=Posterize, Rotate=Rotate, AutoContrast=AutoContrast)


def _run_same(a, b, seeds=range(8)):
    img = np.random.RandomState(0).randint(0, 256, (32, 32, 3), np.uint8)
    for seed in seeds:
        np.random.seed(seed)
        ra = a(dict(img=img.copy()))
        np.random.seed(seed)
        rb = b(dict(img=img.copy()))
        np.testing.assert_array_equal(ra['img'], rb['img'])


def test_auto_augment_class_policies():

    def policies(t):
        return [[
            dict(type=t['Posterize'], bits=4, prob=0.6),
            dict(type=t['Rotate'], angle=30., prob=0.6)
        ], [dict(type=t['AutoContrast'], prob=0.6)]]

    a, b = AutoAugment(policies(STR)), AutoAugment(policies(CLS))
    assert repr(a) == repr(b)
    _run_same(a, b)


def test_rand_augment_class_policies():

    def policies(t):
        return [
            dict(type=t['Rotate'], magnitude_range=(0, 30)),
            dict(type=t['AutoContrast']),
            dict(type=t['Posterize'], magnitude_range=(4, 0)),
        ]

    kw = dict(num_policies=2, magnitude_level=9)
    a = RandAugment(policies=policies(STR), **kw)
    b = RandAugment(policies=policies(CLS), **kw)
    assert repr(a) == repr(b)
    _run_same(a, b)


@pytest.mark.parametrize('t', ['ApplyToList', ApplyToList])
def test_nlvr_init_pipeline(t):
    cfg = Config(
        dict(
            test_dataloader=dict(
                dataset=dict(pipeline=[
                    dict(
                        type=t,
                        scatter_key='img_path',
                        transforms=[
                            dict(type=LoadImageFromFile),
                            dict(type=Resize, scale=(32, 32)),
                        ],
                        collate_keys=['img', 'scale_factor', 'ori_shape']),
                    dict(type=PackInputs, algorithm_keys=['text']),
                ]))))
    pipeline = NLVRInferencer._init_pipeline(None, cfg)
    apply_to_list = pipeline.transforms[0]
    assert isinstance(apply_to_list, ApplyToList)
    assert apply_to_list.scatter_key == 'img'
    # ``LoadImageFromFile`` has been removed
    assert len(apply_to_list.transforms.transforms) == 1


def test_get_transform_idx():
    assert get_transform_idx([dict(type='Resize'),
                              dict(type=LoadImageFromFile)],
                             'LoadImageFromFile') == 1
    assert get_transform_idx([dict(type=Resize)], 'LoadImageFromFile') == -1
