# Converted from configs/minigpt4/minigpt-4_vicuna-7b_caption.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.datasets.coco_caption import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403

from mmcv.transforms import LoadImageFromFile, Resize
from mmengine.optim import CosineAnnealingLR
from torch.optim import AdamW

from mmpretrain.datasets import PackInputs
from mmpretrain.models import BEiTViT, MiniGPT4, Qformer


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

# dataset settings
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=Resize,
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type=PackInputs, meta_keys=['image_id']),
]

val_dataloader.merge(dict(batch_size=1, dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# model settings
model = dict(
    type=MiniGPT4,
    vision_encoder=dict(
        type=BEiTViT,
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=224,
        patch_size=14,
        layer_scale_init_value=0.0,
        frozen_stages=39,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw',
        pretrained=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_eva-g-p14_20230615-e908c021.pth'  # noqa
    ),
    q_former_model=dict(
        type=Qformer,
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32,
        pretrained=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_qformer_20230615-1dfa889c.pth'  # noqa
    ),
    lang_encoder=dict(
        type='AutoModelForCausalLM', name_or_path='YOUR_PATH_TO_VICUNA'),
    tokenizer=dict(type='LlamaTokenizer', name_or_path='YOUR_PATH_TO_VICUNA'),
    task='caption',
    prompt_template=dict([('en', '###Ask: {} ###Answer: '),
                          ('zh', '###问：{} ###答：')]),
    raw_prompts=dict([
        ('en', [('<Img><ImageHere></Img> '
                 'Describe this image in detail.'),
                ('<Img><ImageHere></Img> '
                 'Take a look at this image and describe what you notice.'),
                ('<Img><ImageHere></Img> '
                 'Please provide a detailed description of the picture.'),
                ('<Img><ImageHere></Img> '
                 'Could you describe the contents of this image for me?')]),
        ('zh', [('<Img><ImageHere></Img> '
                 '详细描述这张图片。'), ('<Img><ImageHere></Img> '
                                '浏览这张图片并描述你注意到什么。'),
                ('<Img><ImageHere></Img> '
                 '请对这张图片进行详细的描述。'),
                ('<Img><ImageHere></Img> '
                 '你能为我描述这张图片的内容吗？')])
    ]),
    max_txt_len=160,
    end_sym='###')

# schedule settings
optim_wrapper = dict(optimizer=dict(type=AdamW, lr=1e-5, weight_decay=0.05))

param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        by_epoch=True,
        begin=0,
        end=5,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=5)
val_cfg = dict()
test_cfg = dict()

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(test_pipeline, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    test_pipeline.pop('_delete_', None)
