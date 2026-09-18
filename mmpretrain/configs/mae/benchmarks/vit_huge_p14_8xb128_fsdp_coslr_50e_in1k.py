# Converted from configs/mae/benchmarks/vit-huge-p14_8xb128-fsdp-coslr-50e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .vit_huge_p14_8xb128_coslr_50e_in1k import *  # noqa: F401,F403

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.size_based_auto_wrap_policy',
            min_num_params=1e7)))

optim_wrapper.merge(dict(type='AmpOptimWrapper'))

# runner which supports strategies
runner_type = 'FlexibleRunner'
