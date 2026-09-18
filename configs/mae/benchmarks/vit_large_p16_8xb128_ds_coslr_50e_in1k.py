# Converted from configs/mae/benchmarks/vit-large-p16_8xb128-ds-coslr-50e_in1k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .vit_large_p16_8xb128_coslr_50e_in1k import *  # noqa: F401,F403

from mmengine._strategy import DeepSpeedStrategy
from mmengine._strategy.deepspeed import DeepSpeedOptimWrapper

# optimizer wrapper
optim_wrapper.merge(dict(type=DeepSpeedOptimWrapper))

# training strategy
strategy = dict(
    type=DeepSpeedStrategy,
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=1,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False,
    ))

# runner which supports strategies
runner_type = 'FlexibleRunner'
