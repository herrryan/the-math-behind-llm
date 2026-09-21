# Lab E4: 2-GPU Tensor Parallelism Engine

## Objective
Implement distributed Megatron-style column-parallel and row-parallel linear layers from scratch using PyTorch `torch.distributed` collective communication primitives.

## Learning Milestones
1. Initialize a 2-rank distributed process group.
2. Construct a `ColumnParallelLinear` layer that partitions weights along output channels.
3. Construct a `RowParallelLinear` layer with manual `all_reduce` collective summation.
4. Assemble a complete 2-GPU parallel MLP and verify numerical parity with a standard single-device layer.

## Key Code Components
- Distributed rank initialization.
- ColumnParallelLinear and RowParallelLinear modules.
- Verification script comparing single-GPU vs. 2-GPU numerical outputs.
