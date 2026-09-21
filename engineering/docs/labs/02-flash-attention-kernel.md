# Lab E2: The Minimal FlashAttention Kernel

## Objective
Implement a functional tiled multi-head attention kernel with online softmax in Triton or PyTorch C++, keeping intermediate attention matrices strictly within on-chip SRAM.

## Learning Milestones
1. Partition Query, Key, and Value tensors into SRAM-sized blocks.
2. Implement the online softmax normalizer algorithm ($m_i, d_i$).
3. Eliminate materialization of the $S \times S$ attention matrix in HBM.
4. Benchmark speedup and memory savings against naive PyTorch attention across sequence lengths from 512 to 16,384.

## Key Code Components
- SRAM block tiling loops.
- Online softmax accumulation logic.
- Triton kernel launch configuration.
