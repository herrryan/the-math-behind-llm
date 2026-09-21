# Lab E1: The Roofline Profiler

## Objective
Build a lightweight Python and PyTorch profiler that computes operational intensity ($I = \text{FLOPs}/\text{Byte}$) for arbitrary neural network layers and plots their execution against hardware GPU rooflines.

## Learning Milestones
1. Use PyTorch profiler hooks and hardware specifications to extract kernel runtime and memory traffic.
2. Calculate arithmetic intensity for RMSNorm, Softmax, and GEMM operations.
3. Automatically classify whether a layer is memory-bandwidth bound or compute-bound.
4. Export interactive Roofline charts illustrating the hardware inflection point $I^*$.

## Key Code Components
- Hardware specification database (H100, A100, RTX 4090).
- Operational intensity calculator.
- Roofline visualization generator.
