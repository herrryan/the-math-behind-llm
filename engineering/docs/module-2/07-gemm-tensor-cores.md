# Chapter E07: GEMM at Scale: Tensor Cores, Systolic Arrays, and Tiling

## Step 1: Hardware Intuition
Imagine multiplying two 1000-page accounting ledgers by hand. If you proceed line by line, calculating one number at a time, you will spend weeks flipping pages back and forth. Instead, a systolic array is like an automated stamping press: numbers flow like rhythmic pulses across a grid of tiny stamping gears, and intermediate results accumulate without ever leaving the press.

## Step 2: Silicon Micro-Mechanics
- **Tensor Core Architecture:** Hardware matrix-multiply-accumulate (MMA) instructions ($16 \times 16 \times 16$ micro-tiles per cycle).
- **Hierarchical Tiling:** Threadblock tiling in SRAM, warp-level tiling in registers, and thread-level accumulation.
- **Dimension Quantization:** Matrix dimensions $(M, N, K)$ must align with multiples of 64 or 128 elements to avoid unaligned memory access and uncoalesced memory stalls.

## Step 3: Cross-Component Coupling
- **Vocabulary & Embedding Alignment:** If vocabulary size $|V|$ is not a multiple of 64 or 128 (e.g. 32,001), the final unembedding projection GEMM drops in arithmetic efficiency. Padding $|V|$ to 32,064 restores peak Tensor Core performance.
- **Batching Effects:** Larger batch sizes during prefill maximize GEMM tile occupancy.

## Step 4: The Exact Performance Formula
$$\text{GEMM FLOPs} = 2 M N K$$
$$\text{Tensor Core Efficiency } \eta_{\text{GEMM}} = \frac{2 M N K}{T_{\text{exec}} \times P_{\text{peak}}}$$

## Step 5: Concrete Benchmark Walkthrough
Benchmarking a projection matrix ($M=4096, K=4096, N=11008$) on an H100 GPU:
- FLOPs: $2 \times 4096 \times 4096 \times 11008 = 3.69 \times 10^{11}\text{ FLOPs} = 369.4\text{ GFLOPs}$.
- At 989 TFLOPS dense FP16 peak, minimum theoretical execution time is $0.373\text{ ms}$.
- Realized CUTLASS kernel runtime: $0.46\text{ ms}$ (81% Tensor Core efficiency).

## Step 6: Core Systems Takeaway
> Tensor Cores require regular, aligned data dimensions. Padding tensor shapes to multiples of 64 or 128 elements prevents hardware pipeline stalls and guarantees maximum arithmetic throughput.
