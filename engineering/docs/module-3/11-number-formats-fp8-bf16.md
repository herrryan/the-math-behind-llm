# Chapter E11: Number Formats in Silicon: FP32, FP16, BF16, and FP8

## Step 1: Hardware Intuition
Think of numbers as rulers. A long wooden ruler (FP32) measures millimeter precision across vast distances. A pocket tape measure (FP16) gives millimeter precision, but breaks if you try to measure something longer than a table (underflow/overflow). BF16 shrinks the markings slightly, but extends the tape to measure football fields. FP8 is a miniature folding ruler: compact, twice as fast to carry, but requiring careful scaling blocks to avoid snapping.

## Step 2: Silicon Micro-Mechanics
- **Bit Allocations:**
  - FP32: 1 sign + 8 exponent + 23 mantissa (Range: $10^{\pm 38}$, precision: $10^{-7}$).
  - FP16: 1 sign + 5 exponent + 10 mantissa (Range: $6.5 \times 10^4$). Prone to loss overflow.
  - BF16: 1 sign + 8 exponent + 7 mantissa. Matches FP32 range, half the memory.
  - FP8 E4M3: 1 sign + 4 exponent + 3 mantissa. Optimized for forward weights and activations.
  - FP8 E5M2: 1 sign + 5 exponent + 2 mantissa. Wider dynamic range, optimized for backward gradients.

## Step 3: Cross-Component Coupling
- **Tensor Core Throughput:** Modern Tensor Cores (NVIDIA Hopper/Blackwell) execute FP8 matrix multiplies at $2\times$ the TFLOPS rate of FP16/BF16.
- **Dynamic Scaling Factors:** FP8 requires tracking per-tensor or per-channel maximum values to scale activations into the valid numerical range.

## Step 4: The Exact Performance Formula
$$\text{Weight Memory (GB)} = \frac{\text{Parameter Count (B)} \times \text{Bits per Parameter}}{8}$$
$$\text{Throughput Ratio} = \frac{\text{FP8 TFLOPS}}{\text{BF16 TFLOPS}} = 2.0\times$$

## Step 5: Concrete Benchmark Walkthrough
Training a 70B model:
- FP32 weights: $280\text{ GB}$.
- BF16 weights: $140\text{ GB}$.
- FP8 weights: $70\text{ GB}$.
- Memory bandwidth savings: Cutting weight traffic from $140\text{ GB}$ to $70\text{ GB}$ immediately doubles decoding throughput in bandwidth-bound regimes.

## Step 6: Core Systems Takeaway
> Precision determines dynamic range, memory bandwidth, and Tensor Core throughput. Modern LLM training standardizes on BF16 for numerical stability and FP8 for high-throughput GEMM execution.
