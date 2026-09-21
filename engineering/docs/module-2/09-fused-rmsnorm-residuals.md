# Chapter E09: Normalization & Residual Streams: Fused RMSNorm

## Step 1: Hardware Intuition
Imagine taking a shirt out of a drawer, inspecting it, putting it back in the drawer, opening the drawer again to add a badge, and closing it again. Unfused residual normalization does exactly this: it writes the residual sum to HBM, reads it back for normalization, and writes the normalized output back to HBM. Fusing means adding the badge in your hands without ever putting the shirt away.

## Step 2: Silicon Micro-Mechanics
- **Unfused Execution Flaw:**
  1. Add residual: read $\mathbf{x}, \mathbf{res}$, write $\mathbf{x}_{\text{sum}}$ to HBM ($3 \times$ traffic).
  2. RMSNorm: read $\mathbf{x}_{\text{sum}}$, compute variance, scale, write $\mathbf{y}$ to HBM ($2 \times$ traffic).
- **Fused Kernel Execution:**
  A single Triton/CUDA kernel performs residual addition, computes the RMS scaling factor inside registers and SRAM, scales the output, and streams the result directly into registers for the next linear projection.

## Step 3: Cross-Component Coupling
- **Memory Bandwidth Savings:** Eliminates two complete HBM round-trips per Transformer sub-layer.
- **Latency Impact:** Drops sub-layer normalization latency by $3\times$, eliminating memory-bound pipeline bubbles.

## Step 4: The Exact Performance Formula
$$\Delta \text{Memory Traffic} = 4 \times (B \cdot S \cdot d_{\text{model}} \cdot \text{sizeof(dtype)})$$
$$\text{Speedup} = \frac{\text{Unfused Memory Traffic}}{\text{Fused Memory Traffic}} \approx \frac{5}{2} = 2.5\times$$

## Step 5: Concrete Benchmark Walkthrough
Profiling a 70B layer ($d = 8192, S = 4096$) in FP16:
- Unfused HBM traffic: $5 \times (4096 \times 8192 \times 2) = 335.5\text{ MB}$.
- Fused HBM traffic: $2 \times (4096 \times 8192 \times 2) = 134.2\text{ MB}$.
- Execution time on H100 ($3.35\text{ TB/s}$): drops from $100.1\;\mu\text{s}$ to $40.0\;\mu\text{s}$.

## Step 6: Core Systems Takeaway
> Never write residual additions back to HBM. Fusing residual accumulation and normalization inside on-chip registers is an essential low-hanging fruit in high-performance transformer runtimes.
