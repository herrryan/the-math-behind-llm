# Chapter E15: Tensor Parallelism (TP): Megatron-LM in Silicon

## Step 1: Hardware Intuition
Imagine a mural painting so wide that no single painter can reach both ends. Instead of having painters paint in shifts, you divide the canvas vertically. Painter 1 paints the left half, Painter 2 paints the right half. When finished, they step back and coordinate only at the boundary lines.

## Step 2: Silicon Micro-Mechanics
- **Megatron-LM Sharding Strategy (Shoeybi et al.):**
  - Self-Attention: Split $W_Q, W_K, W_V$ column-wise (no communication needed); split $W_O$ row-wise (requires 1 All-Reduce).
  - FFN: Split $W_{\text{gate}}, W_{\text{up}}$ column-wise; split $W_{\text{down}}$ row-wise (requires 1 All-Reduce).
  - Total Communication: Exactly 2 All-Reduces per Transformer layer.
- **Sequence Parallelism (SP):** Shards LayerNorm and Dropout along the sequence dimension to eliminate redundant compute and activation memory.

## Step 3: Cross-Component Coupling
- **Hardware Boundary:** TP degree is strictly bounded by single-node NVLink limits (typically $\text{TP} \le 8$). Running TP across nodes causes massive interconnect latency stalls.

## Step 4: The Exact Performance Formula
$$\text{Communication Volume per Layer} = 4 \times B \times S \times d_{\text{model}} \times \text{sizeof(dtype)}$$
$$\text{Layer Speedup} = \frac{T_{\text{single}}}{T_{\text{TP}}} = \frac{\text{FLOPs} / \text{TP} + T_{\text{AllReduce}}}{\text{FLOPs}}$$

## Step 5: Concrete Benchmark Walkthrough
Tracing Megatron TP=8 for LLaMA-3 70B ($d = 8192, S = 4096$) in FP16:
- Parameter footprint per GPU: $140\text{ GB} / 8 = 17.5\text{ GB}$.
- All-Reduce volume per layer: $2 \times (4096 \times 8192 \times 2) = 134.2\text{ MB}$.
- Total communication per forward pass across 80 layers: $10.7\text{ GB}$.

## Step 6: Core Systems Takeaway
> Tensor Parallelism distributes weight matrices with exactly two All-Reduce communications per layer, but requires ultra-fast NVLink interconnects to prevent communication stalls.
