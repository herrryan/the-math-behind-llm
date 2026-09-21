# Chapter E25: The Complete Model Memory Budget Equation

## Step 1: Hardware Intuition
Imagine packing an expedition backpack with strict weight limits. You have:
1. Essential base gear that never changes (Model Weights).
2. Survival rations for the climb (Optimizer States & Gradients).
3. Temporary trail notes you write and erase along the path (Activations).
If your backpack exceeds 80 kilograms, you cannot walk. Knowing the exact weight of every carabiner is how you reach the summit without collapsing.

## Step 2: Silicon Micro-Mechanics
- **Training Memory Equation:**
  $$M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{temp}}$$
  - Weights: $2P$ bytes (BF16).
  - Gradients: $2P$ bytes (BF16).
  - Adam Optimizer: $12P$ bytes ($4P$ master FP32 weights, $4P$ momentum, $4P$ variance).
  - Base State Total: $16P$ bytes!
- **Inference Memory Equation:**
  $$M_{\text{inference}} = M_{\text{weights}} + M_{\text{KV\_Cache}} + M_{\text{scratchpad}}$$

## Step 3: Cross-Component Coupling
- **Activation Checkpointing (Recomputation):** Storing activations for every layer exhausts HBM. Selective recomputation discards non-GEMM activations, reducing activation memory by $5\times$ for only a $30\%$ compute overhead.

## Step 4: The Exact Performance Formula
$$\text{Activation Memory per Transformer Layer (Standard)} = B \cdot S \cdot d_{\text{model}} \cdot \left(34 + 5 \frac{a \cdot S}{d_{\text{model}}}\right) \text{ bytes}$$

## Step 5: Concrete Benchmark Walkthrough
Auditing memory for LLaMA-3 70B training on 8x H100 (80GB each = 640GB total):
- Model weights: $140\text{ GB}$.
- Gradients: $140\text{ GB}$.
- Adam states: $840\text{ GB}$.
- Total base state: $1120\text{ GB}$.
- Conclusion: Does not fit on 8 GPUs without ZeRO-3 / FSDP sharding!

## Step 6: Core Systems Takeaway
> Training memory is dominated by optimizer states (12P), while inference memory is dominated by the KV cache. Master the memory budget equation to prevent Out-Of-Memory crashes before allocating a single tensor.
