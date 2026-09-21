# Chapter E08: Feed-Forward Networks & SwiGLU in Silicon

## Step 1: Hardware Intuition
In a standard restaurant kitchen, a dish goes through two preparation stages: chopping and seasoning. In a modern kitchen (SwiGLU), the dish passes through three parallel prep stations: one chops the vegetables, one whips a savory glaze, and an air-valve blender instantly mixes them together before baking. This gives richer flavor, but requires more counter space to hold ingredients simultaneously.

## Step 2: Silicon Micro-Mechanics
- **3-Matrix Structure:** SwiGLU uses $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ffn}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{ffn} \times d}$.
- Intermediate dimension: $d_{ffn} \approx \frac{8}{3} d_{\text{model}}$.
- FFN accounts for approximately $66\%$ of total model parameters and compute FLOPs.
- **Activation Memory Footprint:** Requires holding intermediate outputs of both $W_{\text{gate}}$ and $W_{\text{up}}$ in memory during forward execution for backpropagation.

## Step 3: Cross-Component Coupling
- **Activation Checkpointing:** SwiGLU increases activation storage by 50% over standard ReLU/GELU MLPs, necessitating selective recomputation during training.
- **GEMM Fusion:** Concatenating $W_{\text{gate}}$ and $W_{\text{up}}$ into a single combined matrix $[W_{\text{gate}} \mid W_{\text{up}}] \in \mathbb{R}^{d \times 2d_{ffn}}$ doubles the column dimension, improving Tensor Core GEMM efficiency.

## Step 4: The Exact Performance Formula
$$\text{FLOPs}_{\text{SwiGLU}} = 3 \times (2 \times d_{\text{model}} \times d_{ffn}) = 16 d_{\text{model}}^2$$
$$\text{Activation Memory per Token} = 2 \times d_{ffn} \times \text{sizeof(dtype)}$$

## Step 5: Concrete Benchmark Walkthrough
Calculating the parameters and activation size for LLaMA-3 70B ($d_{\text{model}} = 8192, d_{ffn} = 28672$):
- Parameter count per layer: $3 \times (8192 \times 28672) = 704.6\text{ million parameters}$.
- Total FFN parameters across 80 layers: $56.3\text{ billion}$ (80% of model weights!).
- Forward activation volume per token: $2 \times 28672 \times 2\text{ bytes} = 114.7\text{ KB}$.

## Step 6: Core Systems Takeaway
> FFNs dominate the parameter count and compute budget of modern LLMs. Combining the gate and up projections into a single fused GEMM maximizes hardware throughput and reduces kernel launch overhead.
