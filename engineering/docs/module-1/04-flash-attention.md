# Chapter E04: FlashAttention (1, 2, & 3): SRAM Tiling & Online Softmax

## Step 1: Hardware Intuition
Instead of writing down the full matrix of comparisons on sheets of paper and shipping them to the basement, imagine breaking the text into small puzzle blocks. You bring one small block of queries and one small block of keys to your kitchen table (on-chip SRAM), compute the attention scores locally, update a running tally, and discard the raw scores immediately. The full matrix is never written down in external storage.

## Step 2: Silicon Micro-Mechanics
- **SRAM Tiling:** Partitions inputs , K, V$ into blocks of size  	imes d$ and  	imes d$ that fit into on-chip shared memory (	ext{ KB}$ per SM on H100).
- **Online Softmax:** Tracks running maximum $ and normalizer $:
  40062m_{\text{new}} = \max(m_{\text{old}}, x), \quad d_{\text{new}} = d_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} + e^{x - m_{\text{new}}}40062
- **FlashAttention-2 & 3 Evolutions:**
  - FlashAttention-2: Moves warp partitioning over sequence length and minimizes non-GEMM FLOPs.
  - FlashAttention-3: Leverages Hopper Tensor Memory Accelerator (TMA), FP8 Tensor Cores, and warp-specialized ping-pong GEMM scheduling.

## Step 3: Cross-Component Coupling
- **Head Dimension Constraints:** Tile dimensions are bounded by SRAM size. Head dimension  = 128$ fits efficiently;  = 256$ cuts tile sizes in half and increases register pressure.
- **Backward Pass Gradient Recomputation:** FlashAttention does not store the  	imes S$ matrix for backpropagation; it recomputes it block-by-block on the fly, saving gigabytes of activation memory.

## Step 4: The Exact Performance Formula
40062\text{Memory Traffic}_{\text{flash}} = \mathcal{O}\left(B \cdot H \cdot S \cdot d + \frac{B \cdot H \cdot S^2 \cdot d^2}{\text{SRAM Size}}\right)40062
Traffic reduction factor compared to naive attention:
40062\text{Speedup Factor} \approx \frac{S \cdot d}{\text{SRAM Size}}40062

## Step 5: Concrete Benchmark Walkthrough
Step-by-step arithmetic comparing HBM traffic for  = 8192$,  = 128$ on an H100:
- Naive attention HBM traffic: .6	ext{ GB}$ per layer.
- FlashAttention HBM traffic: zsh.67	ext{ GB}$ per layer (.8\times$ reduction).
- Execution latency drops from .2	ext{ ms}$ to .1	ext{ ms}$.

## Step 6: Core Systems Takeaway
> By fusing matrix multiplication, softmax normalization, and value accumulation into on-chip SRAM via online rescaling, FlashAttention breaks the (S^2)$ memory bandwidth bottleneck without altering mathematical outputs.
