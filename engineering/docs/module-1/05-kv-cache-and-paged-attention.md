# Chapter E05: The KV Cache Anatomy: Static Waste vs. PagedAttention

## Step 1: Hardware Intuition
Imagine a student taking an open-book exam where they write one sentence per minute. Instead of writing sentences sequentially in an ordinary notebook, the school assigns a massive 500-page leather-bound binder to every student at the start of the exam, just in case they might write 500 pages. Most desks fill up with empty binders, and the classroom runs out of space. PagedAttention acts like a loose-leaf ring binder: pages are allocated on demand, exactly when a page fills up.

## Step 2: Silicon Micro-Mechanics
- **KV Cache Formula:** Each past token requires storing Key and Value projection vectors:
  40062\text{Bytes per Token} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \text{ bytes}40062
- **Static Memory Allocation Waste:**
  - Internal fragmentation: Pre-allocating memory for  (e.g. 4096) when requests only use 200 tokens.
  - External fragmentation: Varying request lengths leaving unusable memory holes.
  - Virtual memory pagination: Kwon et al. (vLLM) introduce **PagedAttention**, dividing the KV cache into fixed-size physical blocks (e.g. 16 tokens per block) tracked by page tables.

## Step 3: Cross-Component Coupling
- **Serving Concurrency:** Eliminating fragmentation reduces memory waste from 60-80% to under 4%, immediately allowing \times$ to \times$ larger concurrent batch sizes.
- **Prefix Sharing:** Enables zero-copy copy-on-write memory sharing for parallel sampling and common prompt templates.

## Step 4: The Exact Performance Formula
40062\text{Cache Size (Bytes)} = 4 \times B \cdot S \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}}40062
40062\text{Memory Utilization Rate } \eta_{\text{mem}} = \frac{\text{Active Tokens Used}}{\text{Total Physical Blocks Allocated} \times \text{Block Size}}40062

## Step 5: Concrete Benchmark Walkthrough
Calculating the KV cache memory footprint for LLaMA-3 70B (80 layers, 8 KV heads,  = 128$) at FP16:
- Bytes per token:  \times 80 \times 8 \times 128 = 327,680	ext{ bytes} = 0.3125	ext{ MB/token}$.
- Batch of 64 requests at 4096 context:  \times 4096 \times 0.3125	ext{ MB} = 81.92	ext{ GB}$.
- Without PagedAttention, an 80GB GPU runs out of memory before serving 64 concurrent requests.

## Step 6: Core Systems Takeaway
> In production LLM inference, serving capacity is bounded by KV cache memory, not compute. Virtual memory paging via PagedAttention eliminates fragmentation and maximizes GPU serving density.
