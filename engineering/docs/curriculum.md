# The Engineering Behind Large Language Models: Master Systems Curriculum

<nav aria-label="Table of Contents">
  <p>
    <strong>Curriculum Navigation:</strong>
    <a href="#thesis">Systems Thesis</a> &bull;
    <a href="#pedagogy">Systems Pedagogy</a> &bull;
    <a href="#interaction-graph">Component Interaction Graph</a> &bull;
    <a href="#evolution-chain">Hands-On Systems Labs</a> &bull;
    <a href="#module-0">Module 0</a> &bull;
    <a href="#module-1">Module 1</a> &bull;
    <a href="#module-2">Module 2</a> &bull;
    <a href="#module-3">Module 3</a> &bull;
    <a href="#module-4">Module 4</a> &bull;
    <a href="#module-5">Module 5</a> &bull;
    <a href="#module-6">Module 6</a> &bull;
    <a href="#module-7">Module 7</a>
  </p>
</nav>

<hr>

<h2 id="thesis">1. Systems Thesis: The Physics of LLM Execution</h2>

In theory, a Large Language Model is an elegant cascade of linear transformations and non-linear activations operating on discrete probability spaces. In silicon, an LLM is a **memory-bound data-routing engine constrained by physical hardware bottlenecks**: SRAM capacity, High Bandwidth Memory (HBM) bus width, Tensor Core throughput, and inter-accelerator interconnects (NVLink, PCIe, InfiniBand).

Every mathematical operation carries a hardware toll:

- **Memory Bandwidth vs. Compute:** A model executing at peak arithmetic capability can still crawl at 5% hardware utilization if data cannot be fetched from HBM into on-chip SRAM fast enough.
- **Component Interdependence:** Changing a single architectural knob (such as replacing Multi-Head Attention with Grouped-Query Attention, or switching from standard MLP to SwiGLU) ripples through the entire system: it alters intermediate activation memory, dictates the SRAM tiling block size, changes the Tensor Parallelism communication volume, and shifts the decode phase from memory-bound to compute-bound.
- **System Metrics:** The ultimate success of an LLM engineering implementation is measured in concrete physical numbers: **Time to First Token (TTFT)**, **Inter-Token Latency (ITL)**, **Tokens per Second per GPU**, **Model FLOPs Utilization (MFU)**, and **Total Cost of Ownership (TCO)**.

---

<h2 id="pedagogy">2. The 6-Step Systems Engineering Pedagogy</h2>

Every chapter in this course follows an unshakeable 6-step systems learning ladder:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center" width="8%">Step</th>
      <th align="left" width="22%">Section Title</th>
      <th align="left" width="35%">What You Learn</th>
      <th align="left" width="35%">Systems Concrete Metaphor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Step 1</strong></td>
      <td><strong>Hardware Intuition</strong></td>
      <td>Physical machine constraints with zero software abstractions</td>
      <td>Bucket brigades, warehouse conveyor belts, desk notepads vs. distant filing cabinets</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 2</strong></td>
      <td><strong>Silicon Micro-Mechanics</strong></td>
      <td>How the component executes on actual hardware (registers, SRAM, HBM, ALUs)</td>
      <td>Memory layouts (strides, coalescing), Tensor Core GEMM tiling, warp scheduling</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 3</strong></td>
      <td><strong>Cross-Component Coupling</strong></td>
      <td>How this component impacts and is constrained by other subsystems</td>
      <td>How attention head dimension constrains FlashAttention tiles; how KV cache size chokes batch size</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 4</strong></td>
      <td><strong>The Exact Performance Formula</strong></td>
      <td>Rigorous equations for FLOPs, byte traffic, arithmetic intensity, and latency</td>
      <td>Explicit mathematical formulas for FLOPs, HBM round-trips, activation memory bytes</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 5</strong></td>
      <td><strong>Concrete Benchmark Walkthrough</strong></td>
      <td>Step-by-step arithmetic on real modern configurations (e.g., LLaMA-3 8B/70B, H100 GPU)</td>
      <td>Exact hand-calculated memory footprints, bandwidth saturation, and theoretical vs. actual latencies</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 6</strong></td>
      <td><strong>Core Systems Takeaway</strong></td>
      <td>Punchy 1-2 sentence engineering rule of thumb</td>
      <td>The production heuristic every LLM systems engineer must retain</td>
    </tr>
  </tbody>
</table>

---

<h2 id="interaction-graph">3. Component Interaction &amp; Bottleneck Graph</h2>

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHYSICAL ACCELERATOR HIERARCHY                                 │
│  [Registers: ~64 KB/SM] ◄──► [Shared Memory / SRAM: ~228 KB/SM] ◄──► [High Bandwidth Memory (HBM)│
│  Latency: ~1 cycle          Latency: ~20-30 cycles                  │  Latency: ~200-400 cycles  │
│  Bandwidth: ~30 TB/s         Bandwidth: ~15-20 TB/s                  │  Bandwidth: ~2-3.35 TB/s   │
└───────────────────────────────────────────────┬──────────────────────┴───────────────────────────┘
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                                                                 ▼
┌────────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────┐
│                    PREFILL PHASE                       │  │                DECODE PHASE                │
│  - Input prompt: S tokens processed concurrently      │  │  - Single token generated per iteration    │
│  - Regime: COMPUTE-BOUND (High Arithmetic Intensity)   │  │  - Regime: MEMORY-BANDWIDTH BOUND          │
│  - Bottleneck: Tensor Core GEMM FLOPS                  │  │  - Bottleneck: HBM-to-SRAM byte transfer    │
│  - Target Metric: TTFT (Time to First Token)           │  │  - Target Metric: ITL (Inter-Token Latency)│
└──────────────────────────┬─────────────────────────────┘  └─────────────────────┬──────────────────────┘
                           │                                                      │
                           ▼                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       CROSS-COMPONENT COUPLING                                         │
├───────────────────────────────────┬───────────────────────────────────┬────────────────────────────────┤
│      Component A: ATTENTION       │      Component B: FFN / SwiGLU    │      Component C: MEMORY & KV  │
├───────────────────────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ - FlashAttention: Fuses QK^T and  │ - 3-Matrix projection: W_gate,    │ - KV Cache scales as:          │
│   Softmax in SRAM. Cuts HBM traffic│   W_up, W_down. Consumes 66% of   │   2 * 2 * n_layers * n_kv_heads│
│   from O(S^2) to O(S).             │   model parameters and FLOPs.     │   * d_head * seq_len * batch.  │
│ - MHA vs GQA: GQA reduces KV      │ - Intermediate dim: 8/3 * d_model │ - PagedAttention eliminates     │
│   heads by factor of 8.           │ - Activations dominate memory     │   virtual fragmentation waste. │
│   Coupling: Frees 8x HBM bandwidth│   footprint during backward pass. │ - Decouples batch size limits  │
│   during decode -> boosts batch   │ - TP splits W_gate/up columnwise  │   from max sequence length.    │
│   size by 4-8x.                   │   and W_down rowwise (1 AllReduce)│                                │
└───────────────────────────────────┴───────────────────────────────────┴────────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DISTRIBUTED & COMPILER ENGINE                                        │
│  [Tensor Parallelism (TP)] ◄──► [Pipeline Parallelism (PP)] ◄──► [ZeRO-3 / FSDP Data Parallelism]      │
│  [Fused Kernel Execution]  ◄──► [Continuous Batching]       ◄──► [Speculative Verification]            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

<h2 id="evolution-chain">4. The Hands-On Systems Labs</h2>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="18%">Lab Stage</th>
      <th align="left" width="22%">Lab Link</th>
      <th align="left" width="30%">Systems Milestone</th>
      <th align="left" width="30%">Hardware Bottleneck Conquered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Lab E1: The Roofline Profiler</strong></td>
      <td><a href="labs/01-roofline-profiler.md">Roofline Profiler</a></td>
      <td>Custom Python/Torch profiler that computes operational intensity and plots actual kernel execution against GPU hardware rooflines.</td>
      <td><strong>Blind Optimization</strong>: Eliminates guesswork by diagnosing whether a slow kernel is memory-bandwidth bound or compute-bound.</td>
    </tr>
    <tr>
      <td><strong>Lab E2: Minimal FlashAttention</strong></td>
      <td><a href="labs/02-flash-attention-kernel.md">FlashAttention Kernel</a></td>
      <td>Triton / C++ implementation of tiled attention with online softmax, keeping intermediate matrices strictly inside on-chip SRAM.</td>
      <td><strong>The O(S^2) Memory Wall</strong>: Eliminates materialization of the attention matrix in HBM, slashing memory traffic by 10x.</td>
    </tr>
    <tr>
      <td><strong>Lab E3: Paged KV Cache Engine</strong></td>
      <td><a href="labs/03-paged-kv-cache.md">Paged KV Cache</a></td>
      <td>A zero-dependency Python/NumPy implementation of virtual memory page-table management for dynamic KV cache allocation.</td>
      <td><strong>Memory Fragmentation Waste</strong>: Eliminates static contiguous buffer pre-allocation, boosting effective serving concurrency by 3x.</td>
    </tr>
    <tr>
      <td><strong>Lab E4: 2-GPU Tensor Parallelism</strong></td>
      <td><a href="labs/04-tensor-parallel-engine.md">Tensor Parallel Engine</a></td>
      <td>Distributed Megatron-style column-parallel and row-parallel linear layers with manual PyTorch collective communications.</td>
      <td><strong>Single-Device Memory Limits</strong>: Distributes weight tensors and activation computations across distinct physical accelerators.</td>
    </tr>
  </tbody>
</table>

---

<h2 id="module-0">Module 0: Hardware Architecture &amp; The Physical Landscape</h2>

- **[Chapter E00: The Silicon Anatomy](module-0/00-silicon-anatomy.md)**: HBM, SRAM, Tensor Cores, and Interconnects.
- **[Chapter E01: The Roofline Model &amp; Arithmetic Intensity](module-0/01-roofline-model.md)**: Compute-bound vs. memory-bound regimes.
- **[Chapter E02: Memory Layouts, Strides, and Memory Coalescing](module-0/02-memory-layouts-and-coalescing.md)**: DRAM burst mechanics and tensor strides.

---

<h2 id="module-1">Module 1: The Attention Subsystem in Silicon</h2>

- **[Chapter E03: Naive Attention's Fatal Flaw](module-1/03-naive-attention-memory-wall.md)**: The $O(S^2)$ HBM memory traffic bottleneck.
- **[Chapter E04: FlashAttention (1, 2, &amp; 3)](module-1/04-flash-attention.md)**: SRAM tiling, online softmax, and TMA async copy.
- **[Chapter E05: The KV Cache Anatomy &amp; PagedAttention](module-1/05-kv-cache-and-paged-attention.md)**: Virtual memory page tables for token caches.
- **[Chapter E06: Attention Architectural Variants (MHA, GQA, MLA)](module-1/06-mha-mqa-gqa-mla.md)**: Multi-query, grouped-query, and multi-head latent attention.

---

<h2 id="module-2">Module 2: Linear Projections, Activations &amp; Memory Bandwidth</h2>

- **[Chapter E07: GEMM at Scale &amp; Tensor Cores](module-2/07-gemm-tensor-cores.md)**: Systolic arrays, warp tiling, and memory alignment.
- **[Chapter E08: FFNs &amp; SwiGLU in Silicon](module-2/08-swiglu-ffn-in-silicon.md)**: 3-matrix projections and activation footprint.
- **[Chapter E09: Fused RMSNorm &amp; Residual Streams](module-2/09-fused-rmsnorm-residuals.md)**: Operator fusion and eliminating HBM round-trips.
- **[Chapter E10: RoPE Kernel Fusion on the Fly](module-2/10-rope-kernel-fusion.md)**: On-the-fly trigonometric rotations in registers.

---

<h2 id="module-3">Module 3: Numerical Formats, Precision &amp; Quantization</h2>

- **[Chapter E11: Number Formats in Silicon (FP32, FP16, BF16, FP8)](module-3/11-number-formats-fp8-bf16.md)**: Exponent vs. mantissa trade-offs.
- **[Chapter E12: Quantization Strategies (W8A8, W4A16, INT4)](module-3/12-quantization-strategies-w8a8-w4a16.md)**: Weight-only vs. weight-activation quantization.
- **[Chapter E13: KV Cache Quantization](module-3/13-kv-cache-quantization.md)**: Compressing context to FP8 and INT4.

---

<h2 id="module-4">Module 4: Distributed Systems &amp; Scaling Parallelism</h2>

- **[Chapter E14: Distributed Communication Primitives](module-4/14-distributed-communication-primitives.md)**: All-Reduce, All-Gather, and Reduce-Scatter.
- **[Chapter E15: Tensor Parallelism (Megatron-LM)](module-4/15-tensor-parallelism-megatron.md)**: Row-column partitioning and dual All-Reduces.
- **[Chapter E16: Pipeline Parallelism (1F1B)](module-4/16-pipeline-parallelism-1f1b.md)**: Bubble analysis and activation stashing.
- **[Chapter E17: Data Parallelism &amp; Zero Redundancy (FSDP / ZeRO)](module-4/17-zero-fsdp-data-parallelism.md)**: Memory sharding vs. communication overhead.
- **[Chapter E18: Context &amp; Sequence Parallelism (Ring Attention)](module-4/18-ring-attention-context-parallelism.md)**: Scaling to 1M+ context via ring communication.

---

<h2 id="module-5">Module 5: High-Throughput Inference Engines &amp; Serving</h2>

- **[Chapter E19: The Two Inference Regimes (Prefill vs. Decode)](module-5/19-prefill-vs-decode-regimes.md)**: TTFT compute-bound vs. ITL memory-bound dynamics.
- **[Chapter E20: Continuous Batching &amp; Dynamic Scheduling](module-5/20-continuous-batching-scheduling.md)**: Iteration-level scheduling in vLLM.
- **[Chapter E21: Speculative Decoding](module-5/21-speculative-decoding.md)**: Draft-verify silicon acceleration.
- **[Chapter E22: Chunked Prefill &amp; Prefix Caching](module-5/22-chunked-prefill-prefix-caching.md)**: Radix trees and bounding decode interference.

---

<h2 id="module-6">Module 6: Mixture of Experts (MoE) Engineering</h2>

- **[Chapter E23: MoE Architecture in Systems (Top-K Gating)](module-6/23-moe-routing-capacity-factor.md)**: Active vs. total parameters and capacity factors.
- **[Chapter E24: All-to-All Dispatch &amp; Combine Bottlenecks](module-6/24-all-to-all-distributed-moe.md)**: Network crossbar traffic in distributed MoE.

---

<h2 id="module-7">Module 7: Auditing, Profiling &amp; Hardware Efficiency Metrics</h2>

- **[Chapter E25: The Complete Model Memory Budget Equation](module-7/25-model-memory-budget.md)**: $16P$ training memory accounting vs. activation checkpoints.
- **[Chapter E26: Measuring True Performance (MFU, HFU, Profiling)](module-7/26-measuring-mfu-hfu-profiling.md)**: Model FLOPs utilization, `nsys`, and `ncu`.
