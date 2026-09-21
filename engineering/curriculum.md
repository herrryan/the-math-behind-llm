# The Engineering Behind Large Language Models: Master Systems Curriculum

<nav aria-label="Table of Contents">
  <p>
    <strong>Engineering Curriculum Navigation:</strong>
    <a href="#thesis">Systems Thesis</a> &bull;
    <a href="#pedagogy">Systems Pedagogy</a> &bull;
    <a href="#interaction-graph">Component Interaction Graph</a> &bull;
    <a href="#evolution-chain">Hands-On Systems Labs</a> &bull;
    <a href="#module-0">Module 0: Hardware Landscape</a> &bull;
    <a href="#module-1">Module 1: Attention in Silicon</a> &bull;
    <a href="#module-2">Module 2: GEMM & Activations</a> &bull;
    <a href="#module-3">Module 3: Quantization & Precision</a> &bull;
    <a href="#module-4">Module 4: Distributed Systems</a> &bull;
    <a href="#module-5">Module 5: Serving & Inference</a> &bull;
    <a href="#module-6">Module 6: MoE Systems</a> &bull;
    <a href="#module-7">Module 7: Auditing & MFU</a>
  </p>
</nav>

<hr>

<h2 id="thesis">1. Systems Thesis: The Physics of LLM Execution</h2>

<p>In theory, a Large Language Model is an elegant cascade of linear transformations and non-linear activations operating on discrete probability spaces. In silicon, an LLM is a <strong>memory-bound data-routing engine constrained by physical hardware bottlenecks</strong>: SRAM capacity, High Bandwidth Memory (HBM) bus width, Tensor Core throughput, and inter-accelerator interconnects (NVLink, PCIe, InfiniBand).</p>

<p>Every mathematical operation carries a hardware toll:</p>
<ul>
  <li><strong>Memory Bandwidth vs. Compute:</strong> A model executing at peak arithmetic capability can still crawl at 5% hardware utilization if data cannot be fetched from HBM into on-chip SRAM fast enough.</li>
  <li><strong>Component Interdependence:</strong> Changing a single architectural knob (such as replacing Multi-Head Attention with Grouped-Query Attention, or switching from standard MLP to SwiGLU) ripples through the entire system: it alters intermediate activation memory, dictates the SRAM tiling block size, changes the Tensor Parallelism communication volume, and shifts the decode phase from memory-bound to compute-bound.</li>
  <li><strong>System Metrics:</strong> The ultimate success of an LLM engineering implementation is measured in concrete physical numbers: <strong>Time to First Token (TTFT)</strong>, <strong>Inter-Token Latency (ITL)</strong>, <strong>Tokens per Second per GPU</strong>, <strong>Model FLOPs Utilization (MFU)</strong>, and <strong>Total Cost of Ownership (TCO)</strong>.</li>
</ul>

<hr>

<h2 id="pedagogy">2. The 6-Step Systems Engineering Pedagogy</h2>

<p>To mirror the rigor of the mathematical curriculum while addressing systems engineering reality, every chapter in this subproject follows an unshakeable 6-step systems learning ladder:</p>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table E.1:</strong> The 6-step systems engineering pedagogical ladder applied to every chapter.</caption>
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
      <td>The production heuristic every LLM systems engineer must tattoo on their brain</td>
    </tr>
  </tbody>
</table>

<hr>

<h2 id="interaction-graph">3. Component Interaction &amp; Bottleneck Graph</h2>

<p>In modern LLM engines, no component operates in isolation. The following dataflow diagram illustrates how hardware memory tiers, linear projections, attention mechanisms, and serving schedulers couple to determine overall model performance:</p>

<figure>
<pre>
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
│                                   DISTRIBUTED &amp; COMPILER ENGINE                                       │
│  [Tensor Parallelism (TP)] ◄──► [Pipeline Parallelism (PP)] ◄──► [ZeRO-3 / FSDP Data Parallelism]      │
│  [Fused Kernel Execution]  ◄──► [Continuous Batching]       ◄──► [Speculative Verification]            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure E.1:</strong> Architectural coupling and hardware hierarchy across LLM training and inference systems.</figcaption>
</figure>

<hr>

<h2 id="evolution-chain">4. The Hands-On Systems Engineering Evolution Chain</h2>

<p>Just as the theoretical course features the 4-Stage Python Brain Evolution Chain, the engineering subproject follows an unbroken, hands-on systems development track. Students build, profile, and optimize real high-performance components:</p>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table E.2:</strong> The 4-Stage Hands-On Systems Lab Evolution Chain.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="18%">Lab Stage</th>
      <th align="left" width="22%">Lab Location</th>
      <th align="left" width="30%">Systems Milestone</th>
      <th align="left" width="30%">Hardware Bottleneck Conquered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Lab E1: The Roofline Profiler</strong></td>
      <td><code>engineering/labs/01-roofline-profiler/</code></td>
      <td>Custom Python/Torch profiler that computes operational intensity and plots actual kernel execution against GPU hardware rooflines.</td>
      <td><mark>Blind Optimization</mark>: Eliminates guesswork by diagnosing whether a slow kernel is memory-bandwidth bound or compute-bound.</td>
    </tr>
    <tr>
      <td><strong>Lab E2: The Minimal FlashAttention Kernel</strong></td>
      <td><code>engineering/labs/02-flash-attention-kernel/</code></td>
      <td>Triton / C++ implementation of tiled attention with online softmax, keeping intermediate matrices strictly inside on-chip SRAM.</td>
      <td><mark>The $O(N^2)$ Memory Wall</mark>: Eliminates materialization of the $S \times S$ attention matrix in HBM, slashing memory traffic by $10\times$.</td>
    </tr>
    <tr>
      <td><strong>Lab E3: The Paged KV Cache Engine</strong></td>
      <td><code>engineering/labs/03-paged-kv-cache/</code></td>
      <td>A zero-dependency Python/NumPy implementation of virtual memory page-table management for dynamic KV cache allocation.</td>
      <td><mark>Memory Fragmentation Waste</mark>: Eliminates static contiguous buffer pre-allocation, boosting effective serving concurrency by $3\times$.</td>
    </tr>
    <tr>
      <td><strong>Lab E4: 2-GPU Tensor Parallelism Engine</strong></td>
      <td><code>engineering/labs/04-tensor-parallel-engine/</code></td>
      <td>Distributed Megatron-style column-parallel and row-parallel linear layers with manual PyTorch collective communications.</td>
      <td><mark>Single-Device Memory Limits</mark>: Distributes weight tensors and activation computations across distinct physical accelerators.</td>
    </tr>
  </tbody>
</table>

<hr>

<h2 id="module-0">Module 0: Hardware Architecture &amp; The Physical Landscape</h2>
<p><em>Understanding the silicon substrate: how modern AI accelerators store, move, and crunch numbers.</em></p>

<h3>Chapter E00: The Silicon Anatomy: HBM, SRAM, Tensor Cores, and Interconnects</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Physical topology of modern GPUs (NVIDIA H100/B200, AMD MI300X, Google TPU v5). Memory hierarchy: Registers (64 KB/SM, 1-cycle latency), Shared Memory / L1 SRAM (228 KB/SM, 20-cycle latency, ~15 TB/s aggregate), L2 Cache (50 MB, ~5 TB/s), High Bandwidth Memory (HBM3/HBM3e: 80-141 GB, 3.35 TB/s). Interconnect topologies: PCIe Gen 5 (64 GB/s bidirectional), NVLink 4 (900 GB/s bidirectional), NVSwitch, and RoCE/InfiniBand networks (400-800 Gbps).</li>
  <li><strong>Cross-Component Interactions:</strong> Why algorithmic speedups that increase arithmetic complexity but keep data in SRAM consistently beat mathematically simpler operations that require round-trips to HBM.</li>
  <li><strong>Performance Formula:</strong> Memory access latency vs. bandwidth saturation:
  $$\text{Transfer Time} = \text{Latency} + \frac{\text{Bytes Transferred}}{\text{Bandwidth}}$$
  The latency-hiding threshold via warp parallelism.</li>
  <li><strong>Real-World Case Study:</strong> NVIDIA Hopper H100 SM architecture: Asynchronous Transaction Barrier and Tensor Memory Accelerator (TMA).</li>
</ul>

<h3>Chapter E01: The Roofline Model &amp; Arithmetic Intensity</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Deriving the Roofline Model. Operational (Arithmetic) Intensity:
  $$I = \frac{\text{FLOPs}}{\text{Memory Access Bytes}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$
  Plotting the inflection point $I^* = \frac{\text{Peak TFLOPS}}{\text{Peak Memory Bandwidth (TB/s)}}$. For an H100 SXM (2,000 TFLOPS FP16, 3.35 TB/s HBM3), $I^* \approx 597 \text{ FLOPs/Byte}$. Any operation with $I < 597$ is strictly <strong>memory-bound</strong>; any operation with $I > 597$ is <strong>compute-bound</strong>.</li>
  <li><strong>Cross-Component Interactions:</strong> Why batch size acts as a universal dial shifting matrix operations from memory-bound (batch size 1 GEMV during decode) to compute-bound (batch size 64+ GEMM during prefill).</li>
  <li><strong>Performance Formula:</strong> Attainable performance:
  $$P = \min\left(P_{\text{peak}}, \; I \times \text{BW}_{\text{mem}}\right)$$</li>
  <li><strong>Real-World Case Study:</strong> Calculating the exact arithmetic intensity of LayerNorm ($I \approx 2$), Softmax ($I \approx 2.5$), and GEMM ($I \approx \frac{2MNK}{2(MK + KN + MN)}$).</li>
</ul>

<h3>Chapter E02: Memory Layouts, Strides, and Memory Coalescing</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Physical DRAM burst architecture and 32-byte / 128-byte cache line transactions. Memory coalescing across a 32-thread warp. Row-major vs. column-major layouts, tensor strides, and non-contiguous memory slices. The hidden overhead of tensor transposition (`.transpose()` / `.contiguous()`) in PyTorch.</li>
  <li><strong>Cross-Component Interactions:</strong> How key-value cache layouts (e.g., `[batch, num_heads, seq_len, head_dim]` vs. `[batch, seq_len, num_heads, head_dim]`) dictate whether memory reads during multi-head attention can achieve 100% coalescing or suffer $4\times$ memory bandwidth waste.</li>
  <li><strong>Performance Formula:</strong> Memory transaction efficiency:
  $$\eta_{\text{mem}} = \frac{\text{Useful Bytes Requested}}{\text{Total Cache Lines Fetched} \times \text{Cache Line Size}}$$</li>
  <li><strong>Real-World Case Study:</strong> Why transposing the Key tensor for $Q K^\top$ historically ruined memory coalescing before fused SRAM transpose kernels were developed.</li>
</ul>

<hr>

<h2 id="module-1">Module 1: The Attention Subsystem in Silicon</h2>
<p><em>From theoretical quadratic dot-products to SRAM-tiled kernels and compressed caches.</em></p>

<h3>Chapter E03: Naive Attention's Fatal Flaw: The $O(S^2)$ HBM Memory Traffic</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Step-by-step breakdown of standard textbook attention:
  $$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top, \quad \mathbf{P} = \text{softmax}(\mathbf{S}), \quad \mathbf{O} = \mathbf{P}\mathbf{V}$$
  Tracking every memory transaction between GPU HBM and SRAM. To process context length $S = 8192$ with $d = 128$, the intermediate attention matrix $\mathbf{S}$ consumes $8192 \times 8192 \times 2 \text{ bytes} = 134 \text{ MB}$ per head. For 32 heads, that is $4.3 \text{ GB}$ per layer! Writing $\mathbf{S}$ to HBM, reading it back for Softmax, writing $\mathbf{P}$ to HBM, and reading it back for $\mathbf{V}$ multiplication creates a catastrophic memory bottleneck.</li>
  <li><strong>Cross-Component Interactions:</strong> How naive attention memory allocation scales quadratically, triggering Out-Of-Memory (OOM) errors during long-context training even when compute utilization remains below 20%.</li>
  <li><strong>Performance Formula:</strong> Total HBM memory traffic:
  $$\text{Bytes}_{\text{naive}} = \mathcal{O}(B \cdot H \cdot S \cdot d) + \mathcal{O}(B \cdot H \cdot S^2)$$</li>
  <li><strong>Real-World Case Study:</strong> Benchmarking PyTorch's native `torch.matmul(Q, K.transpose())` vs. available HBM bandwidth on an A100 GPU.</li>
</ul>

<h3>Chapter E04: FlashAttention (1, 2, &amp; 3): SRAM Tiling &amp; Online Softmax</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> The breakthrough of Dao et al.: Never materialize the $S \times S$ matrix in HBM. Tiling inputs into SRAM blocks of size $B_r \times d$ and $B_c \times d$. Using Milakov &amp; Gimelshein's <strong>Online Softmax</strong> to compute running normalizers:
  $$m_{\text{new}} = \max(m_{\text{old}}, x), \quad d_{\text{new}} = d_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} + e^{x - m_{\text{new}}}$$
  FlashAttention-2 optimizations: reducing non-matmul FLOPs, warp partitioning over sequence length. FlashAttention-3: utilizing Hopper FP8 Tensor Cores, TMA asynchronous copy, and ping-pong GEMM scheduling.</li>
  <li><strong>Cross-Component Interactions:</strong> How SRAM tile sizes are strictly constrained by head dimension $d_k$: why $d_k = 64$ or $128$ fits comfortably into 228 KB SRAM, while $d_k = 256$ causes register spilling and reduces warp occupancy.</li>
  <li><strong>Performance Formula:</strong> FlashAttention HBM byte reduction:
  $$\text{Bytes}_{\text{flash}} = \mathcal{O}\left(B \cdot H \cdot S \cdot d + \frac{B \cdot H \cdot S^2 \cdot d^2}{\text{SRAM Size}}\right)$$
  Slashing memory round-trips by a factor of $\frac{S \cdot d}{\text{SRAM Size}}$.</li>
  <li><strong>Real-World Case Study:</strong> Kernel disassembly of FlashAttention-2 vs FlashAttention-3 on an H100.</li>
</ul>

<h3>Chapter E05: The KV Cache Anatomy: Static Waste vs. PagedAttention</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Autoregressive token generation generates one token at a time. To prevent recomputing past tokens, Keys and Values are cached. Anatomy of a KV Cache tensor:
  $$\text{Memory per Token} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \text{ bytes}$$
  For LLaMA-3 70B (80 layers, 8 KV heads, $d = 128$) at FP16: $327,680 \text{ bytes (0.32 MB) per token}$. For a batch of 64 requests with 4K context, the KV cache alone demands $85.8 \text{ GB}$ of pure HBM!</li>
  <li><strong>Cross-Component Interactions:</strong> Static memory allocation forces engines to pre-allocate memory for the maximum possible sequence length (e.g. 4096), causing 60-80% memory waste via internal and external fragmentation, capping maximum serving concurrency.</li>
  <li><strong>Performance Formula:</strong> Kwon et al. (vLLM) <strong>PagedAttention</strong>: Virtual memory pagination with page tables, block size $B = 16$, and non-contiguous physical block allocation. Near-zero memory waste ($< 4\%$).</li>
  <li><strong>Real-World Case Study:</strong> Calculating the maximum concurrent batch size on an 80GB A100 before and after PagedAttention.</li>
</ul>

<h3>Chapter E06: Attention Architectural Variants: MHA vs. MQA vs. GQA vs. MLA</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Why Multi-Head Attention (MHA) creates an unsustainable memory bandwidth burden during decoding.
  Multi-Query Attention (MQA, Shazeer 2019): Single KV head shared across all Q heads.
  Grouped-Query Attention (GQA, Ainslie et al. 2023): $G$ query heads share 1 KV head (e.g. 8:1 ratio in LLaMA-3).
  Multi-Head Latent Attention (MLA, DeepSeek-V2/V3): Low-rank joint compression of Keys and Values into a single compressed latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$.</li>
  <li><strong>Cross-Component Interactions:</strong> How switching from MHA to GQA cuts KV Cache memory traffic by $8\times$, directly accelerating the decode phase by up to $4\times$ and allowing $4-8\times$ larger serving batch sizes without quality degradation.</li>
  <li><strong>Performance Formula:</strong> KV cache memory bandwidth demand per decode token:
  $$\text{BW}_{\text{decode}} = \frac{2 \times \text{Precision Bytes} \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S}{\text{ITL Target (seconds)}}$$</li>
  <li><strong>Real-World Case Study:</strong> LLaMA-2 (MHA) vs. LLaMA-3 (GQA) vs. DeepSeek-V3 (MLA) KV cache footprint comparison table.</li>
</ul>

<hr>

<h2 id="module-2">Module 2: Linear Projections, Activations &amp; Memory Bandwidth</h2>
<p><em>Optimizing the heavy workhorses: GEMM engines, SwiGLU 3-matrix projections, and kernel fusion.</em></p>

<h3>Chapter E07: GEMM at Scale: Tensor Cores, Systolic Arrays, and Tiling</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Anatomy of an accelerated General Matrix Multiply (GEMM: $C = \alpha A B + \beta C$). Tensor Core execution: MMA (Matrix Multiply-Accumulate) instructions executing $16 \times 16 \times 16$ micro-matrix tiles per clock cycle. Multi-level hierarchical tiling: Grid-level (Threadblocks), Warp-level, and Thread-level. Shared memory bank conflicts and double-buffering / asynchronous copy pipelines.</li>
  <li><strong>Cross-Component Interactions:</strong> Why matrix dimensions $(M, N, K)$ must be multiples of 64 or 128 bytes to achieve maximum hardware tensor core efficiency, and how token padding or odd vocabulary sizes cause catastrophic speed drops.</li>
  <li><strong>Performance Formula:</strong> GEMM execution cycles and arithmetic efficiency:
  $$\text{FLOPS}_{\text{actual}} = \frac{2 M N K}{\text{Kernel Execution Time (seconds)}} = \eta_{\text{GEMM}} \times \text{FLOPS}_{\text{peak}}$$</li>
  <li><strong>Real-World Case Study:</strong> CUTLASS vs. cuBLAS GEMM profiling on Hopper Tensor Cores.</li>
</ul>

<h3>Chapter E08: Feed-Forward Networks &amp; SwiGLU in Silicon</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Standard MLP ($2$ matrices: $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$) vs. Modern SwiGLU ($3$ matrices: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ffn}}$, $W_{\text{down}} \in \mathbb{R}^{d_{ffn} \times d}$ where $d_{ffn} \approx \frac{8}{3}d$). FFN accounts for approximately $66\%$ of total model parameters and compute FLOPs. Intermediate activation memory footprint and Swish element-wise multiplication ($\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})$).</li>
  <li><strong>Cross-Component Interactions:</strong> The 3-matrix structure increases intermediate activation memory by $50\%$ over standard ReLU/GELU MLPs. How this impacts Activation Checkpointing (Recomputation) during distributed training.</li>
  <li><strong>Performance Formula:</strong> FFN compute FLOPs per token:
  $$\text{FLOPs}_{\text{SwiGLU}} = 3 \times (2 \times d_{\text{model}} \times d_{ffn}) = 6 \times d_{\text{model}} \times \left(\frac{8}{3} d_{\text{model}}\right) = 16 d_{\text{model}}^2$$</li>
  <li><strong>Real-World Case Study:</strong> Fusing $W_{\text{gate}}$ and $W_{\text{up}}$ into a single concatenated GEMM matrix $\left[W_{\text{gate}} \mid W_{\text{up}}\right]$ to double GEMM tile efficiency.</li>
</ul>

<h3>Chapter E09: Normalization &amp; Residual Streams: Fused RMSNorm</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Why LayerNorm and RMSNorm are severely memory-bandwidth bound ($I \approx 1-2 \text{ FLOPs/Byte}$). An unfused implementation reads the residual stream vector from HBM, calculates the root-mean-square, writes normalized activations to HBM, reads them back for linear projection, and reads the old residual stream again to perform element-wise addition.</li>
  <li><strong>Cross-Component Interactions:</strong> <strong>Operator Fusion</strong>: Combining the residual addition and the subsequent RMSNorm into a single GPU kernel that operates entirely within registers and L1 SRAM, completely eliminating two HBM write/read round-trips.</li>
  <li><strong>Performance Formula:</strong> Memory traffic reduction from operator fusion:
  $$\Delta \text{Bytes} = 4 \times (\text{Batch} \times \text{SeqLen} \times d_{\text{model}} \times \text{sizeof(dtype)})$$</li>
  <li><strong>Real-World Case Study:</strong> Comparing memory traces of unfused PyTorch `RMSNorm(x + res)` vs. Triton Fused Add-RMSNorm.</li>
</ul>

<h3>Chapter E10: Positional Embeddings on the Fly: RoPE Kernel Fusion</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Rotary Position Embedding (RoPE) applies 2D rotation matrices to $Q$ and $K$ vectors:
  $$\mathbf{R}_{\Theta, m} \mathbf{x} = \begin{bmatrix} x_1 \cos(m\theta_1) - x_2 \sin(m\theta_1) \\ x_1 \sin(m\theta_1) + x_2 \cos(m\theta_1) \end{bmatrix}$$
  In naive implementations, precomputing and storing large $\cos$ and $\sin$ tables consumes memory bandwidth.</li>
  <li><strong>Cross-Component Interactions:</strong> Fusing RoPE directly into the Query and Key projection GEMM epilogue or the FlashAttention prologue, calculating trigonometric rotations on the fly using fast hardware math registers (`__sinf`, `__cosf`) without touching HBM.</li>
  <li><strong>Performance Formula:</strong> Register pressure vs. memory bandwidth trade-off in RoPE fusion.</li>
  <li><strong>Real-World Case Study:</strong> How long-context extensions (e.g., YaRN, RoPE frequency scaling to 128K context) impact on-the-fly kernel computation.</li>
</ul>

<hr>

<h2 id="module-3">Module 3: Numerical Formats, Precision &amp; Quantization</h2>
<p><em>Balancing numerical dynamic range, mantissa precision, and silicon throughput.</em></p>

<h3>Chapter E11: Number Formats in Silicon: FP32, FP16, BF16, and FP8</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Anatomy of floating point bits: Sign, Exponent (dynamic range), Mantissa (precision).
  FP32 (1+8+23), FP16 (1+5+10), BF16 (1+8+7). Why FP16 suffers catastrophic underflow/overflow during LLM training without loss scaling, while BF16 matches FP32's dynamic range at half the memory footprint.
  The rise of <strong>FP8</strong>: E4M3 (higher precision for forward weights and activations) vs. E5M2 (higher dynamic range for backward gradients).</li>
  <li><strong>Cross-Component Interactions:</strong> How FP8 doubles Tensor Core compute throughput ($2\times$ over FP16/BF16 on Ada/Hopper) and cuts weight/activation memory in half, but requires dynamic per-tensor or per-channel scaling factors.</li>
  <li><strong>Performance Formula:</strong> Memory footprint across formats:
  $$\text{Model Weight Memory (GB)} = \frac{\text{Parameter Count (Billions)} \times \text{Bits per Weight}}{8}$$</li>
  <li><strong>Real-World Case Study:</strong> Mixed-precision training dynamics: Master weights in FP32, forward/backward passes in BF16, optimizer moments in FP32.</li>
</ul>

<h3>Chapter E12: Quantization Strategies: Weight-Only vs. Weight-Activation (W8A8, W4A16, INT4)</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong>
  Weight-Only Quantization (W4A16, W8A16 / AWQ, GPTQ): Weights are stored in INT4/INT8 to minimize HBM traffic, but unpacked/dequantized to FP16 in SRAM registers before Tensor Core execution. Ideal for <strong>memory-bound decoding</strong>.
  Weight-Activation Quantization (W8A8, FP8 / SmoothQuant): Both weights and activations are quantized to 8 bits, enabling native INT8/FP8 Tensor Core matrix multiplication. Ideal for <strong>compute-bound prefill</strong>.</li>
  <li><strong>Cross-Component Interactions:</strong> The activation outlier problem: Emergent high-magnitude feature channels in deep layers destroy naive INT8 quantization; how SmoothQuant migrates quantization difficulty from activations to weights.</li>
  <li><strong>Performance Formula:</strong> Dequantization arithmetic overhead vs. memory bandwidth speedup across different batch size regimes.</li>
  <li><strong>Real-World Case Study:</strong> AWQ (Activation-aware Weight Quantization) vs. GPTQ latency and perplexity trade-offs on 70B models.</li>
</ul>

<h3>Chapter E13: KV Cache Quantization: Compressing Context to FP8 and INT4</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Quantizing cached Key and Value tensors from 16 bits to 8 bits (FP8 E4M3 or INT8) or 4 bits. Storage layout, asymmetric per-head scaling factors, and zero-points.</li>
  <li><strong>Cross-Component Interactions:</strong> Cutting KV cache size by $50\%$ (FP8) or $75\%$ (INT4) doubles or quadruples the maximum concurrent batch size in serving systems, directly delaying out-of-memory crashes during multi-turn chats.</li>
  <li><strong>Performance Formula:</strong> Quantized attention error propagation:
  $$\Delta \text{Attention} = \left\| \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right)V - \text{softmax}\left(\frac{Q \hat{K}^\top}{\sqrt{d}}\right)\hat{V} \right\|$$</li>
  <li><strong>Real-World Case Study:</strong> Implementing FP8 KV cache in vLLM and measuring throughput gain vs. needle-in-a-haystack retrieval accuracy.</li>
</ul>

<hr>

<h2 id="module-4">Module 4: Distributed Systems &amp; Scaling Parallelism</h2>
<p><em>Splitting giant brains across thousands of communicating GPUs without choking on networks.</em></p>

<h3>Chapter E14: Distributed Communication Primitives: All-Reduce, All-Gather, Reduce-Scatter</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Physical network topology: Ring, Tree, and Mesh communication algorithms. Collective operations:
  <strong>All-Reduce</strong> (sum vectors across all GPUs), <strong>All-Gather</strong> (gather shards into full vectors), <strong>Reduce-Scatter</strong> (reduce vectors and scatter shards to distinct ranks), and <strong>Send/Recv</strong> point-to-point. Ring All-Reduce communication volume: $2 \frac{N-1}{N} S$ bytes transferred per rank.</li>
  <li><strong>Cross-Component Interactions:</strong> How collective communications compete with compute kernels for PCIe/NVLink bandwidth, and the necessity of overlapping communication with compute via CUDA streams.</li>
  <li><strong>Performance Formula:</strong> Ring All-Reduce communication time:
  $$T_{\text{AllReduce}} = 2(N-1) \alpha + 2 \frac{N-1}{N} \frac{S}{\text{Bus Bandwidth}}$$
  where $\alpha$ is network latency and $S$ is message size in bytes.</li>
  <li><strong>Real-World Case Study:</strong> NCCL (NVIDIA Collective Communications Library) tuning and NVLink vs. InfiniBand scaling barriers.</li>
</ul>

<h3>Chapter E15: Tensor Parallelism (TP): Megatron-LM in Silicon</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Sharding individual weight matrices across $N$ GPUs.
  Self-Attention: Split $W_Q, W_K, W_V$ column-wise (no communication needed); split $W_O$ row-wise (requires 1 All-Reduce across TP ranks).
  FFN: Split $W_{\text{gate}}, W_{\text{up}}$ column-wise; split $W_{\text{down}}$ row-wise (requires 1 All-Reduce across TP ranks).
  Total communication: Exactly 2 All-Reduce operations per Transformer layer. Sequence Parallelism (SP) optimization: sharding LayerNorm/RMSNorm activations to eliminate redundant compute.</li>
  <li><strong>Cross-Component Interactions:</strong> TP requires ultra-fast, low-latency interconnects (NVLink); running TP across slow Ethernet or inter-node networks creates severe communication bottlenecks that degrade performance. TP degree is practically bounded by the number of GPUs per single server node (usually 8).</li>
  <li><strong>Performance Formula:</strong> TP communication byte volume per layer:
  $$\text{Bytes}_{\text{TP}} = 4 \times B \times S \times d_{\text{model}} \times \text{sizeof(dtype)}$$</li>
  <li><strong>Real-World Case Study:</strong> Step-by-step tensor shape walkthrough of a forward pass in Megatron-LM across 8 GPUs.</li>
</ul>

<h3>Chapter E16: Pipeline Parallelism (PP): 1F1B Scheduling &amp; Bubble Analysis</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Partitioning layers sequentially across stages (e.g., layers 1-10 on GPU 0, layers 11-20 on GPU 1). The naive pipeline bubble ($F_{\text{bubble}} = \frac{p-1}{p}$).
  Advanced scheduling: <strong>1F1B</strong> (One Forward, One Backward) steady-state scheduling to bound activation memory. Interleaved 1F1B schedules to compress the bubble fraction.</li>
  <li><strong>Cross-Component Interactions:</strong> How pipeline stage boundaries dictate activation memory buffers: Stage 0 must stash activations for $p$ microbatches while waiting for the backward pass, increasing memory pressure on early GPUs.</li>
  <li><strong>Performance Formula:</strong> 1F1B pipeline bubble fraction:
  $$\text{Bubble Fraction} = \frac{p - 1}{m + p - 1}$$
  where $p$ is pipeline stages and $m$ is the number of microbatches.</li>
  <li><strong>Real-World Case Study:</strong> DeepSpeed vs. Megatron-LM pipeline scheduling benchmarks on 64-GPU clusters.</li>
</ul>

<h3>Chapter E17: Data Parallelism &amp; Zero Redundancy: ZeRO-1, 2, 3 and FSDP</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Standard Data Parallelism (DDP) replicates full model weights, optimizer states, and gradients on every GPU.
  Rajbhandari et al. (ZeRO / PyTorch FSDP):
  <strong>ZeRO-1:</strong> Shards optimizer states ($4\times$ memory reduction).
  <strong>ZeRO-2:</strong> Shards optimizer states and gradients ($8\times$ memory reduction).
  <strong>ZeRO-3 (Full Sharding):</strong> Shards optimizer states, gradients, and model parameters. Dynamically fetches weights via All-Gather right before forward/backward compute, then immediately discards them.</li>
  <li><strong>Cross-Component Interactions:</strong> ZeRO-3 completely eliminates parameter memory bottlenecks, enabling 70B+ model training on consumer clusters, but increases total communication volume by $50\%$ (requires an extra All-Gather during forward pass).</li>
  <li><strong>Performance Formula:</strong> Memory consumption per GPU under ZeRO stages:
  $$M_{\text{ZeRO-3}} = \frac{M_{\text{params}} + M_{\text{grads}} + M_{\text{opt}}}{N_{\text{GPUs}}} + M_{\text{activations}}$$</li>
  <li><strong>Real-World Case Study:</strong> PyTorch FSDP configuration: `HYBRID_SHARD` combining intra-node TP/FSDP with inter-node ZeRO.</li>
</ul>

<h3>Chapter E18: Context &amp; Sequence Parallelism: Ring Attention &amp; 1M+ Context</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> When sequence length $S$ reaches 100K-1M+ tokens, activations for a single sequence cannot fit into the HBM of a single GPU.
  <strong>Ring Attention</strong> (Liu et al.): Shards the sequence across a ring of GPUs. While GPU $i$ computes attention between its local Query block and local Key/Value block, it asynchronously transmits its Key/Value block to GPU $i+1$ via point-to-point ring communication.</li>
  <li><strong>Cross-Component Interactions:</strong> Perfect overlap: If local attention computation time $\ge$ network transmission time of the KV block, the communication latency of 1-million-token contexts is completely hidden behind compute!</li>
  <li><strong>Performance Formula:</strong> Ring Attention overlap condition:
  $$\text{Time}_{\text{compute}}\left(\frac{S}{N} \times \frac{S}{N} \text{ tile}\right) \ge \text{Time}_{\text{transfer}}\left(\frac{S}{N} \text{ KV block}\right)$$</li>
  <li><strong>Real-World Case Study:</strong> DeepSeek and LLaMA-3 128K context parallelism topologies.</li>
</ul>

<hr>

<h2 id="module-5">Module 5: High-Throughput Inference Engines &amp; Serving</h2>
<p><em>Building low-latency, high-concurrency production serving infrastructure.</em></p>

<h3>Chapter E19: The Two Inference Regimes: Prefill vs. Decode</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> The fundamental operational asymmetry of LLM inference:
  <strong>Prefill Phase (Prompt Evaluation):</strong> Ingests all $S$ prompt tokens simultaneously. High operational intensity ($I > 100$). Highly compute-bound. Saturates Tensor Cores. Primary metric: <strong>Time to First Token (TTFT)</strong>.
  <strong>Decode Phase (Autoregressive Generation):</strong> Ingests exactly 1 token per request per step. Very low operational intensity ($I \approx 1-10$). Severely memory-bandwidth bound. Primary metric: <strong>Inter-Token Latency (ITL)</strong>.</li>
  <li><strong>Cross-Component Interactions:</strong> How running prefill and decode requests inside the same batch creates the <strong>Prefill-Decode Interference Problem</strong>: massive prefill matrix multiplies spike latency for ongoing decode streams.</li>
  <li><strong>Performance Formula:</strong> Decode latency bound by memory bandwidth:
  $$T_{\text{step}} \approx \frac{\text{Model Weights (Bytes)} + \text{KV Cache (Bytes)}}{\text{HBM Bandwidth (Bytes/s)}}$$</li>
  <li><strong>Real-World Case Study:</strong> Splitwise &amp; Mooncake architectures: Disaggregating prefill nodes and decode nodes onto separate hardware clusters.</li>
</ul>

<h3>Chapter E20: Continuous (In-Flight) Batching &amp; Dynamic Scheduling</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Static batching locks requests into a fixed tensor grid, padding shorter sequences with zeroes until the longest sequence completes. This wastes up to $70\%$ of GPU compute on useless padding tokens.
  <strong>Continuous Batching</strong> (Orca / vLLM): Operates at the iteration level rather than the request level. As soon as a request emits an `<EOS>` token, its slot and KV pages are released, and a new waiting request is injected immediately into the next forward pass.</li>
  <li><strong>Cross-Component Interactions:</strong> Eliminates static tensor padding entirely; requires dynamic tensor reshaping and specialized attention kernels that handle ragged, non-uniform sequence lengths without memory reallocation.</li>
  <li><strong>Performance Formula:</strong> Token throughput efficiency:
  $$\text{Throughput} = \frac{\sum_{i=1}^B \text{Tokens Generated}_i}{\text{Total Execution Time}}$$</li>
  <li><strong>Real-World Case Study:</strong> Profiling GPU utilization under static batching vs. continuous batching in vLLM.</li>
</ul>

<h3>Chapter E21: Speculative Decoding: Draft-Verify Silicon Acceleration</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Leviathan et al. &amp; Chen et al.: Exploiting the memory-bound nature of the decode phase. Generating 1 token from a large 70B model requires reading 140 GB of weights from HBM. A tiny 1B draft model can generate $K$ speculative tokens rapidly in low-capacity memory. The 70B target model then evaluates all $K$ tokens in a <strong>single parallel forward pass</strong> (prefill regime, high arithmetic intensity).</li>
  <li><strong>Cross-Component Interactions:</strong> Speculative tree structures (Medusa, EAGLE) vs. autoregressive draft models. How target model acceptance rate $\alpha$ determines net wall-clock speedup.</li>
  <li><strong>Performance Formula:</strong> Expected wall-clock speedup:
  $$\text{Speedup} = \frac{1 + \alpha + \alpha^2 + \dots + \alpha^K}{1 + \frac{c_{\text{draft}}}{c_{\text{target}}} \cdot K}$$</li>
  <li><strong>Real-World Case Study:</strong> Measuring latency reduction and token acceptance histograms on LLaMA-3 70B paired with LLaMA-3 8B.</li>
</ul>

<h3>Chapter E22: Chunked Prefill &amp; Prefix Caching</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong>
  <strong>Chunked Prefill:</strong> Slicing long prompt prefills into smaller chunks (e.g. 512 tokens) and co-scheduling them alongside decode requests to smooth out compute spikes and bound ITL.
  <strong>Automatic Prefix Caching (APC):</strong> Radix-tree caching of KV pages for common prompt prefixes (system instructions, few-shot examples, multi-turn chat history).</li>
  <li><strong>Cross-Component Interactions:</strong> APC turns $O(S)$ compute and HBM allocation into an instantaneous $O(1)$ pointer copy, dropping TTFT from seconds to milliseconds for multi-turn conversations.</li>
  <li><strong>Performance Formula:</strong> TTFT reduction with prefix hit rate $h$:
  $$\text{TTFT}_{\text{cached}} = (1 - h) \cdot \text{TTFT}_{\text{full}} + \epsilon_{\text{lookup}}$$</li>
  <li><strong>Real-World Case Study:</strong> vLLM Radix Tree cache eviction algorithms (LRU vs. LFU) under high system load.</li>
</ul>

<hr>

<h2 id="module-6">Module 6: Mixture of Experts (MoE) Engineering</h2>
<p><em>Scaling parameters to trillions without multiplying compute: sparse routing, dispatch, and communication.</em></p>

<h3>Chapter E23: MoE Architecture in Systems: Top-K Gating &amp; Capacity Factor</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Replacing dense FFN layers with $E$ parallel expert networks, routed dynamically by a gating network:
  $$\mathbf{y} = \sum_{i \in \text{TopK}(G(\mathbf{x}))} g_i(\mathbf{x}) \text{Expert}_i(\mathbf{x})$$
  Active parameters vs. total parameters (e.g., Mixtral 8x7B has 47B total parameters, but only 13B active per token; DeepSeek-V3 has 671B total parameters, but only 37B active per token).
  The <strong>Expert Capacity Factor</strong>: Enforcing fixed buffer sizes per expert to prevent static GPU allocation blowups; managing token dropping when experts overflow.</li>
  <li><strong>Cross-Component Interactions:</strong> Load balancing loss terms prevent expert collapse, but uneven real-world prompt distributions cause expert load imbalance, forcing some GPUs to stall while waiting for overloaded experts to finish.</li>
  <li><strong>Performance Formula:</strong> Compute vs. memory capacity trade-off:
  $$\text{FLOPs}_{\text{MoE}} \ll \text{FLOPs}_{\text{Dense}(P_{\text{total}})}, \quad \text{Memory}_{\text{MoE}} \equiv \text{Memory}_{\text{Dense}(P_{\text{total}})}$$</li>
  <li><strong>Real-World Case Study:</strong> Mixtral 8x7B and DeepSeek-V3 routing topology and auxiliary-loss-free load balancing.</li>
</ul>

<h3>Chapter E24: All-to-All Dispatch &amp; Combine: Distributed MoE Bottlenecks</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> In Expert Parallelism (EP), different experts reside on different physical GPUs. When token $t$ on GPU 0 routes to Expert $k$ on GPU 7, the token activation vector must be physically shipped across the network.
  The two communication barriers: <strong>All-to-All Dispatch</strong> (routing tokens to expert ranks) and <strong>All-to-All Combine</strong> (returning expert outputs to original ranks).</li>
  <li><strong>Cross-Component Interactions:</strong> All-to-All communication creates massive all-to-all crossbar traffic on inter-node switches. If network bandwidth is insufficient, MoE speedups vanish, and GPUs spend 60%+ of execution time waiting on network collective barriers.</li>
  <li><strong>Performance Formula:</strong> All-to-All communication volume per layer:
  $$\text{Bytes}_{\text{All-to-All}} = 2 \times K \times B \times S \times d_{\text{model}} \times \text{sizeof(dtype)}$$
  where $K$ is the number of active experts selected per token.</li>
  <li><strong>Real-World Case Study:</strong> DeepSeek DualPipe overlap: Bi-directional overlapping of All-to-All communication with FFN computation.</li>
</ul>

<hr>

<h2 id="module-7">Module 7: Auditing, Profiling &amp; Hardware Efficiency Metrics</h2>
<p><em>Quantifying engineering excellence: MFU, memory accounting, and profiling tools.</em></p>

<h3>Chapter E25: The Complete Model Memory Budget Equation</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> The definitive memory breakdown of modern LLM systems.
  Training memory budget:
  $$M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{temp\_buffers}}$$
  For FP16 training with Adam: Weights ($2P$ bytes), Gradients ($2P$ bytes), Adam First/Second Moments ($8P$ bytes), Master FP32 Weights ($4P$ bytes) $\to 16P$ bytes total before counting activations!
  Inference memory budget:
  $$M_{\text{inference}} = M_{\text{weights}} + M_{\text{KV\_Cache}} + M_{\text{scratchpad}}$$</li>
  <li><strong>Cross-Component Interactions:</strong> How Activation Checkpointing (recomputing activations during backward pass instead of storing them) trades $33\%$ extra compute FLOPs for a massive $5-10\times$ reduction in activation memory.</li>
  <li><strong>Performance Formula:</strong> Activation memory per Transformer layer under selective recomputation.</li>
  <li><strong>Real-World Case Study:</strong> Memory audit spreadsheet for training a 70B parameter model on 128x H100 GPUs.</li>
</ul>

<h3>Chapter E26: Measuring True Performance: MFU, HFU, TTFT, and ITL</h3>
<ul>
  <li><strong>Hardware Mechanics:</strong> Why raw TFLOPS ratings on hardware datasheets are marketing fictions.
  <strong>Model FLOPs Utilization (MFU):</strong> The ratio of theoretical mathematical FLOPs required by the model architecture to the peak theoretical hardware FLOPs of the GPU cluster:
  $$\text{MFU} = \frac{\text{Theoretical Tokens/Sec} \times 6P}{\text{Total Cluster Peak FLOPS}}$$
  Hardware FLOPs Utilization (HFU): Includes recomputation and backward overhead. Industry benchmarks: Good implementations achieve 40-50% MFU; world-class implementations achieve 55-65% MFU.</li>
  <li><strong>Cross-Component Interactions:</strong> How bubble fractions in PP, communication stalls in TP, and small batch sizes in decode directly degrade MFU.</li>
  <li><strong>Performance Formula:</strong> Deriving the standard $6P$ FLOPs per token rule of thumb for training and $2P$ FLOPs per token for inference.</li>
  <li><strong>Real-World Case Study:</strong> End-to-end profiling session using PyTorch Profiler, NVIDIA Nsight Systems (nsys), and NVIDIA Nsight Compute (ncu).</li>
</ul>

<hr>

<h2 id="metrics-reference">5. Systems Metrics Reference Table</h2>

<p>Every decision in LLM engineering impacts one or more of these core production metrics:</p>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table E.3:</strong> Core LLM Engineering Metrics, Definitions, and Primary Influencing Components.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="15%">Metric</th>
      <th align="left" width="20%">Definition</th>
      <th align="left" width="30%">Primary Physical Bottleneck</th>
      <th align="left" width="35%">Key Influencing Architecture Knobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>TTFT</strong></td>
      <td>Time to First Token (prefill latency)</td>
      <td>Compute bound (Tensor Core GEMM FLOPS)</td>
      <td>Prompt length, Chunked Prefill, Prefix Caching, Tensor Parallelism</td>
    </tr>
    <tr>
      <td><strong>ITL</strong></td>
      <td>Inter-Token Latency (time per output token)</td>
      <td>Memory bandwidth bound (HBM byte transfer rate)</td>
      <td>Model parameter size, Quantization (W4/W8), GQA vs. MHA, Speculative Decoding</td>
    </tr>
    <tr>
      <td><strong>Throughput</strong></td>
      <td>Total tokens processed / generated per second per GPU</td>
      <td>Batch size concurrency and memory capacity</td>
      <td>PagedAttention, Continuous Batching, KV Cache Quantization, FP8 compute</td>
    </tr>
    <tr>
      <td><strong>MFU</strong></td>
      <td>Model FLOPs Utilization (efficiency vs. peak silicon)</td>
      <td>Pipeline bubbles, collective communication overhead, memory stalls</td>
      <td>FlashAttention, Kernel Fusion, 1F1B scheduling, ZeRO sharding overlap</td>
    </tr>
    <tr>
      <td><strong>KV Footprint</strong></td>
      <td>HBM consumed by cached past attention states</td>
      <td>HBM capacity per accelerator</td>
      <td>GQA group ratio, MLA latent dimension, FP8 KV caching, Context length</td>
    </tr>
  </tbody>
</table>

<hr>

<nav aria-label="Course Navigation">
  <p>
    &larr; <a href="../curriculum.md">Mathematical Foundations Curriculum</a> &bull;
    <a href="../index.html">Master Course Overview</a> &bull;
    <strong>Engineering Subproject Curriculum</strong>
  </p>
</nav>
