# The Math Behind LLM Inference: Master Curriculum

<nav aria-label="Table of Contents">
  <p>
    <strong>Inference Track Navigation:</strong>
    <a href="#pedagogy">5-Step Pedagogy</a> &bull;
    <a href="#foundations">Hardware &amp; Execution Physics</a> &bull;
    <a href="#memory">KV Cache &amp; PagedAttention</a> &bull;
    <a href="#kernels">IO-Aware Kernels</a> &bull;
    <a href="#scheduling">Serving Dynamics</a> &bull;
    <a href="#speculative">Speculative Decoding</a> &bull;
    <a href="#quantization">Quantization &amp; Precision</a> &bull;
    <a href="#distributed">Distributed Serving</a>
  </p>
</nav>

<hr>

<fieldset>
<legend><strong>Core Mission: The Physics of LLM Inference</strong></legend>
<p>Inference is not just running a trained model forward. It is a distinct computational and mathematical discipline governed by the physical laws of modern computer hardware: GPU memory bandwidth, SRAM capacity, tensor core arithmetic intensity, and interconnect latency.</p>
<p>Every chapter in this curriculum answers two foundational questions:</p>
<ol>
  <li><em>What physical intuition makes this operational bottleneck instantly obvious to a child?</em></li>
  <li><em>Where does the exact mathematical equation come from, and why did systems researchers write it this way?</em></li>
</ol>
</fieldset>

---

<h2 id="pedagogy">The Mandatory 5-Step Pedagogy</h2>

Every chapter in the Inference Curriculum follows the unshakeable 6-step learning ladder:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table I.1:</strong> The 6-step pedagogical learning sequence for LLM inference.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center" width="10%">Step</th>
      <th align="left" width="22%">Section Title</th>
      <th align="left" width="38%">What You Learn</th>
      <th align="left" width="30%">Tactile Physical Metaphor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Step 1</strong></td>
      <td><strong>3-Year-Old Intuition</strong></td>
      <td>Pure physical metaphor with zero mathematical jargon</td>
      <td>Moving trucks, conveyor belts, kitchen cutting boards, hotel booking</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 2</strong></td>
      <td><strong>The Bridging Question</strong></td>
      <td>Translating physical mechanical games into hardware numbers</td>
      <td>Arithmetic intensity, memory bytes, clock cycles, FLOP ratios</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 3</strong></td>
      <td><strong>The Exact Math &amp; Formula</strong></td>
      <td>The genuine equations governing modern inference engines</td>
      <td>Roofline models, online softmax recurrences, acceptance probabilities</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 4</strong></td>
      <td><strong>Where Did It Come From?</strong></td>
      <td>Historical engineering origin and failure modes</td>
      <td>Why naive algorithms collapsed at long context and what broke first</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 5</strong></td>
      <td><strong>Concrete Toy Example</strong></td>
      <td>Hand-calculated arithmetic with tiny matrices and vectors</td>
      <td>Step-by-step additions, multiplications, and byte counts</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 6</strong></td>
      <td><strong>Core Takeaway</strong></td>
      <td>1–2 sentence conceptual punchline</td>
      <td>The architectural anchor for production serving systems</td>
    </tr>
  </tbody>
</table>

---

<h2 id="curriculum-overview">The 9-Chapter Inference Curriculum Overview</h2>

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE LLM INFERENCE COMPUTATIONAL PIPELINE                        │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. HARDWARE &amp; PHYSICS    │ 2. MEMORY &amp; CACHE        │ 3. ACCELERATION &amp; SERVING        │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Ch 20: Prefill vs Decode │ Ch 21: KV Cache &amp; GQA    │ Ch 23: FlashAttention (SRAM)     │
│        (Roofline Model)  │        (The Memory Wall) │ Ch 24: FlashDecoding (Split-K)   │
│                          │ Ch 22: PagedAttention    │ Ch 25: Continuous Batching       │
│                          │        (Virtual Memory)  │ Ch 26: Speculative Decoding      │
│                          │                          │ Ch 27: Quantization (AWQ/FP8)    │
│                          │                          │ Ch 28: Distributed Serving (TP)  │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
</pre>
<figcaption><strong>Figure I.1:</strong> The structural taxonomy of the LLM Inference Curriculum.</figcaption>
</figure>

---

<h3 id="foundations">Track 1: Hardware Physics &amp; The Core Dichotomy</h3>

#### [Chapter 20: The Dual-Phase Engine (Prefill vs. Decode &amp; The Roofline Model)](20-prefill-vs-decode/index.html)
- **The Metaphor**: The Moving Truck (Prefill: packing 1,000 boxes into a truck in one massive push) vs. The Delivery Courier (Decode: driving back and forth between warehouse and home to deliver one tiny letter per trip).
- **The Math**:
  - Operational Intensity: $I = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$.
  - The GPU Roofline Model: Attainable performance $P = \min\left(P_{\text{peak}}, I \times B_{\text{peak}}\right)$.
  - The Critical Ridge Point $I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}$.
  - Why Prefill has $I \gg I^*$ (Compute-Bound, GEMM, saturated Tensor Cores), while Decode has $I \approx 1 \ll I^*$ (Memory-Bandwidth-Bound, GEMV, 98% idle compute cores).
- **Formula Origin**: Williams, Waterman, &amp; Patterson (2009) Roofline Model applied to autoregressive sequence generation.

---

<h3 id="memory">Track 2: Memory Dynamics &amp; Virtual Cache</h3>

#### [Chapter 21: The Memory Wall (KV Cache Mechanics &amp; GQA / MLA Compression)](21-kv-cache-and-memory-wall/index.html)
- **The Metaphor**: The Accountant's Scratchpad vs. Re-reading the Entire Library from the first page every time you write a single word.
- **The Math**:
  - Recursive KV Cache concatenation: $\mathbf{K}_{1:t} = [\mathbf{K}_{1:t-1}; \mathbf{k}_t]$, $\mathbf{V}_{1:t} = [\mathbf{V}_{1:t-1}; \mathbf{v}_t]$.
  - Exact KV Cache RAM equation:
    $$\text{RAM}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{ctx}} \times b \times p$$
  - The 128k context crisis (312.50 GB for MHA on 70B models).
  - Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and DeepSeek Multi-Head Latent Attention (MLA) low-rank matrix projection $\mathbf{c}_t^{KV} = \mathbf{x}_t \mathbf{W}_{DKV}$.
- **Formula Origin**: Shazeer (2019), Ainslie et al. (2023), DeepSeek-AI (2024).

#### [Chapter 22: PagedAttention &amp; Virtual Memory (vLLM &amp; Zero Fragmentation)](22-paged-attention/index.html)
- **The Metaphor**: Reserving an entire 500-room hotel for a guest who might stay 1 day or 10 days (static allocation: 80% empty rooms) vs. Assigning room keys one night at a time (PagedAttention: allocate block by block as guests arrive).
- **The Math**:
  - Logical blocks of size $B$ mapped to non-contiguous physical DRAM blocks via page table $\mathcal{T}(r, b_{\text{logical}}) = b_{\text{physical}}$.
  - Paged attention kernel gather loops without memory copying.
  - Mathematical elimination of external fragmentation; bounding internal fragmentation to $\le (B-1)/T$.
  - Copy-on-Write (CoW) fork dynamics for tree search and parallel sampling.
- **Formula Origin**: Kwon et al. (vLLM, SOSP 2023) adapting 1960s operating system virtual memory to tensor serving.

---

<h3 id="kernels">Track 3: IO-Aware Attention &amp; Decode Parallelism</h3>

#### [Chapter 23: The IO-Aware Speedup (FlashAttention &amp; Online Softmax)](23-flash-attention/index.html)
- **The Metaphor**: Cooking in a compact kitchen: instead of walking back and forth to the basement pantry 1,000 times for every onion slice (HBM), keep the ingredients right on your cutting board (SRAM), finish the dish, and take only the final meal to the table.
- **The Math**:
  - Memory hierarchy: SRAM (~19 TB/s) vs. HBM (~3.35 TB/s).
  - Online Softmax recurrence relations (Milakov &amp; Gimelshtein):
    $$m_{\text{new}} = \max(m, x), \quad d_{\text{new}} = d \cdot e^{m - m_{\text{new}}} + e^{x - m_{\text{new}}}$$
  - Block tiling: Accumulating unnormalized outputs and scaling by running denominator without materializing $T \times T$ intermediate attention matrices.
  - IO Complexity reduction from $O(T^2)$ to $O(T^2 d / M_{\text{SRAM}})$.
- **Formula Origin**: Milakov &amp; Gimelshtein (2018), Tri Dao et al. (2022, 2023, 2024 - FlashAttention 1, 2, 3).

#### [Chapter 24: Decode Parallelism (FlashDecoding &amp; Split-K Attention)](24-flash-decoding/index.html)
- **The Metaphor**: 100 builders watching 1 builder dig a narrow trench. FlashDecoding gives all 100 builders their own shovel along different parts of the trench, then merges their progress.
- **The Math**:
  - Why FlashAttention underutilizes GPUs during decode ($T_Q = 1$ provides too little parallelism to fill 132 SMs).
  - Split-K attention: Partitioning the historical KV sequence dimension into $S$ independent chunks.
  - Computing partial softmax $(m^{(s)}, \ell^{(s)}, \mathbf{o}^{(s)})$ per SM block in parallel, followed by a hierarchical reduction pass across chunks.
- **Formula Origin**: Tri Dao et al. (Flash-Decoding, 2023).

---

<h3 id="scheduling">Track 4: Serving Dynamics &amp; Batching Schedulers</h3>

#### [Chapter 25: Cellular Scheduling (Continuous Batching &amp; Chunked Prefill)](25-continuous-batching/index.html)
- **The Metaphor**: The Subway Train vs. The Rollercoaster. In a rollercoaster (static batching), everyone must wait until the slowest passenger finishes. In a subway train (continuous batching), doors open at every station: passengers board, passengers leave, train never stops.
- **The Math**:
  - The static batching padding waste ratio: $1 - \sum T_i / (B \times \max T_i)$.
  - Iteration-level scheduling: $\mathcal{B}_t = (\mathcal{B}_{t-1} \setminus \mathcal{F}_{t-1}) \cup \mathcal{A}_t$.
  - Chunked prefill (Sarathi-Serve): Chopping large prompts into $C_{\text{chunk}}$ slices to interleave prompt prefill with decode steps, eliminating latency bubbles and TTFT spikes.
- **Formula Origin**: Yu et al. (Orca, OSDI 2022), Agrawal et al. (Sarathi-Serve, 2024).

#### [Hands-on Lab 05: Streaming KV &amp; Continuous Batching Engine](25b-lab-streaming-engine/index.html)
- **Architecture**: A 200-line pure Python vLLM-style serving kernel featuring PagedAttention block manager, copy-on-write prefix sharing, and iteration-level continuous scheduler.
- **Verification**: Zero-dependency standard-library Python engine with automated test suite demonstrating 0.0% bubble waste.

---

<h3 id="speculative">Track 5: Breaking the Autoregressive Barrier</h3>

#### [Chapter 26: Breaking the Sequential Barrier (Speculative Decoding &amp; Verification Math)](26-speculative-decoding/index.html)
- **The Metaphor**: An ambitious apprentice writing 5 sentences in pencil, while the master craftsman looks over all 5 sentences in a single glance, approves the first 4, fixes the 5th in ink, and discards the rest.
- **The Math**:
  - Draft model $q(x)$ generating $\gamma$ candidate tokens; target model $p(x)$ evaluating all $\gamma+1$ tokens concurrently in a single forward step.
  - Acceptance criterion: $\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$.
  - Residual recovery sampling: $p_{\text{res}}(x) = \frac{\max(0, p(x) - q(x))}{\sum \max(0, p - q)}$.
  - Mathematical proof that output distribution strictly matches target distribution: $\mathbb{P}(X=x) \equiv p(x)$.
  - Tree attention verification masks (Medusa, Eagle).
- **Formula Origin**: Leviathan et al. (2023), Chen et al. (2023).

---

<h3 id="quantization">Track 6: Precision &amp; Distributed Serving</h3>

#### [Chapter 27: Squeezing the Numbers (Quantization: AWQ, SmoothQuant, FP8, &amp; INT4)](27-quantization/index.html)
- **The Metaphor**: Vacuum-sealing heavy winter coats (weights) into flat bags, while wrapping fragile glass ornaments (outlier activation channels) in protective bubble wrap.
- **The Math**:
  - Affine &amp; Symmetric quantization: $q = \operatorname{clip}(\lfloor x/s \rceil + z)$, $\hat{x} = s(q - z)$.
  - The Outlier Channel Phenomenon: Why $>6.7\text{B}$ LLMs develop systematic $100\times$ activation spikes in specific channels.
  - SmoothQuant: Migrating outlier scale factors from activations to weights via $\mathbf{Y} = (\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1})(\operatorname{diag}(\mathbf{s}) \mathbf{W})$.
  - AWQ: Preserving salient 1% weights via activation-aware loss minimization.
  - Numerical formats: FP8 (E4M3 vs. E5M2), INT8, INT4, NF4.
- **Formula Origin**: Dettmers et al. (2022), Xiao et al. (SmoothQuant, 2023), Lin et al. (AWQ, 2023).

#### [Hands-on Lab 06: Speculative Decoding &amp; INT4 Quantization Engine](27b-lab-speculative-engine/index.html)
- **Architecture**: A 220-line pure Python inference accelerator implementing uniform symmetric INT4 weight quantization and lossless speculative rejection sampling with residual recovery.
- **Verification**: Zero-dependency standard-library Python engine with automated test suite achieving 3.99x memory compression and 1.75x-3.0x net generation speedup.

#### [Chapter 28: Distributed Serving (Tensor Parallelism &amp; Prefill-Decode Disaggregation)](28-distributed-serving/index.html)
- **The Metaphor**: The Assembly Line Factory: Splitting a giant engine blueprint across 8 workstations (Tensor Parallelism) vs. Separating the heavy smelting foundry (Prefill cluster) from the delicate assembly room (Decode cluster).
- **The Math**:
  - Megatron-LM inference tensor decomposition: ColumnParallelLinear $\to$ RowParallelLinear with All-Reduce communication costs.
  - Communication latency: $T_{\text{allreduce}} = 2\left(\frac{P-1}{P}\right) \frac{M}{B_{\text{NVLink}}} + 2(P-1) \alpha_{\text{lat}}$.
  - Prefill-Decode (PD) Disaggregation (Splitwise / Mooncake / DistServe): Separating compute-bound prefill workers from memory-bound decode workers over RDMA networks.
- **Formula Origin**: Shoeybi et al. (Megatron-LM, 2019), Patel et al. (Splitwise, 2024), Zhong et al. (DistServe, 2024).

---

<fieldset>
<legend><strong>Inference Mastery Summary</strong></legend>
<p>By mastering these 9 chapters, you understand the exact mathematical, architectural, and systems principles that power modern inference engines (vLLM, TensorRT-LLM, SGLang, TGI). You can calculate exact latency bounds, design memory allocations, and optimize throughput from first principles.</p>
</fieldset>
