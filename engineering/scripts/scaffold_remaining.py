import os

modules_data = {
    # Module 2
    'engineering/docs/module-2/07-gemm-tensor-cores.md': """# Chapter E07: GEMM at Scale: Tensor Cores, Systolic Arrays, and Tiling

## Step 1: Hardware Intuition
Imagine multiplying two 1000-page accounting ledgers by hand. If you proceed line by line, calculating one number at a time, you will spend weeks flipping pages back and forth. Instead, a systolic array is like an automated stamping press: numbers flow like rhythmic pulses across a grid of tiny stamping gears, and intermediate results accumulate without ever leaving the press.

## Step 2: Silicon Micro-Mechanics
- **Tensor Core Architecture:** Hardware matrix-multiply-accumulate (MMA) instructions ($16 \\times 16 \\times 16$ micro-tiles per cycle).
- **Hierarchical Tiling:** Threadblock tiling in SRAM, warp-level tiling in registers, and thread-level accumulation.
- **Dimension Quantization:** Matrix dimensions $(M, N, K)$ must align with multiples of 64 or 128 elements to avoid unaligned memory access and uncoalesced memory stalls.

## Step 3: Cross-Component Coupling
- **Vocabulary & Embedding Alignment:** If vocabulary size $|V|$ is not a multiple of 64 or 128 (e.g. 32,001), the final unembedding projection GEMM drops in arithmetic efficiency. Padding $|V|$ to 32,064 restores peak Tensor Core performance.
- **Batching Effects:** Larger batch sizes during prefill maximize GEMM tile occupancy.

## Step 4: The Exact Performance Formula
$$\\text{GEMM FLOPs} = 2 M N K$$
$$\\text{Tensor Core Efficiency } \\eta_{\\text{GEMM}} = \\frac{2 M N K}{T_{\\text{exec}} \\times P_{\\text{peak}}}$$

## Step 5: Concrete Benchmark Walkthrough
Benchmarking a projection matrix ($M=4096, K=4096, N=11008$) on an H100 GPU:
- FLOPs: $2 \\times 4096 \\times 4096 \\times 11008 = 3.69 \\times 10^{11}\\text{ FLOPs} = 369.4\\text{ GFLOPs}$.
- At 989 TFLOPS dense FP16 peak, minimum theoretical execution time is $0.373\\text{ ms}$.
- Realized CUTLASS kernel runtime: $0.46\\text{ ms}$ (81% Tensor Core efficiency).

## Step 6: Core Systems Takeaway
> Tensor Cores require regular, aligned data dimensions. Padding tensor shapes to multiples of 64 or 128 elements prevents hardware pipeline stalls and guarantees maximum arithmetic throughput.
""",

    'engineering/docs/module-2/08-swiglu-ffn-in-silicon.md': """# Chapter E08: Feed-Forward Networks & SwiGLU in Silicon

## Step 1: Hardware Intuition
In a standard restaurant kitchen, a dish goes through two preparation stages: chopping and seasoning. In a modern kitchen (SwiGLU), the dish passes through three parallel prep stations: one chops the vegetables, one whips a savory glaze, and an air-valve blender instantly mixes them together before baking. This gives richer flavor, but requires more counter space to hold ingredients simultaneously.

## Step 2: Silicon Micro-Mechanics
- **3-Matrix Structure:** SwiGLU uses $W_{\\text{gate}}, W_{\\text{up}} \\in \\mathbb{R}^{d \\times d_{ffn}}$ and $W_{\\text{down}} \\in \\mathbb{R}^{d_{ffn} \\times d}$.
- Intermediate dimension: $d_{ffn} \\approx \\frac{8}{3} d_{\\text{model}}$.
- FFN accounts for approximately $66\%$ of total model parameters and compute FLOPs.
- **Activation Memory Footprint:** Requires holding intermediate outputs of both $W_{\\text{gate}}$ and $W_{\\text{up}}$ in memory during forward execution for backpropagation.

## Step 3: Cross-Component Coupling
- **Activation Checkpointing:** SwiGLU increases activation storage by 50% over standard ReLU/GELU MLPs, necessitating selective recomputation during training.
- **GEMM Fusion:** Concatenating $W_{\\text{gate}}$ and $W_{\\text{up}}$ into a single combined matrix $[W_{\\text{gate}} \\mid W_{\\text{up}}] \\in \\mathbb{R}^{d \\times 2d_{ffn}}$ doubles the column dimension, improving Tensor Core GEMM efficiency.

## Step 4: The Exact Performance Formula
$$\\text{FLOPs}_{\\text{SwiGLU}} = 3 \\times (2 \\times d_{\\text{model}} \\times d_{ffn}) = 16 d_{\\text{model}}^2$$
$$\\text{Activation Memory per Token} = 2 \\times d_{ffn} \\times \\text{sizeof(dtype)}$$

## Step 5: Concrete Benchmark Walkthrough
Calculating the parameters and activation size for LLaMA-3 70B ($d_{\\text{model}} = 8192, d_{ffn} = 28672$):
- Parameter count per layer: $3 \\times (8192 \\times 28672) = 704.6\\text{ million parameters}$.
- Total FFN parameters across 80 layers: $56.3\\text{ billion}$ (80% of model weights!).
- Forward activation volume per token: $2 \\times 28672 \\times 2\\text{ bytes} = 114.7\\text{ KB}$.

## Step 6: Core Systems Takeaway
> FFNs dominate the parameter count and compute budget of modern LLMs. Combining the gate and up projections into a single fused GEMM maximizes hardware throughput and reduces kernel launch overhead.
""",

    'engineering/docs/module-2/09-fused-rmsnorm-residuals.md': """# Chapter E09: Normalization & Residual Streams: Fused RMSNorm

## Step 1: Hardware Intuition
Imagine taking a shirt out of a drawer, inspecting it, putting it back in the drawer, opening the drawer again to add a badge, and closing it again. Unfused residual normalization does exactly this: it writes the residual sum to HBM, reads it back for normalization, and writes the normalized output back to HBM. Fusing means adding the badge in your hands without ever putting the shirt away.

## Step 2: Silicon Micro-Mechanics
- **Unfused Execution Flaw:**
  1. Add residual: read $\\mathbf{x}, \\mathbf{res}$, write $\\mathbf{x}_{\\text{sum}}$ to HBM ($3 \\times$ traffic).
  2. RMSNorm: read $\\mathbf{x}_{\\text{sum}}$, compute variance, scale, write $\\mathbf{y}$ to HBM ($2 \\times$ traffic).
- **Fused Kernel Execution:**
  A single Triton/CUDA kernel performs residual addition, computes the RMS scaling factor inside registers and SRAM, scales the output, and streams the result directly into registers for the next linear projection.

## Step 3: Cross-Component Coupling
- **Memory Bandwidth Savings:** Eliminates two complete HBM round-trips per Transformer sub-layer.
- **Latency Impact:** Drops sub-layer normalization latency by $3\\times$, eliminating memory-bound pipeline bubbles.

## Step 4: The Exact Performance Formula
$$\\Delta \\text{Memory Traffic} = 4 \\times (B \\cdot S \\cdot d_{\\text{model}} \\cdot \\text{sizeof(dtype)})$$
$$\\text{Speedup} = \\frac{\\text{Unfused Memory Traffic}}{\\text{Fused Memory Traffic}} \\approx \\frac{5}{2} = 2.5\\times$$

## Step 5: Concrete Benchmark Walkthrough
Profiling a 70B layer ($d = 8192, S = 4096$) in FP16:
- Unfused HBM traffic: $5 \\times (4096 \\times 8192 \\times 2) = 335.5\\text{ MB}$.
- Fused HBM traffic: $2 \\times (4096 \\times 8192 \\times 2) = 134.2\\text{ MB}$.
- Execution time on H100 ($3.35\\text{ TB/s}$): drops from $100.1\\;\\mu\\text{s}$ to $40.0\\;\\mu\\text{s}$.

## Step 6: Core Systems Takeaway
> Never write residual additions back to HBM. Fusing residual accumulation and normalization inside on-chip registers is an essential low-hanging fruit in high-performance transformer runtimes.
""",

    'engineering/docs/module-2/10-rope-kernel-fusion.md': """# Chapter E10: Positional Embeddings on the Fly: RoPE Kernel Fusion

## Step 1: Hardware Intuition
Imagine an archer who needs to adjust their aim angle based on their distance from the target. If they consult a giant printed paper atlas of angles for every shot, flipping pages takes longer than firing the arrow. Instead, calculating the angle using mental arithmetic on the fly takes virtually zero time.

## Step 2: Silicon Micro-Mechanics
- **RoPE Mechanics:** Multiplies adjacent coordinate pairs by 2D rotation matrices $\\mathbf{R}_{\\Theta, m}$.
- **Precomputed Table Bottleneck:** Loading precomputed $\\cos$ and $\\sin$ tables from HBM consumes memory bandwidth.
- **Register-Level Fusion:** High-throughput hardware transcendental functions (`__sinf`, `__cosf`) evaluate trigonometric angles on the fly inside register ALUs during the Q/K projection epilogue or FlashAttention prologue.

## Step 3: Cross-Component Coupling
- **Long-Context Scaling:** Supporting extended context (e.g. 128K via YaRN or frequency scaling) without expanding static memory tables.
- **Attention Pipeline Integration:** Fusing RoPE into FlashAttention eliminates intermediate tensor materialization entirely.

## Step 4: The Exact Performance Formula
$$\\mathbf{R}_{\\Theta, m} \\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = \\begin{bmatrix} x_1 \\cos(m\\theta) - x_2 \\sin(m\\theta) \\\\ x_1 \\sin(m\\theta) + x_2 \\cos(m\\theta) \\end{bmatrix}$$
Memory traffic when fused on the fly: Exactly zero additional HBM bytes transferred.

## Step 5: Concrete Benchmark Walkthrough
Comparing memory traffic for RoPE across 80 layers ($S = 32,768, d = 128, H = 64$):
- Unfused table lookup traffic: $80 \\times 2 \\times (32768 \\times 64 \\times 128 \\times 2) = 85.9\\text{ GB}$ per forward pass!
- Fused on-the-fly traffic: $0\\text{ GB}$.
- Wall-clock speedup: $15\\text{ ms}$ saved per forward iteration.

## Step 6: Core Systems Takeaway
> Compute trigonometric position rotations on the fly in registers. Never store precomputed rotary position embedding tables in HBM.
""",

    # Module 3
    'engineering/docs/module-3/11-number-formats-fp8-bf16.md': """# Chapter E11: Number Formats in Silicon: FP32, FP16, BF16, and FP8

## Step 1: Hardware Intuition
Think of numbers as rulers. A long wooden ruler (FP32) measures millimeter precision across vast distances. A pocket tape measure (FP16) gives millimeter precision, but breaks if you try to measure something longer than a table (underflow/overflow). BF16 shrinks the markings slightly, but extends the tape to measure football fields. FP8 is a miniature folding ruler: compact, twice as fast to carry, but requiring careful scaling blocks to avoid snapping.

## Step 2: Silicon Micro-Mechanics
- **Bit Allocations:**
  - FP32: 1 sign + 8 exponent + 23 mantissa (Range: $10^{\\pm 38}$, precision: $10^{-7}$).
  - FP16: 1 sign + 5 exponent + 10 mantissa (Range: $6.5 \\times 10^4$). Prone to loss overflow.
  - BF16: 1 sign + 8 exponent + 7 mantissa. Matches FP32 range, half the memory.
  - FP8 E4M3: 1 sign + 4 exponent + 3 mantissa. Optimized for forward weights and activations.
  - FP8 E5M2: 1 sign + 5 exponent + 2 mantissa. Wider dynamic range, optimized for backward gradients.

## Step 3: Cross-Component Coupling
- **Tensor Core Throughput:** Modern Tensor Cores (NVIDIA Hopper/Blackwell) execute FP8 matrix multiplies at $2\\times$ the TFLOPS rate of FP16/BF16.
- **Dynamic Scaling Factors:** FP8 requires tracking per-tensor or per-channel maximum values to scale activations into the valid numerical range.

## Step 4: The Exact Performance Formula
$$\\text{Weight Memory (GB)} = \\frac{\\text{Parameter Count (B)} \\times \\text{Bits per Parameter}}{8}$$
$$\\text{Throughput Ratio} = \\frac{\\text{FP8 TFLOPS}}{\\text{BF16 TFLOPS}} = 2.0\\times$$

## Step 5: Concrete Benchmark Walkthrough
Training a 70B model:
- FP32 weights: $280\\text{ GB}$.
- BF16 weights: $140\\text{ GB}$.
- FP8 weights: $70\\text{ GB}$.
- Memory bandwidth savings: Cutting weight traffic from $140\\text{ GB}$ to $70\\text{ GB}$ immediately doubles decoding throughput in bandwidth-bound regimes.

## Step 6: Core Systems Takeaway
> Precision determines dynamic range, memory bandwidth, and Tensor Core throughput. Modern LLM training standardizes on BF16 for numerical stability and FP8 for high-throughput GEMM execution.
""",

    'engineering/docs/module-3/12-quantization-strategies-w8a8-w4a16.md': """# Chapter E12: Quantization Strategies: Weight-Only vs. Weight-Activation

## Step 1: Hardware Intuition
Imagine packing luggage for a flight.
- **Weight-Only Quantization (W4A16 / W8A16):** You vacuum-seal your heavy winter coats (weights) into tiny bags to fit more into the suitcase. Once you reach the hotel (SRAM), you open the bag and let the coat expand before wearing it. This saves luggage space (HBM bandwidth) during decode.
- **Weight-Activation Quantization (W8A8):** Both your clothes and your shoes are manufactured in lightweight mini sizes, allowing you to walk directly through security without unpacking.

## Step 2: Silicon Micro-Mechanics
- **Weight-Only (AWQ, GPTQ):** Weights are stored in INT4 or INT8 in HBM. During GEMV, weights are streamed across the memory bus at reduced size and dequantized to FP16 in registers before compute. Ideal for **memory-bound decode**.
- **Weight-Activation (SmoothQuant, FP8):** Both weights and activations are quantized to 8 bits, enabling native INT8/FP8 Tensor Core MMA instructions. Ideal for **compute-bound prefill**.
- **Activation Outliers:** 0.1% of activation channels exhibit extreme values ($100\\times$ normal magnitude); SmoothQuant mathematically migrates the outlier difficulty from activations to weights.

## Step 3: Cross-Component Coupling
- **Prefill vs. Decode Trade-offs:** W4A16 accelerates batch size 1 decode, but slows down large-batch prefill due to dequantization overhead.
- **Quantization Granularity:** Per-tensor vs. per-channel vs. block-wise (group size 128) scaling factors.

## Step 4: The Exact Performance Formula
$$W_{\\text{quant}} = \\text{round}\\left(\\frac{W}{\\text{scale}}\\right) + \\text{zero\\_point}$$
$$\\text{HBM Bandwidth Speedup} = \\frac{\\text{FP16 Bytes}}{\\text{Quantized Bytes}} = \\frac{16}{\\text{bits}}$$

## Step 5: Concrete Benchmark Walkthrough
Evaluating a 70B model under INT4 vs FP16 on a single 80GB GPU:
- FP16 model: $140\\text{ GB}$ (does not fit on one GPU; requires 2 GPUs).
- INT4 model (AWQ): $35\\text{ GB}$ (fits easily on a single 80GB GPU with room for a 40GB KV cache).
- Single-token decode latency drops from $84\\text{ ms}$ (distributed over 2 GPUs) to $22\\text{ ms}$ on 1 GPU.

## Step 6: Core Systems Takeaway
> Choose your quantization strategy based on your operational regime: Weight-Only quantization accelerates memory-bound decoding; Weight-Activation quantization accelerates compute-bound prefill.
""",

    'engineering/docs/module-3/13-kv-cache-quantization.md': """# Chapter E13: KV Cache Quantization: Compressing Context to FP8 and INT4

## Step 1: Hardware Intuition
If your office file cabinet is overflowing with historical documents, you have two choices: buy another expensive room (more GPUs), or scan every document into compressed black-and-white microfilms (quantization). As long as the microfilm text remains legible, your office stays organized and you can store 4 times as many client case files.

## Step 2: Silicon Micro-Mechanics
- **Storage Layout:** Compressing cached Key and Value vectors from 16 bits to 8 bits (FP8 E4M3 or INT8) or 4 bits.
- **Per-Channel & Per-Token Scaling:** Keys and Values have different numerical distributions. Key vectors benefit from per-head dynamic scaling; Value vectors benefit from per-token scaling.
- **Dequantization in FlashAttention:** Dequantizing INT8/FP8 KV blocks directly inside on-chip SRAM before computing attention tiles.

## Step 3: Cross-Component Coupling
- **Serving Concurrency Explosion:** Cutting KV cache size by $50\%$ (FP8) or $75\%$ (INT4) immediately doubles or quadruples the maximum concurrent batch size in serving engines.
- **Needle-in-a-Haystack Accuracy:** Poorly calibrated KV quantization degrades long-range retrieval accuracy.

## Step 4: The Exact Performance Formula
$$\\text{KV Memory Savings} = \\left(1 - \\frac{\\text{Quantized Bits}}{16}\\right) \\times 100\\%$$
For FP8: 50% memory reduction; for INT4: 75% memory reduction.

## Step 5: Concrete Benchmark Walkthrough
Serving 128 concurrent users at 8K context on LLaMA-3 70B:
- FP16 KV cache: $128 \\times 8192 \\times 0.3125\\text{ MB} = 327.6\\text{ GB}$ (requires at least 5x 80GB GPUs just for cache).
- FP8 KV cache: $163.8\\text{ GB}$ ($2\\times$ fewer GPUs required).
- INT4 KV cache: $81.9\\text{ GB}$ ($4\\times$ fewer GPUs required).

## Step 6: Core Systems Takeaway
> Quantizing the KV cache is the most cost-effective architectural lever for unlocking massive concurrent serving capacity without degrading reasoning accuracy.
""",

    # Module 4
    'engineering/docs/module-4/14-distributed-communication-primitives.md': """# Chapter E14: Distributed Communication Primitives: All-Reduce, All-Gather, and Reduce-Scatter

## Step 1: Hardware Intuition
Imagine 8 accountants spread across 8 tables. Each accountant has calculated a partial total.
- **All-Reduce:** The accountants quickly pass summaries around the circle until every accountant has the grand global total.
- **All-Gather:** Each accountant has a unique chapter of a book; they make copies and share so everyone holds the full book.
- **Reduce-Scatter:** They combine partial results, but each accountant keeps only their assigned final chapter.
The time spent talking and passing papers across the room is pure overhead where no accounting math gets done.

## Step 2: Silicon Micro-Mechanics
- **Collective Algorithms:**
  - Ring All-Reduce: Transmits $2 \\frac{N-1}{N} S$ bytes per GPU in $2(N-1)$ steps.
  - Tree All-Reduce: Optimizes for low latency at small message sizes.
- **Network Bandwidth:** Intra-node NVLink (900 GB/s) vs. Inter-node InfiniBand/RoCE (400-800 Gbps $\\approx 50-100\\text{ GB/s}$).
- **CUDA Streams & Overlap:** Executing communication on dedicated communication streams while Tensor Cores execute matrix math on default compute streams.

## Step 3: Cross-Component Coupling
- **Parallelism Strategy Selection:** Fast NVLink enables high-bandwidth Tensor Parallelism (TP); slow inter-node networks force engineers to use Pipeline (PP) or Data Parallelism (DP).

## Step 4: The Exact Performance Formula
$$T_{\\text{Ring-AllReduce}} = 2(N-1)\\alpha + 2\\left(\\frac{N-1}{N}\\right)\\frac{S}{\\text{Bus Bandwidth}}$$
where $\\alpha$ is network latency, $N$ is number of GPUs, and $S$ is tensor size in bytes.

## Step 5: Concrete Benchmark Walkthrough
Benchmarking an All-Reduce of a 70B activation tensor ($S = 4096, d = 8192, \\text{FP16} = 67.1\\text{ MB}$) across 8 GPUs:
- Over NVLink 4 ($900\\text{ GB/s}$): $T \\approx 0.13\\text{ ms}$.
- Over 100 Gbps Ethernet ($12.5\\text{ GB/s}$): $T \\approx 9.4\\text{ ms}$ ($72\\times$ slower!).

## Step 6: Core Systems Takeaway
> Collective communication overhead dictates parallel scaling efficiency. Match the communication volume of your parallelism strategy to the physical bandwidth of the underlying network interconnect.
""",

    'engineering/docs/module-4/15-tensor-parallelism-megatron.md': """# Chapter E15: Tensor Parallelism (TP): Megatron-LM in Silicon

## Step 1: Hardware Intuition
Imagine a mural painting so wide that no single painter can reach both ends. Instead of having painters paint in shifts, you divide the canvas vertically. Painter 1 paints the left half, Painter 2 paints the right half. When finished, they step back and coordinate only at the boundary lines.

## Step 2: Silicon Micro-Mechanics
- **Megatron-LM Sharding Strategy (Shoeybi et al.):**
  - Self-Attention: Split $W_Q, W_K, W_V$ column-wise (no communication needed); split $W_O$ row-wise (requires 1 All-Reduce).
  - FFN: Split $W_{\\text{gate}}, W_{\\text{up}}$ column-wise; split $W_{\\text{down}}$ row-wise (requires 1 All-Reduce).
  - Total Communication: Exactly 2 All-Reduces per Transformer layer.
- **Sequence Parallelism (SP):** Shards LayerNorm and Dropout along the sequence dimension to eliminate redundant compute and activation memory.

## Step 3: Cross-Component Coupling
- **Hardware Boundary:** TP degree is strictly bounded by single-node NVLink limits (typically $\\text{TP} \\le 8$). Running TP across nodes causes massive interconnect latency stalls.

## Step 4: The Exact Performance Formula
$$\\text{Communication Volume per Layer} = 4 \\times B \\times S \\times d_{\\text{model}} \\times \\text{sizeof(dtype)}$$
$$\\text{Layer Speedup} = \\frac{T_{\\text{single}}}{T_{\\text{TP}}} = \\frac{\\text{FLOPs} / \\text{TP} + T_{\\text{AllReduce}}}{\\text{FLOPs}}$$

## Step 5: Concrete Benchmark Walkthrough
Tracing Megatron TP=8 for LLaMA-3 70B ($d = 8192, S = 4096$) in FP16:
- Parameter footprint per GPU: $140\\text{ GB} / 8 = 17.5\\text{ GB}$.
- All-Reduce volume per layer: $2 \\times (4096 \\times 8192 \\times 2) = 134.2\\text{ MB}$.
- Total communication per forward pass across 80 layers: $10.7\\text{ GB}$.

## Step 6: Core Systems Takeaway
> Tensor Parallelism distributes weight matrices with exactly two All-Reduce communications per layer, but requires ultra-fast NVLink interconnects to prevent communication stalls.
""",

    'engineering/docs/module-4/16-pipeline-parallelism-1f1b.md': """# Chapter E16: Pipeline Parallelism (PP): 1F1B Scheduling & Bubble Analysis

## Step 1: Hardware Intuition
Imagine an automotive assembly line with 4 stations: Chassis, Engine, Body, and Paint. If Worker 1 builds the entire car before passing it to Worker 2, three workers stand idle at any moment. By breaking the car into 20 small micro-parts flowing continuously through the stations, all 4 workers stay busy simultaneously, with only tiny startup and shutdown delays.

## Step 2: Silicon Micro-Mechanics
- **Partitioning:** Splitting 80 layers across $p$ pipeline stages (e.g. 10 layers per GPU across 8 GPUs).
- **The Pipeline Bubble:** The idle time during pipeline ramp-up and ramp-down:
  $$F_{\\text{bubble}} = \\frac{p - 1}{m + p - 1}$$
  where $m$ is the number of microbatches.
- **1F1B Scheduling:** Once warm, each stage alternates one forward pass with one backward pass, bounding activation memory to at most $p$ microbatches.

## Step 3: Cross-Component Coupling
- **Activation Stashing:** Stage 0 must retain activations for $p$ microbatches, increasing memory pressure on early pipeline stages.
- **Interleaved 1F1B:** Assigning multiple non-contiguous virtual stages per GPU to shrink the bubble factor by $2\\times$.

## Step 4: The Exact Performance Formula
$$\\text{Bubble Overhead Percentage} = \\frac{p - 1}{m} \\times 100\\% \\quad (\\text{for } m \\gg p)$$

## Step 5: Concrete Benchmark Walkthrough
Pipeline parallelism on 8 stages ($p = 8$) with 32 microbatches ($m = 32$):
- Bubble fraction: $\\frac{8 - 1}{32 + 8 - 1} = \\frac{7}{39} \\approx 17.9\\%$.
- With interleaved 1F1B (virtual stages $v = 2$): bubble fraction drops to $9.8\\%$.

## Step 6: Core Systems Takeaway
> Pipeline Parallelism allows scaling across slow inter-node networks, but requires high microbatch counts to keep the idle bubble fraction below 10%.
""",

    'engineering/docs/module-4/17-zero-fsdp-data-parallelism.md': """# Chapter E17: Data Parallelism & Zero Redundancy: ZeRO-1, 2, 3 and FSDP

## Step 1: Hardware Intuition
Imagine a team of 64 researchers analyzing survey responses. Standard data parallelism requires every researcher to purchase and carry the entire 20-volume encyclopedia set in their backpack. ZeRO-3 assigns each researcher only 1 volume. When Researcher 5 needs to look up a word from Volume 2, they ask Researcher 2 over the intercom, read the definition, and hand the page back.

## Step 2: Silicon Micro-Mechanics
- **Memory Footprint of Training:**
  - Parameters: $2P$ bytes (FP16).
  - Gradients: $2P$ bytes (FP16).
  - Adam Optimizer: $4P$ (master weights) $+ 4P$ (momentum) $+ 4P$ (variance) $= 12P$ bytes.
  - Total: $16P$ bytes!
- **ZeRO Stages (Rajbhandari et al.):**
  - ZeRO-1: Shard optimizer states ($4\\times$ memory reduction).
  - ZeRO-2: Shard optimizer states and gradients ($8\\times$ memory reduction).
  - ZeRO-3 (FSDP): Shard optimizer, gradients, and model parameters. Fetches parameters on the fly via All-Gather and discards them immediately after forward/backward compute.

## Step 3: Cross-Component Coupling
- **Communication Trade-off:** ZeRO-3 increases total communication volume by 50% (adds 1 All-Gather per forward layer), requiring communication-compute overlap to hide latency.

## Step 4: The Exact Performance Formula
$$M_{\\text{ZeRO-3}} = \\frac{2P + 2P + 12P}{N_{\\text{GPUs}}} + M_{\\text{activations}} = \\frac{16P}{N} + M_{\\text{activations}}$$

## Step 5: Concrete Benchmark Walkthrough
Training a 70B parameter model across 64 GPUs:
- Standard DDP memory per GPU: $16 \\times 70\\text{ GB} = 1120\\text{ GB}$ (impossible on any single GPU).
- ZeRO-3 memory per GPU: $\\frac{1120\\text{ GB}}{64} \\approx 17.5\\text{ GB}$ (comfortably fits on an 80GB A100/H100!).

## Step 6: Core Systems Takeaway
> ZeRO-3 / FSDP completely democratizes massive LLM training by sharding the 16P memory burden across all cluster GPUs, trading 50% extra communication for boundless memory scalability.
""",

    'engineering/docs/module-4/18-ring-attention-context-parallelism.md': """# Chapter E18: Context & Sequence Parallelism: Ring Attention & 1M+ Context

## Step 1: Hardware Intuition
Imagine 8 students sitting in a circle tasked with proofreading a 1-million-word book. Instead of copying the entire book for everyone, Student 1 takes Chapter 1, Student 2 takes Chapter 2, etc. As they read, they pass their chapter notes clockwise around the circle. By the time notes make one full lap around the ring, every student has cross-referenced their section with the entire book.

## Step 2: Silicon Micro-Mechanics
- **The Long-Context Activation Barrier:** At 1M tokens, activation memory exceeds hundreds of gigabytes, exceeding single-GPU HBM.
- **Ring Attention (Liu et al.):**
  - Shards sequence $S$ into $N$ blocks of size $S/N$.
  - Each GPU computes local attention between its local Query block and local Key/Value block.
  - Concurrently transmits Key/Value blocks to the next rank in a ring topology using asynchronous P2P communication.

## Step 3: Cross-Component Coupling
- **Zero Communication Overhead (Latency Hiding):** If local tile compute time $\\ge$ transfer time of the KV block, the network communication is 100% hidden behind compute!

## Step 4: The Exact Performance Formula
$$\\text{Overlap Condition: } T_{\\text{compute}}\\left(\\frac{S}{N} \\times \\frac{S}{N}\\right) \\ge T_{\\text{transfer}}\\left(\\frac{S}{N} \\text{ KV block}\\right)$$

## Step 5: Concrete Benchmark Walkthrough
Scaling a 70B model to 1-million token context on 64 GPUs:
- Local sequence block: $1,000,000 / 64 = 15,625$ tokens.
- KV block size per layer: $15,625 \\times 8 \\times 128 \\times 2 \\text{ bytes} \\approx 32\\text{ MB}$.
- At $50\\text{ GB/s}$ inter-node network bandwidth, transfer takes $0.64\\text{ ms}$.
- Local tile compute takes $1.2\\text{ ms}$. Compute exceeds transfer ($1.2 > 0.64$), achieving **100% communication hiding**!

## Step 6: Core Systems Takeaway
> Ring Attention enables infinitely long context horizons by overlapping P2P ring communication with attention tile computation, transforming quadratic sequence memory barriers into linear ring scaling.
""",

    # Module 5
    'engineering/docs/module-5/19-prefill-vs-decode-regimes.md': """# Chapter E19: The Two Inference Regimes: Prefill vs. Decode

## Step 1: Hardware Intuition
Imagine a post office sorting mail:
- **Prefill (Prompt Ingestion):** A delivery truck unloads 4,000 letters all at once. The sorting machine runs at full power, batching and scanning hundreds of envelopes per second.
- **Decode (Generation):** A customer stands at the counter asking one question every 30 seconds, forcing the clerk to walk back to the storage vault each time to retrieve one single document.

## Step 2: Silicon Micro-Mechanics
- **Prefill Phase:** High operational intensity ($I > 100$), highly compute-bound, saturates Tensor Cores. Primary metric: **Time to First Token (TTFT)**.
- **Decode Phase:** Low operational intensity ($I \\approx 1-5$), severely memory-bandwidth bound. Primary metric: **Inter-Token Latency (ITL)**.
- **Prefill-Decode Interference:** Co-scheduling a heavy prefill job inside an ongoing decode batch causes latency spikes for interactive streaming users.

## Step 3: Cross-Component Coupling
- **Architecture Disaggregation:** Splitwise & Mooncake systems physically separate prefill worker nodes from decode worker nodes to isolate interference.

## Step 4: The Exact Performance Formula
$$\\text{TTFT} \\approx \\frac{2 \\times P \\times S_{\\text{prompt}}}{\\text{Attainable FLOPS}}$$
$$\\text{ITL} \\approx \\frac{\\text{Model Weights} + \\text{KV Cache}}{\\text{HBM Bandwidth}}$$

## Step 5: Concrete Benchmark Walkthrough
Comparing prefill vs. decode execution on LLaMA-3 8B ($16\\text{ GB}$ FP16 weights) on an H100 ($3.35\\text{ TB/s}$, 1000 TFLOPS):
- Prefill (2048 prompt tokens): Compute takes $\\approx 0.065\\text{ ms}$; bandwidth takes $0.005\\text{ ms}$. Compute-bound!
- Decode (1 token, batch 1): Compute takes $0.00003\\text{ ms}$; memory bandwidth takes $4.77\\text{ ms}$. 99.9% memory-bound!

## Step 6: Core Systems Takeaway
> An LLM engine is two completely different beasts: a compute-bound GEMM cruncher during prefill, and a memory-bound bandwidth streamer during decode. Production serving architectures must optimize both regimes independently.
""",

    'engineering/docs/module-5/20-continuous-batching-scheduling.md': """# Chapter E20: Continuous Batching & Dynamic Scheduling

## Step 1: Hardware Intuition
Imagine a ski chairlift.
- **Static Batching:** The chairlift waits until 4 people arrive, rides up the mountain, and refuses to let anyone new get on or off until all 4 skiers finish their entire afternoon of skiing.
- **Continuous Batching:** Skiers get on empty chairs as soon as one arrives; when a skier finishes, their seat is immediately filled by the next skier waiting in line.

## Step 2: Silicon Micro-Mechanics
- **Static Batching Flaw:** Padding shorter sequences with zeroes until the longest sequence in the batch completes wastes up to 70% of GPU compute on useless padding tokens.
- **Continuous / In-Flight Batching (Orca, vLLM):** Operates at the iteration step level. Requests are added and retired dynamically at every forward pass.
- **Ragged Tensors:** Packing non-uniform sequence lengths into a single 1D vector using offset pointers to eliminate all padding.

## Step 3: Cross-Component Coupling
- **Integration with PagedAttention:** Continuous batching requires dynamic page allocation to append KV blocks on a per-step basis without memory defragmentation stalls.

## Step 4: The Exact Performance Formula
$$\\text{Throughput} = \\frac{\\sum_{i=1}^N \\text{Tokens Generated}_i}{T_{\\text{wall\\_clock}}}$$
$$\\text{Compute Efficiency Gain} = \\frac{\\text{Useful Tokens}}{\\text{Useful Tokens} + \\text{Padding Tokens}}$$

## Step 5: Concrete Benchmark Walkthrough
Benchmarking 64 requests with lengths uniformly distributed from 50 to 1000 tokens:
- Static batching: Padded to $1000$ tokens per request ($64,000$ tokens processed, of which $33,600$ are useless padding $\\to 52.5\\%$ waste).
- Continuous batching: Exactly $30,400$ tokens processed ($0\\%$ padding waste, $2.1\\times$ throughput speedup).

## Step 6: Core Systems Takeaway
> Never pad tokens in production. Iteration-level continuous batching eliminates tensor bubbles and doubles effective serving throughput.
""",

    'engineering/docs/module-5/21-speculative-decoding.md': """# Chapter E21: Speculative Decoding: Draft-Verify Silicon Acceleration

## Step 1: Hardware Intuition
Imagine a slow professor and a fast student. The student rapidly drafts 5 potential sentences for an essay. The professor glances at all 5 sentences simultaneously in 1 second, approves the first 4, corrects the 5th, and moves on. The essay progresses $4\\times$ faster than if the professor had penned every word from scratch.

## Step 2: Silicon Micro-Mechanics
- **Memory-Bound Bottleneck Exploitation:** Generating 1 token from a 70B model requires reading 140 GB of weights. Evaluating 5 tokens simultaneously takes virtually the same time because prefill evaluation is compute-bound.
- **The Speculative Loop (Leviathan et al.):**
  1. Draft model (e.g. 1B) autoregressively drafts $K$ candidate tokens.
  2. Target model (e.g. 70B) validates all $K$ candidates in a **single parallel forward pass**.
  3. Acceptance via modified rejection sampling guarantees exact mathematical distribution equivalence.

## Step 3: Cross-Component Coupling
- **Draft Model Selection:** Optimal draft model size is typically 5-10% of the target model's parameter count.
- **Speculative Trees (Medusa / EAGLE):** Using multiple decoding heads to branch candidate paths, boosting acceptance rates.

## Step 4: The Exact Performance Formula
$$\\text{Expected Speedup} = \\frac{1 + \\alpha + \\alpha^2 + \\dots + \\alpha^K}{1 + \\frac{c_{\\text{draft}}}{c_{\\text{target}}} \\cdot K}$$
where $\\alpha$ is the average token acceptance rate.

## Step 5: Concrete Benchmark Walkthrough
Pairing LLaMA-3 70B with LLaMA-3 8B at $K = 5$ speculative tokens, assuming acceptance rate $\\alpha = 0.75$:
- Expected accepted tokens per step: $1 + 0.75 + 0.56 + 0.42 + 0.31 = 3.04$ tokens.
- Draft cost ratio: $\\approx 0.12$.
- Realized wall-clock speedup: $2.1\\times$ faster generation with 100% mathematical output fidelity!

## Step 6: Core Systems Takeaway
> Speculative decoding trades cheap idle compute during the memory-bound decode phase to generate multiple tokens per HBM weight sweep, delivering pure wall-clock acceleration without changing output quality.
""",

    'engineering/docs/module-5/22-chunked-prefill-prefix-caching.md': """# Chapter E22: Chunked Prefill & Prefix Caching

## Step 1: Hardware Intuition
Imagine an amusement park ride where a massive group of 500 tourists arrives. If the ride operator stops all regular visitors to load the entire tourist group, regular visitors wait for an hour. Instead, the operator lets 50 tourists board alongside 50 regular visitors every 5 minutes, keeping lines moving smoothly.

## Step 2: Silicon Micro-Mechanics
- **Chunked Prefill:** Slices large prompt prefills into uniform chunks (e.g. 512 tokens) and co-schedules them alongside decode requests.
- **Interference Mitigation:** Bounds Inter-Token Latency (ITL) spikes for existing streaming users.
- **Automatic Prefix Caching (APC):** Organizes KV pages in a Radix Tree. Common prompt prefixes (system instructions, multi-turn chat history) reuse cached KV blocks via pointer lookups.

## Step 3: Cross-Component Coupling
- **TTFT Elimination:** High prefix hit rates ($h > 80\\%$) convert $O(S)$ prefill compute into an $O(1)$ pointer copy, dropping TTFT from seconds to milliseconds.

## Step 4: The Exact Performance Formula
$$\\text{TTFT}_{\\text{cached}} = (1 - h) \\cdot \\text{TTFT}_{\\text{full}} + \\epsilon_{\\text{lookup}}$$
where $h$ is the prefix cache hit rate.

## Step 5: Concrete Benchmark Walkthrough
Benchmarking a multi-turn customer support agent with a 2,000-token system prompt:
- Without prefix caching: Every turn recomputes 2,000 tokens ($\\\\text{TTFT} \\approx 120\\text{ ms}$).
- With prefix caching: Reuses cached KV pages ($h = 100\\%$ for prefix; $\\text{TTFT} drops to $4\\text{ ms}$).

## Step 6: Core Systems Takeaway
> Chunked prefill tames decode latency spikes, while prefix caching eliminates redundant prompt compute. Together, they form the backbone of predictable multi-tenant LLM serving.
""",

    # Module 6
    'engineering/docs/module-6/23-moe-routing-capacity-factor.md': """# Chapter E23: MoE Architecture in Systems: Top-K Gating & Capacity Factor

## Step 1: Hardware Intuition
Imagine a hospital with 64 specialized doctors. When a patient arrives, a triage nurse directs them to the 2 most relevant specialists. You don't need all 64 doctors examining every patient; only 2 doctors work per case. However, if all patients rush to the cardiologist simultaneously, the cardiologist's waiting room overflows while the other 63 doctors sit idle.

## Step 2: Silicon Micro-Mechanics
- **MoE Gating Mechanics:** Gating network scores experts, selects top-$K$ experts per token:
  $$\\mathbf{y} = \\sum_{i \\in \\text{TopK}(G(\\mathbf{x}))} g_i(\\mathbf{x}) \\text{Expert}_i(\\mathbf{x})$$
- **Parameter Decoupling:** Total parameters vs. Active parameters (e.g. DeepSeek-V3: 671B total, 37B active).
- **The Capacity Factor:** Enforcing a maximum token buffer per expert to prevent static GPU allocation blowups; overflow tokens are dropped or routed to residual paths.

## Step 3: Cross-Component Coupling
- **Load Balancing Losses:** Auxiliary loss terms penalize expert load imbalance, preventing expert collapse where all tokens route to the same 2 experts.

## Step 4: The Exact Performance Formula
$$\\text{Expert Capacity} = \\text{Capacity Factor} \\times \\left(\\frac{\\text{Total Tokens}}{E}\\right) \\times K$$
$$\\text{FLOPs}_{\\text{MoE}} \\approx \\frac{K}{E} \\times \\text{FLOPs}_{\\text{Dense}(P_{\\text{total}})}$$

## Step 5: Concrete Benchmark Walkthrough
Profiling Mixtral 8x7B (8 experts, Top-2 routing):
- Total parameters: $46.7\\text{ billion}$.
- Active parameters per token: $12.9\\text{ billion}$.
- Compute FLOPs match a 13B model, while knowledge capacity rivals a 45B dense model.

## Step 6: Core Systems Takeaway
> Mixture of Experts decouples parameter capacity from compute cost, delivering massive parameter scaling at a fraction of dense FLOP budgets.
""",

    'engineering/docs/module-6/24-all-to-all-distributed-moe.md': """# Chapter E24: All-to-All Dispatch & Combine: Distributed MoE Bottlenecks

## Step 1: Hardware Intuition
Imagine 8 sorting hubs in 8 different cities. Each city receives mail meant for all 8 cities. Every hour, all 8 hubs must simultaneously exchange mail trucks in an all-to-all highway gridlock. If the highways are narrow, trucks sit in traffic jams and the sorting centers grind to a halt.

## Step 2: Silicon Micro-Mechanics
- **Expert Parallelism (EP):** Sharding $E$ experts across $N$ GPUs (e.g. 8 experts per GPU across 8 nodes).
- **The Two Communication Phases:**
  1. **All-to-All Dispatch:** Routes token activations from their origin GPU to the GPU hosting the assigned expert.
  2. **All-to-All Combine:** Returns processed expert outputs back to the original token sequence positions.
- **The Network Wall:** All-to-All transfers stress crossbar switches and inter-node network links.

## Step 3: Cross-Component Coupling
- **Overlap Innovations (DeepSeek DualPipe):** Bi-directional pipelining that overlaps All-to-All communication with FFN computation of adjacent microbatches.

## Step 4: The Exact Performance Formula
$$\\text{All-to-All Volume per Layer} = 2 \\times K \\times B \\times S \\times d_{\\text{model}} \\times \\text{sizeof(dtype)}$$

## Step 5: Concrete Benchmark Walkthrough
Profiling All-to-All dispatch for 256 tokens per GPU across 64 GPUs ($d = 7168$, FP8 = 1 byte):
- Communication volume per GPU: $2 \\times 8 \\times 256 \\times 7168 \\times 1 \\text{ byte} \\approx 29.4\\text{ MB}$ per layer.
- Across 60 layers: $1.76\\text{ GB}$ transferred per step.
- Over 400 Gbps InfiniBand: takes $35.2\\text{ ms}$, accounting for up to 40% of step time without communication overlap.

## Step 6: Core Systems Takeaway
> Distributed MoE efficiency is governed by All-to-All network crossbar bandwidth. Overlapping dispatch communication with expert computation is essential to unlocking sparse scaling.
""",

    # Module 7
    'engineering/docs/module-7/25-model-memory-budget.md': """# Chapter E25: The Complete Model Memory Budget Equation

## Step 1: Hardware Intuition
Imagine packing an expedition backpack with strict weight limits. You have:
1. Essential base gear that never changes (Model Weights).
2. Survival rations for the climb (Optimizer States & Gradients).
3. Temporary trail notes you write and erase along the path (Activations).
If your backpack exceeds 80 kilograms, you cannot walk. Knowing the exact weight of every carabiner is how you reach the summit without collapsing.

## Step 2: Silicon Micro-Mechanics
- **Training Memory Equation:**
  $$M_{\\text{total}} = M_{\\text{weights}} + M_{\\text{gradients}} + M_{\\text{optimizer}} + M_{\\text{activations}} + M_{\\text{temp}}$$
  - Weights: $2P$ bytes (BF16).
  - Gradients: $2P$ bytes (BF16).
  - Adam Optimizer: $12P$ bytes ($4P$ master FP32 weights, $4P$ momentum, $4P$ variance).
  - Base State Total: $16P$ bytes!
- **Inference Memory Equation:**
  $$M_{\\text{inference}} = M_{\\text{weights}} + M_{\\text{KV\\_Cache}} + M_{\\text{scratchpad}}$$

## Step 3: Cross-Component Coupling
- **Activation Checkpointing (Recomputation):** Storing activations for every layer exhausts HBM. Selective recomputation discards non-GEMM activations, reducing activation memory by $5\\times$ for only a $30\\%$ compute overhead.

## Step 4: The Exact Performance Formula
$$\\text{Activation Memory per Transformer Layer (Standard)} = B \\cdot S \\cdot d_{\\text{model}} \\cdot \\left(34 + 5 \\frac{a \\cdot S}{d_{\\text{model}}}\\right) \\text{ bytes}$$

## Step 5: Concrete Benchmark Walkthrough
Auditing memory for LLaMA-3 70B training on 8x H100 (80GB each = 640GB total):
- Model weights: $140\\text{ GB}$.
- Gradients: $140\\text{ GB}$.
- Adam states: $840\\text{ GB}$.
- Total base state: $1120\\text{ GB}$.
- Conclusion: Does not fit on 8 GPUs without ZeRO-3 / FSDP sharding!

## Step 6: Core Systems Takeaway
> Training memory is dominated by optimizer states (12P), while inference memory is dominated by the KV cache. Master the memory budget equation to prevent Out-Of-Memory crashes before allocating a single tensor.
""",

    'engineering/docs/module-7/26-measuring-mfu-hfu-profiling.md': """# Chapter E26: Measuring True Performance: MFU, HFU, and Profiling

## Step 1: Hardware Intuition
A sports car speedometer might say "Top Speed: 300 km/h". But in city traffic with red lights, sharp turns, and speed bumps, your average speed is only 45 km/h. Model FLOPs Utilization (MFU) is the true speedometer of AI engineering: it measures what percentage of the GPU's theoretical top speed was converted into real token learning.

## Step 2: Silicon Micro-Mechanics
- **Model FLOPs Utilization (MFU):**
  $$\\text{MFU} = \\frac{\\text{Tokens/Sec} \\times 6P}{\\text{Total Hardware Peak FLOPS}}$$
- **Hardware FLOPs Utilization (HFU):** Includes recomputation and backward overhead:
  $$\\text{HFU} = \\frac{\\text{Actual Executed FLOPs/Sec}}{\\text{Peak FLOPS}}$$
- **Profiling Tools:**
  - `nsys` (NVIDIA Nsight Systems): System-level timeline, CUDA kernel launches, NVLink collective stalls.
  - `ncu` (NVIDIA Nsight Compute): Kernel-level micro-architecture profiling, memory warp stalls, SM occupancy.

## Step 3: Cross-Component Coupling
- **Efficiency Thresholds:**
  - Naive implementations: 15-25% MFU.
  - Good implementations: 40-50% MFU.
  - World-class implementations (Megatron / LLaMA-3): 55-65% MFU.

## Step 4: The Exact Performance Formula
$$\\text{FLOPs per Token (Training)} \\approx 6P$$
$$\\text{FLOPs per Token (Inference)} \\approx 2P$$

## Step 5: Concrete Benchmark Walkthrough
Calculating MFU for a cluster of 512x H100 GPUs ($989\\text{ TFLOPS}$ each) training a 70B model at 2,500 tokens/sec:
- Cluster peak: $512 \\times 989 \\times 10^{12} = 5.06 \\times 10^{17} \\text{ FLOPS}$.
- Useful FLOPs: $2500 \\times (6 \\times 70 \\times 10^9) = 1.05 \\times 10^{15} \\text{ FLOPs/s}$.
- $\\text{MFU} = \\frac{1.05 \\times 10^{15}}{5.06 \\times 10^{17}} \\approx 52.1\\%$ (Solid, production-grade efficiency!).

## Step 6: Core Systems Takeaway
> Datasheet TFLOPS are marketing; Model FLOPs Utilization (MFU) is engineering truth. Profile before optimizing, and measure your systems code against the physical limits of the machine.
""",

    # Labs
    'engineering/docs/labs/01-roofline-profiler.md': """# Lab E1: The Roofline Profiler

## Objective
Build a lightweight Python and PyTorch profiler that computes operational intensity ($I = \\text{FLOPs}/\\text{Byte}$) for arbitrary neural network layers and plots their execution against hardware GPU rooflines.

## Learning Milestones
1. Use PyTorch profiler hooks and hardware specifications to extract kernel runtime and memory traffic.
2. Calculate arithmetic intensity for RMSNorm, Softmax, and GEMM operations.
3. Automatically classify whether a layer is memory-bandwidth bound or compute-bound.
4. Export interactive Roofline charts illustrating the hardware inflection point $I^*$.

## Key Code Components
- Hardware specification database (H100, A100, RTX 4090).
- Operational intensity calculator.
- Roofline visualization generator.
""",

    'engineering/docs/labs/02-flash-attention-kernel.md': """# Lab E2: The Minimal FlashAttention Kernel

## Objective
Implement a functional tiled multi-head attention kernel with online softmax in Triton or PyTorch C++, keeping intermediate attention matrices strictly within on-chip SRAM.

## Learning Milestones
1. Partition Query, Key, and Value tensors into SRAM-sized blocks.
2. Implement the online softmax normalizer algorithm ($m_i, d_i$).
3. Eliminate materialization of the $S \\times S$ attention matrix in HBM.
4. Benchmark speedup and memory savings against naive PyTorch attention across sequence lengths from 512 to 16,384.

## Key Code Components
- SRAM block tiling loops.
- Online softmax accumulation logic.
- Triton kernel launch configuration.
""",

    'engineering/docs/labs/03-paged-kv-cache.md': """# Lab E3: The Paged KV Cache Engine

## Objective
Construct a zero-dependency Python and NumPy prototype of virtual memory page-table KV cache management (PagedAttention) for high-concurrency autoregressive decoding.

## Learning Milestones
1. Implement a physical memory block allocator with fixed block size (16 tokens).
2. Maintain per-sequence page tables mapping logical token indices to physical block IDs.
3. Handle dynamic token generation and copy-on-write memory sharing for branching prompts.
4. Measure memory fragmentation reduction compared to static buffer allocation.

## Key Code Components
- Physical block manager (`BlockAllocator`).
- Page table directory (`LogicalToPhysicalMap`).
- Dynamic token append and eviction simulation.
""",

    'engineering/docs/labs/04-tensor-parallel-engine.md': """# Lab E4: 2-GPU Tensor Parallelism Engine

## Objective
Implement distributed Megatron-style column-parallel and row-parallel linear layers from scratch using PyTorch `torch.distributed` collective communication primitives.

## Learning Milestones
1. Initialize a 2-rank distributed process group.
2. Construct a `ColumnParallelLinear` layer that partitions weights along output channels.
3. Construct a `RowParallelLinear` layer with manual `all_reduce` collective summation.
4. Assemble a complete 2-GPU parallel MLP and verify numerical parity with a standard single-device layer.

## Key Code Components
- Distributed rank initialization.
- ColumnParallelLinear and RowParallelLinear modules.
- Verification script comparing single-GPU vs. 2-GPU numerical outputs.
"""
}

for path, content in modules_data.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content.strip() + '\n')
print(f'Successfully scaffolded {len(modules_data)} engineering files.')
