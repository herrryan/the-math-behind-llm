# Chapter 20: The Dual-Phase Engine (Prefill vs. Decode & The Roofline Model)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you run a toy delivery service with a giant yellow truck and a single bicycle courier.
> 
> When a school orders 1,000 blocks at once, you load all 1,000 blocks into the giant truck in one go. The truck's engine roars at full power, the wheels grip the road, and with one single trip down the highway, all 1,000 blocks arrive. Every cubic inch of the truck was packed tight. That is **Prefill**.
> 
> But then, the teacher asks for one more block every hour: *"Give me a red block... now wait an hour... now give me a blue block..."*
> 
> To bring that one tiny block, you can't just throw it through the air. You must start the giant truck, drive the entire 50-ton vehicle all the way down the highway, unload one single marble, drive all the way back, and do it again. The truck's giant engine is barely doing any work carrying that marble; 99% of your time and fuel is wasted just driving the heavy truck back and forth. That is **Decode**.
> 
> An LLM does not run one single kind of task. It has two completely different operational personalities: a freight train that processes your entire prompt in parallel, and a lone delivery bicycle that creeps along one word at a time.

---

## Step 2: The Bridging Question

How do we convert this physical difference between hauling 1,000 blocks at once versus driving back and forth for a single marble into exact numbers that computer hardware measures?

In computing hardware, we have two physical components that dictate speed:
1. **The Math Engine (Compute Cores / Tensor Cores)**: How many arithmetic additions and multiplications the chip can execute in one second (measured in <abbr title="Floating Point Operations Per Second">FLOPs</abbr>/sec).
2. **The Memory Highway (High Bandwidth Memory - HBM / DRAM)**: How many gigabytes of numbers the chip can move from storage into the math engine in one second (measured in Bytes/sec).

The central bridging question is:
$$\text{How many math operations do we perform for every byte of data we fetch from the memory warehouse?}$$

If we do hundreds of math operations for every byte fetched, our math engine runs at maximum speed. If we do only one single math operation for every byte fetched, our math engine sits idle waiting for memory to arrive.

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE GPU ROOFLINE MODEL &amp; ARITHMETIC INTENSITY                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Performance (TFLOPS)                                                                   │
│       ▲                                                                                │
│ P_peak│───────────────────────────────┐ &lt;── Flat Roof: Compute-Bound Region             │
│       │                              /      (Prefill / Matrix-Matrix GEMM)             │
│       │                             /                                                  │
│       │                            /                                                   │
│       │                           / &lt;────── Slanted Roof: Memory-Bandwidth-Bound Region│
│       │                          /          (Decode / Matrix-Vector GEMV)              │
│       │                         /                                                      │
│       │                        /                                                       │
│      0└───────────────────────┴──────────────────────────────────────►                 │
│       0                       I* (Ridge Point)         Operational Intensity (FLOPs/B) │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 20.1:</strong> The Roofline Model. Left of the critical ridge point I*, execution speed is throttled strictly by memory bandwidth. Right of I*, execution speed reaches the hardware peak compute ceiling.</figcaption>
</figure>

### 1. Operational Intensity (Arithmetic Intensity)

Let $\text{FLOPs}$ denote the total number of floating-point operations executed in a computational step, and let $\text{Bytes}$ denote the total volume of data transferred between main GPU memory (<abbr title="High Bandwidth Memory">HBM</abbr>) and on-chip SRAM registers.

<dfn id="def-intensity">Operational Intensity</dfn> $I$ is defined as:

$$
I = \frac{\text{FLOPs}}{\text{Bytes Transfers}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
$$

### 2. The Roofline Model Formulation

Let:
- $P_{\text{peak}}$: The theoretical peak compute throughput of the accelerator (e.g., in $\text{TFLOPS} = 10^{12} \text{ FLOP/s}$).
- $B_{\text{peak}}$: The peak memory bandwidth of the memory bus (e.g., in $\text{TB/s} = 10^{12} \text{ Byte/s}$).

The maximum attainable performance $P$ of any algorithm is governed by the upper envelope of compute and memory constraints:

$$
P(I) = \min\left(P_{\text{peak}}, \; I \times B_{\text{peak}}\right)
$$

### 3. The Critical Ridge Point ($I^*$)

The machine's structural turning point occurs where the memory bandwidth limit intersects the peak compute ceiling:

$$
I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}
$$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 20.1:</strong> Hardware parameters and critical ridge point $I^*$ across modern AI accelerators (16-bit precision).</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Accelerator</th>
      <th align="right">Peak Tensor Compute ($P_{\text{peak}}$)</th>
      <th align="right">Peak HBM Bandwidth ($B_{\text{peak}}$)</th>
      <th align="right">Critical Ridge Point ($I^*$)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>NVIDIA A100 (SXM4 80GB)</strong></td>
      <td align="right">312 TFLOPS (BF16)</td>
      <td align="right">2.039 TB/s</td>
      <td align="right"><strong>153.0 FLOPs/Byte</strong></td>
    </tr>
    <tr>
      <td><strong>NVIDIA H100 (SXM5 80GB)</strong></td>
      <td align="right">989 TFLOPS (BF16)</td>
      <td align="right">3.350 TB/s</td>
      <td align="right"><strong>295.2 FLOPs/Byte</strong></td>
    </tr>
    <tr>
      <td><strong>NVIDIA B200 (SXM 192GB)</strong></td>
      <td align="right">2,250 TFLOPS (BF16)</td>
      <td align="right">8.000 TB/s</td>
      <td align="right"><strong>281.3 FLOPs/Byte</strong></td>
    </tr>
  </tbody>
</table>

### 4. Phase 1: The Prefill Phase (GEMM)

In the <dfn id="def-prefill">Prefill Phase</dfn> (also called the prompt processing phase), the user provides an input prompt of length $T_{\text{prompt}}$. The model processes all $T_{\text{prompt}}$ tokens simultaneously.

For a linear weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and an input activation matrix $\mathbf{X} \in \mathbb{R}^{T_{\text{prompt}} \times d_{\text{in}}}$:
- The projection is a General Matrix-Matrix Multiplication (<abbr title="General Matrix-Matrix Multiplication">GEMM</abbr>):
  $$\mathbf{Y} = \mathbf{X} \mathbf{W} \in \mathbb{R}^{T_{\text{prompt}} \times d_{\text{out}}}$$
- **Computation**: Each output element requires $d_{\text{in}}$ multiply-accumulate operations ($2 d_{\text{in}}$ FLOPs). Total compute is:
  $$\text{FLOPs}_{\text{prefill}} = 2 \times T_{\text{prompt}} \times d_{\text{in}} \times d_{\text{out}}$$
- **Memory Transferred**: Storing weights in 16-bit precision requires $p = 2$ bytes per parameter. Assuming activations are small relative to model parameters:
  $$\text{Bytes}_{\text{prefill}} \approx 2 \times d_{\text{in}} \times d_{\text{out}}$$
- **Operational Intensity**:
  $$
  I_{\text{prefill}} = \frac{2 \times T_{\text{prompt}} \times d_{\text{in}} \times d_{\text{out}}}{2 \times d_{\text{in}} \times d_{\text{out}}} = T_{\text{prompt}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

When $T_{\text{prompt}} = 1024$ tokens on an NVIDIA H100 ($I^* \approx 295$):
$$I_{\text{prefill}} = 1024 > 295$$
The prefill phase operates securely in the **Compute-Bound Regime**. Tensor Cores are fully saturated, and performance approaches $P_{\text{peak}}$.

### 5. Phase 2: The Decode Phase (GEMV)

In the <dfn id="def-decode">Decode Phase</dfn> (token generation), the model generates exactly **one** new token at time step $t$.

For the same weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and the current token's single-row activation vector $\mathbf{x}_t \in \mathbb{R}^{1 \times d_{\text{in}}}$:
- The projection is a General Matrix-Vector Multiplication (<abbr title="General Matrix-Vector Multiplication">GEMV</abbr>):
  $$\mathbf{y}_t = \mathbf{x}_t \mathbf{W} \in \mathbb{R}^{1 \times d_{\text{out}}}$$
- **Computation**:
  $$\text{FLOPs}_{\text{decode}} = 2 \times 1 \times d_{\text{in}} \times d_{\text{out}}$$
- **Memory Transferred**: Every single weight in $\mathbf{W}$ must be read from GPU HBM into registers:
  $$\text{Bytes}_{\text{decode}} \approx 2 \times d_{\text{in}} \times d_{\text{out}}$$
- **Operational Intensity**:
  $$
  I_{\text{decode}} = \frac{2 \times 1 \times d_{\text{in}} \times d_{\text{out}}}{2 \times d_{\text{in}} \times d_{\text{out}}} = 1.0 \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

Compare this to the H100 ridge point:
$$I_{\text{decode}} = 1.0 \ll 295.2$$

Because $I_{\text{decode}} \ll I^*$, token generation operates deep in the **Memory-Bandwidth-Bound Regime**. The attainable compute throughput is capped at:
$$
P_{\text{decode}} = I_{\text{decode}} \times B_{\text{peak}} = 1.0 \times 3.35 \times 10^{12} = 3.35 \text{ TFLOPS}
$$

On a 989 TFLOPS GPU, running single-token decode achieves:
$$\frac{3.35}{989} \approx 0.34\% \text{ of theoretical peak compute!}$$
Over 99% of the GPU's arithmetic silicon sits completely unutilized, starved of data while waiting for weights to trickle across the memory bus.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2009">2009</time> &mdash; <strong>The Roofline Model</strong> (<cite>Williams, Waterman, &amp; Patterson, CACM</cite>)</dt>
  <dd>Sam Williams and colleagues at UC Berkeley introduced the Roofline model to provide computer scientists with a visually intuitive, visually rigorous visual model relating processor peak performance, memory bandwidth, and algorithmic locality.</dd>
  <dt><time datetime="2020">2020</time> &mdash; <strong>The Autoregressive Serving Crisis</strong></dt>
  <dd>As models scaled from GPT-2 (1.5B) to GPT-3 (175B), engineers discovered that while training and prompt ingestion ran with near-linear hardware scaling, single-user interactive chat suffered massive latency drops. Systems researchers recognized that autoregressive decoding is not a matrix-matrix problem, but an extreme matrix-vector streaming bottleneck.</dd>
</dl>

### Why Couldn't We Just Use Simple Profiling?
Before the Roofline model was applied to LLMs, engineers frequently attempted to optimize decode speed by restructuring tensor operations, unrolling loops, or reordering operations. These optimizations yielded zero speedup because the bottleneck was not CPU/GPU core execution cycles, but the physical physics of the silicon memory bus: the speed of light and electrical capacitance moving electrons from HBM memory stacks to the compute die.

---

## Step 5: Concrete Toy Example

Let us trace a tiny concrete neural network through both phases with exact numbers.

### The Toy Model Specifications
- Hidden dimension: $d_{\text{in}} = 4, d_{\text{out}} = 4$.
- Weight matrix $\mathbf{W} \in \mathbb{R}^{4 \times 4}$ (16 scalar parameters).
- Precision: 16-bit floating point ($p = 2$ bytes per number).
- Total weight memory: $16 \times 2 = 32\text{ Bytes}$.

Let the weight matrix be:
$$
\mathbf{W} = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0 \\
0 & 0 & 2 & 1
\end{bmatrix}
$$

---

### Part A: The Prefill Phase (Prompt of $T = 3$ tokens)

Suppose the user sends a prompt of 3 tokens whose embedding representations are:
$$
\mathbf{X} = \begin{bmatrix}
1 & 0 & 2 & 1 \\
0 & 1 & 1 & 0 \\
2 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

#### 1. Computation (FLOPs)
We perform the matrix multiplication $\mathbf{Y} = \mathbf{X} \mathbf{W}$:
- Number of output cells: $3 \times 4 = 12$ numbers.
- Each cell requires 4 multiplications and 3 additions (or 4 multiply-accumulates $\approx 8$ FLOPs).
$$\text{Total Compute} = 2 \times 3 \times 4 \times 4 = 96 \text{ FLOPs}$$

#### 2. Memory Transferred (Bytes)
We stream the weight matrix $\mathbf{W}$ from memory **once**:
$$\text{Bytes Read} = 16 \text{ parameters} \times 2 \text{ bytes} = 32 \text{ Bytes}$$

#### 3. Operational Intensity
$$
I_{\text{prefill}} = \frac{96 \text{ FLOPs}}{32 \text{ Bytes}} = 3.0 \text{ FLOPs/Byte}
$$
Every single byte loaded from memory was reused 3 times across the parallel prompt tokens!

---

### Part B: The Decode Phase (Generating 1 Token)

Now the model must predict the very next token. Its current hidden state is a single row:
$$
\mathbf{x}_4 = \begin{bmatrix} 1 & 1 & 0 & 2 \end{bmatrix} \in \mathbb{R}^{1 \times 4}
$$

#### 1. Computation (FLOPs)
We compute $\mathbf{y}_4 = \mathbf{x}_4 \mathbf{W}$:
- Number of output cells: $1 \times 4 = 4$ numbers.
- Each cell requires 4 multiplications and 3 additions ($2 \times 4 = 8$ FLOPs).
$$\text{Total Compute} = 2 \times 1 \times 4 \times 4 = 32 \text{ FLOPs}$$

#### 2. Memory Transferred (Bytes)
To compute this single token, the processor must read the entire weight matrix $\mathbf{W}$ into its registers:
$$\text{Bytes Read} = 16 \text{ parameters} \times 2 \text{ bytes} = 32 \text{ Bytes}$$

#### 3. Operational Intensity
$$
I_{\text{decode}} = \frac{32 \text{ FLOPs}}{32 \text{ Bytes}} = 1.0 \text{ FLOP/Byte}
$$
The operational intensity plummeted by a factor of 3!

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 20.2:</strong> Comparison of Prefill vs. Decode in our toy model.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Metric</th>
      <th align="right">Prefill ($T=3$)</th>
      <th align="right">Decode ($T=1$)</th>
      <th align="center">Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tokens Processed</td>
      <td align="right">3</td>
      <td align="right">1</td>
      <td align="center">$3\times$</td>
    </tr>
    <tr>
      <td>Total FLOPs</td>
      <td align="right">96 FLOPs</td>
      <td align="right">32 FLOPs</td>
      <td align="center">$3\times$</td>
    </tr>
    <tr>
      <td>Weight Bytes Transferred</td>
      <td align="right">32 Bytes</td>
      <td align="right">32 Bytes</td>
      <td align="center"><strong>$1\times$ (Identical!)</strong></td>
    </tr>
    <tr>
      <td><strong>Operational Intensity ($I$)</strong></td>
      <td align="right"><strong>3.0 FLOPs/Byte</strong></td>
      <td align="right"><strong>1.0 FLOP/Byte</strong></td>
      <td align="center"><strong>$3\times$ Higher in Prefill</strong></td>
    </tr>
  </tbody>
</table>

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>Large Language Model inference is not a single unified workload; it is two radically different computational regimes separated by the Roofline ridge point:</p>
<p><strong>Prefill</strong> is a parallel matrix-matrix (<abbr title="General Matrix-Matrix Multiplication">GEMM</abbr>) operation that saturates GPU compute cores, while <strong>Decode</strong> is an iterative matrix-vector (<abbr title="General Matrix-Vector Multiplication">GEMV</abbr>) streaming bottleneck bounded strictly by memory bandwidth.</p>
<p>Every major inference acceleration technique (KV caching, continuous batching, speculative decoding, and quantization) exists to raise arithmetic intensity and overcome the memory bandwidth wall of the decode phase.</p>
</fieldset>
