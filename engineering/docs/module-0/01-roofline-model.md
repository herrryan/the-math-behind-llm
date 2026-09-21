# Chapter E01: The Roofline Model & Operational Intensity

## Step 1: Hardware Intuition

Imagine you own a high-volume fruit smoothie stand with two stations:
1. **The Fruit Peeler (Memory Fetch):** A worker who fetches whole frozen strawberries from the freezer, thaws them, and dumps them into the blender pitcher. The peeler can unpack at most **3 kilograms of fruit per minute**.
2. **The Industrial Blender (Compute Units / Tensor Cores):** An ultra-powerful motorized blender that can puree fruit at a rate of **2,000 rotations per minute**.

Now, suppose a customer orders a **Simple Strawberry Water** (1 strawberry blended into 1 liter of water). The blender needs only 2 rotations to chop the strawberry, but the worker spends 20 seconds hauling the heavy water jug from the freezer. The blender sits idle for 19.9 seconds. Your smoothie stand is **Worker-Bound (Memory-Bound)**.

Next, suppose another customer orders a **Triple-Cream Smoothie Paste** (a thick puree requiring 1,000 blender chops per strawberry). Now, once the worker brings a small bowl of strawberries, the blender runs continuously for 30 seconds straight. The worker can take a rest while the blender whirs at 100% capacity. Your smoothie stand is **Blender-Bound (Compute-Bound)**.

The ratio of **blender work (FLOPs)** to **ingredients fetched (Bytes)** is called **Operational Intensity**. The **Roofline Model** is simply the chart that tells you whether your code is waiting on the blender or waiting on the worker.

---

## Step 2: Silicon Micro-Mechanics

The Roofline Model (Williams, Waterman, & Patterson, 2009) is a visual performance model that relates hardware peak compute capability, memory bandwidth, and software arithmetic intensity.

```
Attainable Performance [TFLOPS]
         ▲
Peak     │                          /──────────────────────── (Compute Ceiling: P_peak)
Compute  │                         /
Ceiling  │                        /
         │                       /
         │                      / ◄── Memory Bandwidth Ceiling
         │                     /      Slope = Peak Memory Bandwidth (BW_mem)
         │                    /
         │                   /
         │                  /
         │                 /
         │                / │
         └───────────────┴──┴────────────────────────────────► Operational Intensity (I)
                         0  I* (Hardware Balance Point)         [FLOPs / Byte]
                          ▲
                          │
          [MEMORY-BOUND]  │  [COMPUTE-BOUND REGIME]
          I < I*          │  I > I*
```

### Key Parameters:
1. **$P_{\text{peak}}$ (Peak Compute Throughput):** The theoretical maximum number of floating-point operations the hardware can execute per second (e.g., $2,000\text{ TFLOPS}$ for NVIDIA H100 SXM in FP16 with Tensor Core structural sparsity, or $1,000\text{ TFLOPS}$ dense).
2. **$\text{BW}_{\text{mem}}$ (Peak Memory Bandwidth):** The maximum rate at which bytes can be read from device HBM into on-chip cache (e.g., $3.35\text{ TB/s}$ for H100 SXM HBM3).
3. **Operational Intensity ($I$):** The number of floating-point operations executed per byte of data transferred from HBM:
   $$I = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Total Memory Traffic to/from HBM (Bytes)}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$
4. **$I^*$ (The Hardware Ridge Point / Balance Point):** The critical inflection point:
   $$I^* = \frac{P_{\text{peak}}}{\text{BW}_{\text{mem}}}$$

For an NVIDIA H100 SXM GPU ($P_{\text{peak}} \approx 1,000\text{ TFLOPS}$ dense FP16, $\text{BW}_{\text{mem}} = 3.35\text{ TB/s}$):

$$I^* = \frac{1,000 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx 298.5 \text{ FLOPs/Byte}$$

With 2:4 structured sparsity ($P_{\text{peak}} = 2,000\text{ TFLOPS}$), $I^* \approx 597\text{ FLOPs/Byte}$.

---

## Step 3: Cross-Component Coupling

Every component in an LLM operates in one of these two regimes:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">LLM Component</th>
      <th align="left" width="20%">Typical Intensity $I$</th>
      <th align="left" width="18%">Operating Regime</th>
      <th align="left" width="40%">Hardware Bottleneck &amp; Optimization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Prefill GEMM (Prompt)</strong></td>
      <td>$I = 100\text{--}500+$</td>
      <td>Compute-Bound</td>
      <td>Tensor Core bound. Maximize tile size, align dimensions to 64/128 bytes.</td>
    </tr>
    <tr>
      <td><strong>Decode GEMV (Generation)</strong></td>
      <td>$I = 1\text{--}5$ (at $B=1$)</td>
      <td>Memory-Bound</td>
      <td>HBM bandwidth bound. Increase batch size, apply W4/W8 quantization, use GQA.</td>
    </tr>
    <tr>
      <td><strong>LayerNorm / RMSNorm</strong></td>
      <td>$I \approx 2$</td>
      <td>Memory-Bound</td>
      <td>HBM bandwidth bound. Must be fused with residual additions into a single kernel.</td>
    </tr>
    <tr>
      <td><strong>Softmax</strong></td>
      <td>$I \approx 2.5$</td>
      <td>Memory-Bound</td>
      <td>HBM bandwidth bound. Must be tiled in SRAM (Online Softmax in FlashAttention).</td>
    </tr>
    <tr>
      <td><strong>Elementwise Activations (GELU/Swish)</strong></td>
      <td>$I \approx 1\text{--}2$</td>
      <td>Memory-Bound</td>
      <td>HBM bandwidth bound. Fuse directly into GEMM output epilogue.</td>
    </tr>
  </tbody>
</table>

### How Batch Size Shifts the Regime
Consider a matrix-vector product vs. a matrix-matrix multiply:
- For batch size $B = 1$ (decoding single token): We multiply a vector $\mathbf{x} \in \mathbb{R}^{1 \times K}$ by weight matrix $\mathbf{W} \in \mathbb{R}^{K \times N}$.
  $$\text{FLOPs} = 2 K N$$
  $$\text{Bytes fetched from HBM} = 2 K N \text{ (weights)} + 2 K \text{ (input)} + 2 N \text{ (output)} \approx 2 K N \text{ bytes (in FP16)}$$
  $$\text{Intensity } I = \frac{2 K N \text{ FLOPs}}{2 K N \text{ Bytes}} = 1 \text{ FLOP/Byte}$$
  Since $1 \ll 298.5$, single-token decoding is **deeply inside the memory-bound regime**.

- For batch size $B = 128$ (prefill or batched decoding): We multiply an input matrix $\mathbf{X} \in \mathbb{R}^{B \times K}$ by $\mathbf{W} \in \mathbb{R}^{K \times N}$.
  $$\text{FLOPs} = 2 B K N$$
  $$\text{Bytes fetched} = 2 K N \text{ (weights are reused } B \text{ times!)} + 2 B K + 2 B N$$
  $$\text{Intensity } I \approx \frac{2 B K N}{2 K N} = B \text{ FLOPs/Byte} = 128 \text{ FLOPs/Byte}$$
  As batch size $B$ grows toward 300+, the operational intensity crosses $I^*$, and the operation transitions from memory-bound to **compute-bound**.

---

## Step 4: The Exact Performance Formula

The maximum attainable execution throughput $P$ (in TFLOPS) is given by the Roofline inequality:

$$P(I) = \min\left(P_{\text{peak}}, \; I \times \text{BW}_{\text{mem}}\right)$$

The actual execution time of a kernel processing $F$ FLOPs with memory traffic $M$ bytes is:

$$T_{\text{exec}} = \max\left(\frac{F}{P_{\text{peak}}}, \; \frac{M}{\text{BW}_{\text{mem}}}\right)$$

Hardware Efficiency (Model FLOPs Utilization - MFU) can be expressed as:

$$\text{MFU} = \frac{P(I)}{P_{\text{peak}}} = \min\left(1, \; \frac{I \times \text{BW}_{\text{mem}}}{P_{\text{peak}}}\right) = \min\left(1, \; \frac{I}{I^*}\right)$$

When $I < I^*$:

$$\text{MFU} = \frac{I}{I^*}$$

This elegant formula reveals that whenever a kernel is memory-bound, **its maximum attainable MFU is strictly capped by the ratio $\frac{I}{I^*}$**, regardless of how many Tensor Cores exist on the chip!

---

## Step 5: Concrete Benchmark Walkthrough

Let us benchmark a standard **RMSNorm** operation across a batch of tokens on an NVIDIA H100 SXM GPU:
- Model hidden dimension: $d_{\text{model}} = 8192$ (LLaMA-3 70B)
- Batch dimension: $S = 2048$ tokens
- Datatype: 16-bit Float (BF16, 2 bytes/element)
- Hardware: NVIDIA H100 ($P_{\text{peak}} = 1,000\text{ TFLOPS}$, $\text{BW}_{\text{mem}} = 3.35\text{ TB/s}$, $I^* = 298.5\text{ FLOPs/Byte}$)

### Step 5.1: Count Total FLOPs
RMSNorm computes for each token vector $\mathbf{x}$:
$$y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$$
1. Square each element: $d$ FLOPs
2. Sum elements: $d$ FLOPs
3. Divide by $d$, add $\epsilon$, square root: 3 FLOPs
4. Divide each element and multiply by scale $\gamma_i$: $2d$ FLOPs
Total per token: $4d + 3 \approx 4d$ FLOPs.

$$\text{Total FLOPs} = 2048 \times (4 \times 8192) \approx 6.71 \times 10^7 \text{ FLOPs} = 67.1 \text{ MFLOPs}$$

### Step 5.2: Count Total Memory Traffic (Unfused Kernel)
- Read vector $\mathbf{x}$ from HBM: $2048 \times 8192 \times 2 \text{ bytes} = 33.55\text{ MB}$
- Read scale vector $\boldsymbol{\gamma}$ from HBM: $8192 \times 2 \text{ bytes} = 0.016\text{ MB}$
- Write output vector $\mathbf{y}$ to HBM: $2048 \times 8192 \times 2 \text{ bytes} = 33.55\text{ MB}$
$$\text{Total Bytes Transferred} \approx 67.12 \text{ MB}$$

### Step 5.3: Compute Operational Intensity and Performance
$$I = \frac{67.1 \times 10^6 \text{ FLOPs}}{67.12 \times 10^6 \text{ Bytes}} \approx 1.0 \text{ FLOP/Byte}$$

Maximum attainable throughput:
$$P(1.0) = \min(1000\text{ TFLOPS}, \; 1.0 \times 3.35\text{ TB/s}) = 3.35 \text{ TFLOPS}$$

Attainable MFU:
$$\text{MFU} = \frac{1.0}{298.5} \approx 0.33\%$$

### Step 5.4: Minimum Execution Time
$$T_{\text{exec}} = \frac{67.12 \times 10^6 \text{ Bytes}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx 20.0 \times 10^{-6} \text{ seconds} = 20.0 \;\mu\text{s}$$

Notice: The compute units would have finished the $67.1\text{ MFLOPs}$ in $0.067\;\mu\text{s}$. But because memory transfer took $20.0\;\mu\text{s}$, the kernel runs $300\times$ slower than compute capacity.

---

## Step 6: Core Systems Takeaway

> You cannot optimize a kernel by speeding up its math if its operational intensity is below the hardware balance point $I^*$. When $I < I^*$, your only options to improve throughput are reducing memory traffic (via operator fusion or quantization) or increasing data reuse (via larger batch sizes or SRAM caching).
