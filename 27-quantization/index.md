# Chapter 27: Squeezing the Numbers (Quantization: AWQ, SmoothQuant, FP8, & INT4)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you are packing for a trip and you have a giant, heavy suitcase filled with 16-pound bowling balls.
> 
> You can barely lift the suitcase, the baggage fee costs a fortune, and dragging it through the airport takes forever.
> 
> Then, a clever toy maker visits you:
> *"Why carry 16-pound solid bowling balls when we can replace them with 4-pound lightweight wooden balls that look and roll almost the exact same way?"*
> 
> You swap the 16-pound balls for 4-pound balls (**INT4 Quantization**). Suddenly, your suitcase weighs one-fourth as much! You can sprint through the airport 4 times faster, and you can pack 4 times as many toys into the same trunk.
> 
> But you must be careful: if your suitcase contains a fragile gold crown (**an activation outlier**), and you blindly crush it into a crude wooden block, the crown breaks.
> Clever quantization methods like **SmoothQuant** and **AWQ** act like smart bubble wrap: they identify the 1% of fragile gold pieces and protect them with extra care, while safely shrinking the 99% of ordinary clothes down to tiny 4-bit sizes.

---

## Step 2: The Bridging Question

How do we convert high-precision 16-bit floating point numbers into compact 8-bit or 4-bit integers on a computer chip, and how do we perform matrix multiplication $\mathbf{Y} = \mathbf{X} \mathbf{W}$ when the numbers are represented with discrete integer grids?

In Chapter 20, we established that the autoregressive Decode phase is strictly **memory bandwidth bound**:
$$I_{\text{decode}} \approx 1.0 \text{ FLOP/Byte} \ll I^*$$
During decode, the GPU spends nearly all its time waiting for weights to travel from HBM to compute registers.

The data size of a parameter depends directly on its numerical precision:
- **FP16 / BF16**: 16 bits = 2.0 bytes per weight.
- **FP8 (Hopper/Blackwell)**: 8 bits = 1.0 byte per weight ($2\times$ memory compression).
- **INT4 (AWQ / GPTQ)**: 4 bits = 0.5 bytes per weight ($4\times$ memory compression).

If we shrink a 70B parameter model from FP16 (140 GB) to INT4 (35 GB):
1. The entire model fits into a single 40GB/80GB GPU instead of requiring a multi-GPU cluster.
2. The bytes transferred per decode step drop by $4\times$, **quadrupling generation speed**!

The bridging question is:
$$\text{How do we map continuous real numbers } x \in \mathbb{R} \text{ onto a discrete integer grid } \{-8, \dots, 7\} \text{ with minimal mathematical distortion, and how do we tame explosive outlier activations?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        UNIFORM SYMMETRIC INT4 QUANTIZATION GRID                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Continuous FP16 Range:                                                                 │
│   -x_max ────────────────────────────── 0 ────────────────────────────── +x_max        │
│      ▲                                  ▲                                  ▲           │
│      │ Scale s = x_max / 7              │                                  │           │
│      ▼                                  ▼                                  ▼           │
│ Discrete INT4 Grid:                                                                    │
│     -7    -6    -5    -4    -3   -2   -1    0   +1   +2   +3   +4   +5   +6   +7       │
│   [ 4 bits: 1001 ]                   [ 0000 ]                   [ 4 bits: 0111 ]       │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 27.1:</strong> Uniform symmetric quantization maps a continuous dynamic range onto 15 discrete signed integer bins using a scalar scale factor s.</figcaption>
</figure>

### 1. Uniform Symmetric Quantization &amp; Dequantization

For a $b$-bit signed integer (e.g., $b = 4 \implies [-7, 7]$, or $b = 8 \implies [-127, 127]$):

Let $\mathbf{w} \in \mathbb{R}^d$ be a vector of continuous weights. The <dfn id="def-scale">Quantization Scale</dfn> $s$ is:

$$
s = \frac{\max_{i} |w_i|}{2^{b-1} - 1} \in \mathbb{R}^+
$$

The <dfn id="def-quant">Quantization Function</dfn> $Q(x)$ maps each real number to its nearest integer grid point:

$$
q = Q(w) = \operatorname{clip}\left(\left\lfloor \frac{w}{s} \right\rceil, \; -(2^{b-1}-1), \; 2^{b-1}-1\right) \in \mathbb{Z}
$$

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

The <dfn id="def-dequant">Dequantization Function</dfn> $\hat{w} = \tilde{Q}(q)$ reconstructs the real approximation:

$$
\hat{w} = s \times q \approx w
$$

The quantization error is bounded by half the grid step: $|w - \hat{w}| \le \frac{s}{2}$.

---

### 2. The Weight-Only vs. Weight-Activation Dichotomy

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 27.1:</strong> Classification of quantization schemes for LLM serving.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Quantization Strategy</th>
      <th align="center">Weight Precision</th>
      <th align="center">Activation Precision</th>
      <th align="left">Hardware Execution Mechanism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>FP16 Baseline</strong></td>
      <td align="center">16-bit</td>
      <td align="center">16-bit</td>
      <td>Native FP16 Tensor Cores</td>
    </tr>
    <tr>
      <td><strong>W4A16 (AWQ / GPTQ)</strong></td>
      <td align="center"><strong>4-bit INT</strong></td>
      <td align="center">16-bit FP</td>
      <td>Weights stored in 4-bit, dequantized on-the-fly in SRAM to FP16 for compute</td>
    </tr>
    <tr>
      <td><strong>W8A8 (SmoothQuant)</strong></td>
      <td align="center"><strong>8-bit INT</strong></td>
      <td align="center"><strong>8-bit INT</strong></td>
      <td>Native INT8 Tensor Core GEMM (saturates integer compute cores)</td>
    </tr>
    <tr>
      <td><strong>FP8 (Hopper / Ada)</strong></td>
      <td align="center"><strong>8-bit FP (E4M3)</strong></td>
      <td align="center"><strong>8-bit FP (E4M3)</strong></td>
      <td>Native FP8 Tensor Core GEMM ($2\times$ faster compute + $2\times$ less RAM)</td>
    </tr>
  </tbody>
</table>

---

### 3. SmoothQuant: Resolving the Activation Outlier Catastrophe

In 2022, Dettmers et al. discovered that once models exceed 6.7B parameters, <dfn id="def-outliers">Emergent Outlier Channels</dfn> appear: 0.1% of activation channels suddenly exhibit magnitudes up to 100x larger than normal channels (e.g., normal values $\approx 2.0$, outlier channel $\approx 150.0$).
Quantizing activations across this massive dynamic range crushes the other 99.9% of channels into 0, destroying model reasoning.

**SmoothQuant** solves this via a mathematically equivalent diagonal transformation:

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = \left(\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}\right) \cdot \left(\operatorname{diag}(\mathbf{s}) \mathbf{W}\right) = \hat{\mathbf{X}} \hat{\mathbf{W}}
$$

where $\mathbf{s} \in \mathbb{R}^{d}$ is a per-channel smoothing scale vector.
To balance the quantization difficulty evenly between activations and weights, each channel scale $s_j$ is chosen via the geometric migration formula:

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}
$$

where $\alpha \in [0, 1]$ is a migration hyperparameter (typically $\alpha = 0.5$).
SmoothQuant migrates the outlier peak out of the activations and smooths it into the weights, enabling seamless **W8A8 INT8 Tensor Core execution** with zero loss of accuracy!

---

### 4. AWQ: Activation-Aware Weight Quantization

Lin et al. (2024) observed that in W4A16 quantization, not all weights are equally important: **only 1% of weights correspond to salient activation features**.

AWQ protects these critical weights by finding a per-channel scaling matrix $\mathbf{S} = \operatorname{diag}(\mathbf{s})$ that minimizes the output reconstruction error:

$$
\mathbf{W}^* = \arg\min_{\mathbf{W}'} \left\| \mathbf{W} \mathbf{X} - Q(\mathbf{W} \mathbf{S}) \mathbf{S}^{-1} \mathbf{X} \right\|_F^2
$$

By scaling salient channels up before integer truncation, AWQ preserves model perplexity at 4 bits without requiring slow backpropagation or retraining.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2022">2022</time> &mdash; <strong>LLM.int8() &amp; Emergent Outliers</strong> (<cite>Tim Dettmers et al., University of Washington</cite>)</dt>
  <dd>Discovered that language models exhibit an abrupt phase transition around 6.7B parameters where extreme activation outliers emerge in specific feature dimensions.</dd>
  <dt><time datetime="2023">2023</time> &mdash; <strong>SmoothQuant</strong> (<cite>Guangxuan Xiao et al., MIT, ICML 2023</cite>)</dt>
  <dd>Introduced mathematically equivalent activation-weight scale migration, allowing 100B+ models to run fully in INT8 compute without quality degradation.</dd>
  <dt><time datetime="2023">2023–2024</time> &mdash; <strong>GPTQ &amp; AWQ (W4A16 Standard)</strong> (<cite>Frantar et al., ICLR 2023; Lin et al., MLSys 2024</cite>)</dt>
  <dd>Established 4-bit weight-only quantization as the global open-source serving standard, enabling 70B parameter models to run on consumer hardware.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace symmetric INT4 quantization on a tiny 4-element weight vector by hand.

Let continuous FP16 weights be:
$$
\mathbf{w} = [-6.3, \; 1.8, \; 0.4, \; 7.0]
$$
Target: 4-bit signed integer grid with values in $\{-7, -6, \dots, 0, \dots, +7\}$.

---

### Step 1: Calculate the Quantization Scale ($s$)
Find the maximum absolute value:
$$
\max |w_i| = \max(|-6.3|, |1.8|, |0.4|, |7.0|) = 7.0
$$
Grid max for 4-bit signed integer: $2^{4-1} - 1 = 7$.
$$
s = \frac{7.0}{7} = 1.0
$$

---

### Step 2: Quantize Each Element ($q = \lfloor w / s \rceil$)
1. For $w_1 = -6.3$:
   $$q_1 = \left\lfloor \frac{-6.3}{1.0} \right\rceil = \lfloor -6.3 \rceil = -6$$
2. For $w_2 = 1.8$:
   $$q_2 = \left\lfloor \frac{1.8}{1.0} \right\rceil = \lfloor 1.8 \rceil = +2$$
3. For $w_3 = 0.4$:
   $$q_3 = \left\lfloor \frac{0.4}{1.0} \right\rceil = \lfloor 0.4 \rceil = 0$$
4. For $w_4 = 7.0$:
   $$q_4 = \left\lfloor \frac{7.0}{1.0} \right\rceil = \lfloor 7.0 \rceil = +7$$

The quantized INT4 vector stored in GPU memory is:
$$
\mathbf{q} = [-6, \; +2, \; 0, \; +7] \in \mathbb{Z}^4
$$
Memory used: $4 \times 4\text{ bits} = 16\text{ bits}$ (down from 64 bits in FP16, a **$4\times$ reduction**!).

---

### Step 3: Dequantization &amp; Error Analysis
Reconstruct the real approximation $\hat{\mathbf{w}} = s \times \mathbf{q}$:
$$
\hat{\mathbf{w}} = 1.0 \times [-6, \; +2, \; 0, \; +7] = [-6.0, \; 2.0, \; 0.0, \; 7.0]
$$

Calculate the quantization error $\mathbf{e} = \mathbf{w} - \hat{\mathbf{w}}$:
- $e_1 = -6.3 - (-6.0) = -0.3$
- $e_2 = 1.8 - 2.0 = -0.2$
- $e_3 = 0.4 - 0.0 = +0.4$
- $e_4 = 7.0 - 7.0 = 0.0$

Notice that every error $|e_i| \le \frac{s}{2} = 0.5$! The overall vector representation was preserved with tiny error while cutting memory consumption by 75%.

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>Because LLM autoregressive decoding is strictly memory-bandwidth bound, <strong>Quantization</strong> delivers an immediate, linear wall-clock speedup by shrinking weight tensors from 16 bits to 8 or 4 bits.</p>
<p>Modern breakthroughs like <strong>SmoothQuant</strong> and <strong>AWQ</strong> eliminate historical accuracy degradation by mathematically isolating and protecting outlier activation channels, enabling massive 70B+ models to run at $4\times$ speed on commodity hardware.</p>
</fieldset>
