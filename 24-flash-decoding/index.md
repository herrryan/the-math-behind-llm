# Chapter 24: Decode Parallelism (FlashDecoding & Split-K Attention)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you have an office with 132 smart detectives sitting at their desks.
> 
> A client rushes into the office with a single mystery question and hands over a 10,000-page historical book: *"Find the clue that answers my question!"*
> 
> In traditional decode attention:
> You assign the entire 10,000-page book to **just one detective**. That poor detective sits at their desk flipping page by page through all 10,000 pages for 2 hours. Meanwhile, the other 131 detectives sit at their desks playing with their thumbs and watching the one person sweat! 99% of your workforce is completely idle.
> 
> Then, a clever captain invents **FlashDecoding (Split-K)**:
> *"Tear the 10,000-page book into 100 thin chapters of 100 pages each! Hand Chapter 1 to Detective 1, Chapter 2 to Detective 2, Chapter 3 to Detective 3, all the way to Detective 100. Everyone search your 100 pages at the exact same second!"*
> 
> In just 10 seconds, all 100 detectives find the best clue in their own chapter. They walk to the whiteboard, write down their top clue and their confidence score, and in 1 final second, the captain merges the clues to find the absolute truth.
> 
> By putting all 132 detectives to work at the exact same time, the task finishes in seconds instead of hours.

---

## Step 2: The Bridging Question

How do we convert tearing a 10,000-page book into GPU CUDA thread blocks, and how can independent streaming multiprocessors (<abbr title="Streaming Multiprocessor">SM</abbr>s) compute attention on isolated chunks of time without losing the global Softmax denominator?

In the decode phase, the query is just a single token:
$$\mathbf{q} \in \mathbb{R}^{B \times 1 \times H \times d_{\text{head}}}$$
while the past KV cache has grown to length $T_{\text{ctx}}$:
$$\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times T_{\text{ctx}} \times H \times d_{\text{head}}}$$

FlashAttention parallelizes computation across two dimensions: **Batch size ($B$)** and **Number of heads ($H$)**.
Total parallel thread blocks = $B \times H$.

Now observe the hardware crisis:
- An NVIDIA H100 GPU has **132 Streaming Multiprocessors (SMs)**.
- For a single-user request ($B = 1$) on a model with Grouped-Query Attention ($H_{\text{kv}} = 8$):
  $$\text{Thread Blocks} = 1 \times 8 = 8$$
  Only 8 out of 132 SMs are doing work! The remaining 124 SMs sit completely idle ($< 6\%$ GPU occupancy) while those 8 SMs crawl through a 64k-token context.

The bridging question is:
$$\text{How can we parallelize along the historical sequence dimension } T_{\text{ctx}} \text{ (the Key/Value dimension), and how do we mathematically stitch the partial Softmax results together across CUDA blocks?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FLASHDECODING SPLIT-K ATTENTION PIPELINE                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Single Query Token:  q [1 x d]                                                         │
│ Historical KV Cache (T_ctx = 64k) ──► Split into S Chunks along Time (K) Dimension     │
│                                                                                        │
│ PHASE 1: PARALLEL CHUNK ATTENTION (Dispatched across S independent GPU SMs)            │
│   SM 0:   q @ K[0:C].T       ──► Local FlashAttn ──► (m_0, d_0, o_0) [1 x d]           │
│   SM 1:   q @ K[C:2C].T      ──► Local FlashAttn ──► (m_1, d_1, o_1) [1 x d]           │
│   ...                                                                                  │
│   SM S-1: q @ K[(S-1)C:SC].T ──► Local FlashAttn ──► (m_{S-1}, d_{S-1}, o_{S-1})      │
│                                                                                        │
│ PHASE 2: PARALLEL LOG-SUM-EXP REDUCTION (Final Fast Merge Kernel)                      │
│   Global Max:         m* = max(m_0, m_1, ..., m_{S-1})                                 │
│   Global Denominator: d* = sum_{s} d_s * exp(m_s - m*)                                 │
│   Final Output:       o* = (1 / d*) * sum_{s} exp(m_s - m*) * o_s  [1 x d]             │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 24.1:</strong> FlashDecoding splits the historical KV cache into S chunks along the sequence dimension, running local FlashAttention across S independent GPU SMs in parallel, followed by a fast reduction merge.</figcaption>
</figure>

### 1. Split-K Partitioning of the KV Cache

Let $T_{\text{ctx}}$ be the total context length, and let $S$ be the number of parallel splits (typically chosen such that $B \times H \times S \ge \text{Total SMs}$).
The sequence is divided into $S$ contiguous chunks of size:
$$
C = \left\lceil \frac{T_{\text{ctx}}}{S} \right\rceil
$$

For each chunk $s \in \{0, 1, \dots, S - 1\}$:
- The chunk key slice is $\mathbf{K}^{(s)} = \mathbf{K}[s \cdot C : (s+1) \cdot C, :] \in \mathbb{R}^{C \times d_{\text{head}}}$
- The chunk value slice is $\mathbf{V}^{(s)} = \mathbf{V}[s \cdot C : (s+1) \cdot C, :] \in \mathbb{R}^{C \times d_{\text{head}}}$

---

### 2. Phase 1: Independent Local Chunk Attention

Each of the $S$ chunks is assigned to a separate CUDA thread block running on an independent Streaming Multiprocessor.

Thread block $s$ calculates the local raw attention scores:
$$
\mathbf{s}^{(s)} = \frac{\mathbf{q} (\mathbf{K}^{(s)})^\top}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{1 \times C}
$$

It computes three local summary statistics:
1. **Local Maximum Score**:
   $$
   m^{(s)} = \max_{j=1}^C s_j^{(s)}
   $$
2. **Local Unnormalized Denominator**:
   $$
   d^{(s)} = \sum_{j=1}^C \exp\left(s_j^{(s)} - m^{(s)}\right)
   $$
3. **Local Unnormalized Output Vector**:
   $$
   \mathbf{o}^{(s)} = \sum_{j=1}^C \exp\left(s_j^{(s)} - m^{(s)}\right) \mathbf{v}_j^{(s)} \in \mathbb{R}^{1 \times d_{\text{head}}}
   $$

These $S$ triplets $\left(m^{(s)}, d^{(s)}, \mathbf{o}^{(s)}\right)$ are written to small temporary buffers in GPU memory. Because $S$ is small (e.g., $S = 64$), this buffer is negligible in size ($\approx \text{a few kilobytes}$).

---

### 3. Phase 2: Mathematical Reduction Across Splits

A lightweight reduction kernel reads the $S$ summary triplets and reconstructs the globally exact Softmax attention output.

#### Step A: Global Maximum Finding
$$
m^* = \max_{s=0}^{S-1} m^{(s)}
$$

#### Step B: Global Denominator Reconstruction
To combine denominators computed with different local reference baselines $m^{(s)}$, each local sum is scaled by $e^{m^{(s)} - m^*}$:
$$
d^* = \sum_{s=0}^{S-1} d^{(s)} \cdot \exp\left(m^{(s)} - m^*\right)
$$

#### Step C: Global Output Rescaling
The final, globally normalized attention vector $\mathbf{o}^* \in \mathbb{R}^{1 \times d_{\text{head}}}$ is:
$$
\mathbf{o}^* = \frac{1}{d^*} \sum_{s=0}^{S-1} \exp\left(m^{(s)} - m^*\right) \mathbf{o}^{(s)}
$$

This reduction is mathematically identical to running attention across the entire sequence as a single monolithic block, but achieves **near-100% GPU SM occupancy**.

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 24.1:</strong> Decode Phase Comparison: FlashAttention vs. FlashDecoding (Batch=1, Heads=8, H100 with 132 SMs).</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Metric</th>
      <th align="center">Standard FlashAttention (Decode)</th>
      <th align="center">FlashDecoding (Split-K)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Parallelism Dimensions</strong></td>
      <td align="center">Batch $\times$ Heads ($B \times H$)</td>
      <td align="center">Batch $\times$ Heads $\times$ Splits ($B \times H \times S$)</td>
    </tr>
    <tr>
      <td><strong>Active GPU Blocks ($B=1, H=8$)</strong></td>
      <td align="center">8 blocks</td>
      <td align="center"><strong>$8 \times 32 = 256$ blocks</strong></td>
    </tr>
    <tr>
      <td><strong>GPU SM Occupancy</strong></td>
      <td align="center">$\frac{8}{132} \approx 6.0\%$ (Severely starved)</td>
      <td align="center"><strong>100% (All 132 SMs fully saturated)</strong></td>
    </tr>
    <tr>
      <td><strong>Decode Latency (64k context)</strong></td>
      <td align="center">High (linear in sequence length)</td>
      <td align="center"><strong>Up to 8x Faster Generation!</strong></td>
    </tr>
  </tbody>
</table>

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2023">2023</time> &mdash; <strong>Flash-Decoding for Long-Context Inference</strong> (<cite>Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov, Dao-AILab</cite>)</dt>
  <dd>Following the widespread adoption of FlashAttention for training, the team noticed that inference generation at long contexts failed to saturate modern GPUs. They introduced Split-K sequence parallelization, reducing long-context decode latency by up to 8x.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>FlashDecoding++</strong> (<cite>Hong et al., Tsinghua University</cite>)</dt>
  <dd>Eliminated the reduction synchronization overhead by maintaining static universal maximum priors, further accelerating autoregressive serving on heterogeneous GPUs.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace Split-K attention on a sequence of length $T_{\text{ctx}} = 4$ split into $S = 2$ chunks of size $C = 2$.

Let head dimension $d = 2$, and suppose query $\mathbf{q} = [1.0, 0.0]$.
Suppose the calculated unscaled scores across the 4 historical tokens are:
$$
\mathbf{s} = [s_0, s_1, s_2, s_3] = [1.0, \; 3.0, \; 2.0, \; 4.0]
$$
and their corresponding cached values are:
$$
\mathbf{v}_0 = [1, 0], \quad \mathbf{v}_1 = [0, 1], \quad \mathbf{v}_2 = [1, 1], \quad \mathbf{v}_3 = [2, 0]
$$

---

### Phase 1: Local Chunk Computation

#### Chunk 0 (Tokens 0 and 1): Scores $[1.0, 3.0]$
1. Local max: $m^{(0)} = \max(1.0, 3.0) = 3.0$
2. Exponentials:
   $$e^{1.0 - 3.0} = e^{-2} \approx 0.1353, \quad e^{3.0 - 3.0} = e^0 = 1.0000$$
3. Local denominator:
   $$d^{(0)} = 0.1353 + 1.0000 = 1.1353$$
4. Local unnormalized output:
   $$\mathbf{o}^{(0)} = 0.1353 \cdot [1, 0] + 1.0000 \cdot [0, 1] = [0.1353, \; 1.0000]$$

---

#### Chunk 1 (Tokens 2 and 3): Scores $[2.0, 4.0]$
1. Local max: $m^{(1)} = \max(2.0, 4.0) = 4.0$
2. Exponentials:
   $$e^{2.0 - 4.0} = e^{-2} \approx 0.1353, \quad e^{4.0 - 4.0} = e^0 = 1.0000$$
3. Local denominator:
   $$d^{(1)} = 0.1353 + 1.0000 = 1.1353$$
4. Local unnormalized output:
   $$\mathbf{o}^{(1)} = 0.1353 \cdot [1, 1] + 1.0000 \cdot [2, 0] = [0.1353 + 2.0, \; 0.1353 + 0] = [2.1353, \; 0.1353]$$

---

### Phase 2: Reduction Kernel

Now merge the two chunk summaries:
Chunk 0: $(m^{(0)}=3.0, \; d^{(0)}=1.1353, \; \mathbf{o}^{(0)}=[0.1353, 1.0])$
Chunk 1: $(m^{(1)}=4.0, \; d^{(1)}=1.1353, \; \mathbf{o}^{(1)}=[2.1353, 0.1353])$

1. **Global Maximum**:
   $$m^* = \max(3.0, 4.0) = 4.0$$
2. **Rescaling Factors**:
   $$\alpha_0 = e^{m^{(0)} - m^*} = e^{3.0 - 4.0} = e^{-1} \approx 0.3679$$
   $$\alpha_1 = e^{m^{(1)} - m^*} = e^{4.0 - 4.0} = e^0 = 1.0000$$
3. **Global Denominator**:
   $$d^* = \alpha_0 d^{(0)} + \alpha_1 d^{(1)} = (0.3679 \times 1.1353) + (1.0 \times 1.1353) \approx 0.4177 + 1.1353 = 1.5530$$
4. **Final Globally Normalized Output**:
   $$\mathbf{o}^* = \frac{1}{d^*} \left(\alpha_0 \mathbf{o}^{(0)} + \alpha_1 \mathbf{o}^{(1)}\right)$$
   $$\alpha_0 \mathbf{o}^{(0)} = 0.3679 \times [0.1353, 1.0000] = [0.0498, 0.3679]$$
   $$\alpha_1 \mathbf{o}^{(1)} = 1.0000 \times [2.1353, 0.1353] = [2.1353, 0.1353]$$
   $$\text{Sum} = [0.0498 + 2.1353, \; 0.3679 + 0.1353] = [2.1851, \; 0.5032]$$
   $$\mathbf{o}^* = \left[\frac{2.1851}{1.5530}, \; \frac{0.5032}{1.5530}\right] \approx [1.4070, \; 0.3240]$$

This matches standard attention perfectly while executing across two independent GPU cores!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>While FlashAttention was designed for training where prompts parallelize across tokens, single-token <strong>Decode</strong> severely underutilizes GPU silicon due to lack of thread-block parallelism.</p>
<p><strong>FlashDecoding</strong> introduces <strong>Split-K parallelism</strong>, carving the historical KV cache across the time dimension to unleash 100% GPU occupancy during generation and accelerating long-context decoding by up to $8\times$.</p>
</fieldset>
