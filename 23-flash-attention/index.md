# Chapter 23: The IO-Aware Speedup (FlashAttention & Online Softmax)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you are a chef making a salad in a restaurant kitchen.
> 
> You have a tiny wooden cutting board right in front of you (**SRAM**), and a massive cold-storage pantry down in the basement (**HBM**).
> 
> In the traditional way of cooking (**Standard Attention**):
> 1. You slice a carrot, walk all the way down the basement stairs, set it on a shelf, and walk back up.
> 2. You slice a cucumber, walk down to the basement, set it on the shelf, and walk back up.
> 3. Then you walk down to the basement, pour dressing over everything, and walk back up.
> 4. Finally, you walk down to the basement, scoop the salad into a bowl, and bring it up to the dining table.
> You walked up and down the stairs 100 times! Your legs are burning, and 95% of your cooking time was wasted walking the stairs instead of chopping vegetables.
> 
> Then, a master chef invents **FlashAttention**:
> *"Stop running to the basement! Bring a small basket of vegetables up to your cutting board once. Chop them, toss them in a small bowl right on your board, keep track of your dressing ratios in your head, and only walk to the dining room once when the salad is completely finished."*
> 
> FlashAttention does not change the recipe by a single grain of salt—it produces the exact same salad, but does it 4 times faster simply by never walking to the basement.

---

## Step 2: The Bridging Question

How do we convert this physical kitchen metaphor into GPU memory hierarchy numbers, and how can we compute mathematical $\operatorname{softmax}$ over an entire sequence without knowing all the numbers in advance?

In modern GPUs, memory is strictly hierarchical:
1. **On-Chip SRAM (Shared Memory & Registers)**: Extremely small (~50 MB on an NVIDIA H100), but blindingly fast (~19 TB/s bandwidth).
2. **Off-Chip HBM (High Bandwidth Memory / DRAM)**: Giant capacity (80 GB to 192 GB), but much slower (~3.35 TB/s bandwidth).

In standard attention, calculating $\mathbf{O} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$ requires writing the intermediate $T \times T$ attention matrix $\mathbf{S}$ and probability matrix $\mathbf{P}$ to slow HBM and reading them back multiple times:
$$\text{HBM Read/Write Volume} = O(T^2) \text{ Numbers}$$

When sequence length $T = 8192$, a $T \times T$ matrix contains 67 million numbers per attention head. The GPU spends nearly all its time moving this matrix back and forth across the memory bus.

The bridging question is:
$$\text{How can we compute the exact Softmax attention output block-by-block in tiny on-chip SRAM without ever materializing or writing the } T \times T \text{ matrix to HBM?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STANDARD ATTENTION VS. FLASHATTENTION IO                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STANDARD ATTENTION (3 HBM Roundtrips):                                                 │
│   Q, K (HBM) ──► Load to SRAM ──► S = Q @ K.T ──► Write S [T x T] to HBM               │
│   S (HBM)    ──► Load to SRAM ──► P = softmax(S) ──► Write P [T x T] to HBM            │
│   P, V (HBM) ──► Load to SRAM ──► O = P @ V ────► Write O [T x d] to HBM               │
│   TOTAL HBM TRAFFIC: O(T^2 + T * d) bytes                                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ FLASHATTENTION (1 HBM Pass with SRAM Tiling &amp; Online Softmax):                         │
│   Load small block Q_i [Br x d] into SRAM                                              │
│   For each block K_j, V_j [Bc x d]:                                                    │
│     Compute S_ij = Q_i @ K_j.T locally in SRAM                                         │
│     Update running Softmax scale (m, d) and incrementally accumulate O_i in SRAM       │
│   Write final O_i directly to HBM (NEVER write S or P to HBM!)                         │
│   TOTAL HBM TRAFFIC: O(T^2 * d^2 / M_SRAM) bytes (2x - 4x speedup!)                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 23.1:</strong> Standard Attention repeatedly writes intermediate T x T matrices to slow off-chip HBM. FlashAttention computes attention entirely within on-chip SRAM via block tiling.</figcaption>
</figure>

### 1. The Standard Softmax Bottleneck

For a raw attention score row $\mathbf{s} = [s_1, \dots, s_T] \in \mathbb{R}^{1 \times T}$, standard softmax requires two global passes:
1. Find global maximum: $m = \max_{j=1}^T s_j$
2. Compute global denominator: $d = \sum_{j=1}^T e^{s_j - m}$
3. Normalize: $p_j = \frac{e^{s_j - m}}{d}$

This requires holding **all $T$ scores in memory simultaneously** before any output can be computed.

---

### 2. The Online Softmax Recurrence Relations

Milakov &amp; Gimelshtein (2018) proved that Softmax can be computed incrementally across arbitrary streaming blocks.

Let a sequence of scores be split into two consecutive blocks: $\mathbf{s}^{(1)}$ and $\mathbf{s}^{(2)}$.
Suppose we have already processed block 1 and stored its local statistics:
- Local max: $m^{(1)} = \max_j s_j^{(1)}$
- Local denominator: $d^{(1)} = \sum_j e^{s_j^{(1)} - m^{(1)}}$

When block 2 arrives with its own local statistics $(m^{(2)}, d^{(2)})$, the unified global statistics $(m^{\text{new}}, d^{\text{new}})$ are updated via exact mathematical identities:

$$
m^{\text{new}} = \max\left(m^{(1)}, \; m^{(2)}\right)
$$

$$
d^{\text{new}} = d^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + d^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}
$$

Notice the correction factor $e^{m^{(1)} - m^{\text{new}}}$: it rescales the previous running sum to the newly discovered maximum without re-reading a single number from block 1!

---

### 3. FlashAttention Output Rescaling Equation

When accumulating the attention output $\mathbf{O} = \mathbf{P} \mathbf{V}$:

Let $\mathbf{o}^{(1)}$ be the accumulated unnormalized output vector from block 1. When block 2 ($K_2, V_2$) is processed in SRAM, the new accumulated output $\mathbf{o}^{\text{new}}$ is updated as:

$$
\mathbf{o}^{\text{new}} = \mathbf{o}^{(1)} \left(\frac{d^{(1)} e^{m^{(1)} - m^{\text{new}}}}{d^{\text{new}}}\right) + \left(\frac{e^{-m^{\text{new}}}}{d^{\text{new}}}\right) \left(e^{\mathbf{S}^{(2)}} \mathbf{V}^{(2)}\right)
$$

By maintaining only the scalar running statistics $(m, d)$ and the accumulated output vector $\mathbf{o} \in \mathbb{R}^{1 \times d_{\text{head}}}$ in SRAM:
- The $T \times T$ attention matrix is **never stored in HBM**.
- Memory footprint drops from $O(T^2)$ to $O(T)$ in backward/serving passes.
- High Bandwidth Memory access drops by a factor of $\frac{M_{\text{SRAM}}}{4 d}$, delivering a **$2\times$ to $4\times$ wall-clock speedup**.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2018">2018</time> &mdash; <strong>Online Normalizer Calculation for Softmax</strong> (<cite>Maxim Milakov &amp; Natalia Gimelshtein, NVIDIA</cite>)</dt>
  <dd>Demonstrated that Softmax normalization could be calculated in a single memory pass without storing intermediate exponentiations.</dd>
  <dt><time datetime="2022">2022</time> &mdash; <strong>FlashAttention</strong> (<cite>Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, Stanford University</cite>)</dt>
  <dd>Introduced IO-aware algorithm design to Transformers. By tiling query, key, and value matrices to fit within SRAM and recomputing activations in backward passes, FlashAttention eliminated the attention memory bottleneck.</dd>
  <dt><time datetime="2023">2023–2024</time> &mdash; <strong>FlashAttention-2 &amp; FlashAttention-3</strong> (<cite>Tri Dao et al.</cite>)</dt>
  <dd>Refactored loop orders to maximize Tensor Core warp occupancy (FA-2) and exploited FP8 Tensor Core asynchrony on NVIDIA Hopper H100 GPUs (FA-3), achieving up to 75% of theoretical peak GPU FLOPs.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us verify the exact numerical equivalence of Online Softmax on a 4-number sequence split into two blocks of size 2.

Let raw attention scores be:
$$
\mathbf{s} = [s_1, s_2, s_3, s_4] = [2.0, \; 4.0, \; 1.0, \; 5.0]
$$

---

### Method A: Standard Two-Pass Global Softmax
1. Global Max: $m = \max(2, 4, 1, 5) = 5.0$
2. Shifted values: $[2-5, 4-5, 1-5, 5-5] = [-3, -1, -4, 0]$
3. Exponentials:
   $$e^{-3} \approx 0.0498, \quad e^{-1} \approx 0.3679, \quad e^{-4} \approx 0.0183, \quad e^0 = 1.0000$$
4. Global Sum:
   $$d = 0.0498 + 0.3679 + 0.0183 + 1.0000 = 1.4360$$
5. Normalized Probabilities:
   $$\mathbf{p} = \left[\frac{0.0498}{1.4360}, \frac{0.3679}{1.4360}, \frac{0.0183}{1.4360}, \frac{1.0000}{1.4360}\right] \approx [0.0347, \; 0.2562, \; 0.0127, \; 0.6964]$$

---

### Method B: FlashAttention Online Softmax (Block Size = 2)

#### Block 1: $\mathbf{s}^{(1)} = [2.0, \; 4.0]$
1. Local max: $m^{(1)} = \max(2, 4) = 4.0$
2. Local sum:
   $$d^{(1)} = e^{2 - 4} + e^{4 - 4} = e^{-2} + e^0 \approx 0.1353 + 1.0000 = 1.1353$$

---

#### Block 2: $\mathbf{s}^{(2)} = [1.0, \; 5.0]$
1. Local max: $m^{(2)} = \max(1, 5) = 5.0$
2. Local sum:
   $$d^{(2)} = e^{1 - 5} + e^{5 - 5} = e^{-4} + e^0 \approx 0.0183 + 1.0000 = 1.0183$$

---

#### The Online Merge Step
Now update global statistics $(m^{\text{new}}, d^{\text{new}})$ without re-reading Block 1:

1. New Global Max:
   $$m^{\text{new}} = \max(m^{(1)}, m^{(2)}) = \max(4.0, 5.0) = 5.0$$
2. New Global Denominator:
   $$
   d^{\text{new}} = d^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + d^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}
   $$
   $$
   d^{\text{new}} = 1.1353 \cdot e^{4.0 - 5.0} + 1.0183 \cdot e^{5.0 - 5.0}
   $$
   $$
   d^{\text{new}} = 1.1353 \cdot e^{-1} + 1.0183 \cdot 1.0 = (1.1353 \times 0.36788) + 1.0183
   $$
   $$
   d^{\text{new}} = 0.4177 + 1.0183 = 1.4360
   $$

The merged denominator $d^{\text{new}} = 1.4360$ matches Method A's global sum **identically down to the 4th decimal place**!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p><strong>FlashAttention</strong> proves that the ultimate speed of deep learning algorithms on modern GPUs is governed not by operation count (FLOPs), but by <strong>IO complexity</strong> (memory traffic between SRAM and HBM).</p>
<p>By restructuring the Softmax formula into an online streaming recurrence and tiling matrices into SRAM blocks, FlashAttention computes mathematically identical attention while eliminating $O(T^2)$ memory reads and writes, unlocking $2\times$ to $4\times$ real-world speedups.</p>
</fieldset>
