# Chapter 21: The Memory Wall (KV Cache Mechanics & GQA / MLA Compression)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you are a schoolchild writing an essay on a chalkboard.
> 
> If you have no memory scratchpad, every time you want to write word number 50, you must stop, walk to the library, re-read words 1 through 49 from the beginning, figure out what they mean, and then write word 50. When you want to write word 51, you must walk back to the library and re-read words 1 through 50 all over again. By page 10, your legs are exhausted and the school bell rings before you finish a single sentence.
> 
> To save your legs, your teacher gives you a small pocket notebook: the **KV Cache**.
> Whenever you write a word, you jot down its clue badge (Key) and its story value (Value) into your notebook. When writing word 51, you only look at your own notebook right on your desk. You never walk back to the library.
> 
> But there is a catch: what happens when your essay becomes 100 pages long?
> Your little desk becomes piled high with thousands of notebooks. Soon, the notebooks fill the entire classroom from floor to ceiling. You can't even squeeze into your chair! That physical ceiling is **The Memory Wall**.

---

## Step 2: The Bridging Question

How do we convert this pocket notebook into exact tensor operations, calculate the precise number of gigabytes consumed in GPU memory (<abbr title="High Bandwidth Memory">HBM</abbr>), and formulate mathematical compressions that shrink the notebook by $8\times$ without losing memory of the story?

In transformer autoregressive generation, the hidden representation of previous tokens $\mathbf{x}_1, \dots, \mathbf{x}_{t-1}$ never changes once they are generated. If we recompute their Key and Value projections at every single step, generating $T$ tokens requires $O(T^2)$ computation per step, resulting in a disastrous $O(T^3)$ total cost over the entire sequence.

The bridging question is:
$$\text{How do we store previous Keys and Values so that generating token } t \text{ costs only } O(t) \text{ operations, and what is the exact memory cost of doing so?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AUTOREGRESSIVE GENERATION WITH KV CACHING                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Step t:                                                                                │
│ Single New Token:  x_t ──► [ W_q ] ──► q_t  [1 x d_head]                               │
│                    x_t ──► [ W_k ] ──► k_t  [1 x d_head] ──┐                           │
│                    x_t ──► [ W_v ] ──► v_t  [1 x d_head] ──┼──┐                        │
│                                                            │  │                        │
│ Past KV Cache:                                             ▼  │                        │
│   K_{1:t-1} [(t-1) x d_head] ─────────────► Concatenate ──► K_{1:t} [t x d_head]       │
│                                                               │                        │
│   V_{1:t-1} [(t-1) x d_head] ─────────────► Concatenate ──► V_{1:t} [t x d_head] ◄─┘  │
│                                                               │                        │
│ Attention Output:                                             ▼                        │
│   o_t = softmax( (q_t @ K_{1:t}.T) / sqrt(d_head) ) @ V_{1:t}  [1 x d_head]           │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 21.1:</strong> The iterative KV Cache update. At step t, only the newest query, key, and value vectors are computed; keys and values are appended to the historical cache.</figcaption>
</figure>

### 1. The Naive (No-Cache) Disaster

In standard self-attention, for a sequence of length $t$:
$$
\mathbf{Q}_t = \mathbf{X}_{1:t} \mathbf{W}_Q, \quad \mathbf{K}_t = \mathbf{X}_{1:t} \mathbf{W}_K, \quad \mathbf{V}_t = \mathbf{X}_{1:t} \mathbf{W}_V
$$
$$\mathbf{O}_t = \operatorname{softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}}\right) \mathbf{V}_t$$

At step $t$, recomputing all past tokens requires $O(t)$ matrix multiplications. Across a sequence of length $T$:
$$
\text{Total Operations}_{\text{no-cache}} = \sum_{t=1}^T 2 P \cdot t \propto O(T^2 \cdot d_{\text{model}})
$$
Including attention quadratic costs, generation scales as $O(T^3)$. This renders long-context generation computationally impossible.

### 2. The KV Cache Recurrence

Because previous tokens $\mathbf{x}_1, \dots, \mathbf{x}_{t-1}$ are static, their key and value representations are invariant over time:
$$
\mathbf{k}_\tau = \mathbf{x}_\tau \mathbf{W}_K, \quad \mathbf{v}_\tau = \mathbf{x}_\tau \mathbf{W}_V \quad (\forall \tau < t)
$$

At generation step $t$, we compute **only** the single-row projections for the latest token $\mathbf{x}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$:

$$
\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_{\text{head}}}
$$
$$
\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times d_{\text{head}}}
$$
$$
\mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times d_{\text{head}}}
$$

We append $\mathbf{k}_t$ and $\mathbf{v}_t$ to the historical key and value tensors:

$$
\mathbf{K}_{1:t} = \begin{bmatrix} \mathbf{K}_{1:t-1} \\ \mathbf{k}_t \end{bmatrix} \in \mathbb{R}^{t \times d_{\text{head}}}, \quad \mathbf{V}_{1:t} = \begin{bmatrix} \mathbf{V}_{1:t-1} \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{t \times d_{\text{head}}}
$$

The single output token representation is computed via vector-matrix attention:

$$
\mathbf{o}_t = \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{1:t}^\top}{\sqrt{d_{\text{head}}}}\right) \mathbf{V}_{1:t} \in \mathbb{R}^{1 \times d_{\text{head}}}
$$

### 3. The Exact KV Cache Memory Equation

Let:
- $n_{\text{layers}}$: Total number of transformer layers in the model.
- $n_{\text{kv\_heads}}$: Number of key/value attention heads per layer.
- $d_{\text{head}}$: Dimension of each attention head.
- $T_{\text{ctx}}$: Total sequence context length (prompt tokens + generated tokens).
- $b$: Serving batch size (number of concurrent user requests).
- $p$: Precision in bytes per floating point number (e.g., $p = 2$ for FP16/BF16, $p = 1$ for FP8, $p = 0.5$ for INT4).

The total physical memory $\text{RAM}_{\text{KV}}$ occupied by the KV cache is:

$$
\text{RAM}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{ctx}} \times b \times p \quad [\text{Bytes}]
$$

The leading factor of $2$ accounts for the two distinct stored tensors: Keys and Values.

---

### 4. The Memory Wall: The 128k Long-Context Crisis

Consider the industry-standard open-weight model: **LLaMA-3-70B**.
- Layers: $n_{\text{layers}} = 80$
- Hidden dimension: $d_{\text{model}} = 8192$
- Query heads: $H_q = 64$
- Head dimension: $d_{\text{head}} = 128$
- Precision: 16-bit ($p = 2\text{ bytes}$)
- Context length: $T_{\text{ctx}} = 128{,}000\text{ tokens}$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 21.1:</strong> Single-user ($b=1$) KV Cache memory requirements at 128k context across attention architectures (LLaMA-3-70B scale).</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Architecture</th>
      <th align="center">Query Heads ($H_q$)</th>
      <th align="center">KV Heads ($H_{\text{kv}}$)</th>
      <th align="center">KV Compression Ratio</th>
      <th align="right">KV Cache RAM per User ($T=128\text{k}$)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Multi-Head Attention (MHA)</strong></td>
      <td align="center">64</td>
      <td align="center">64</td>
      <td align="center">$1\times$ (Baseline)</td>
      <td align="right"><strong>312.50 GB</strong> (Catastrophic!)</td>
    </tr>
    <tr>
      <td><strong>Grouped-Query Attention (GQA)</strong></td>
      <td align="center">64</td>
      <td align="center">8</td>
      <td align="center"><strong>$8\times$ Reduction</strong></td>
      <td align="right"><strong>39.06 GB</strong> (Fits in single GPU)</td>
    </tr>
    <tr>
      <td><strong>Multi-Query Attention (MQA)</strong></td>
      <td align="center">64</td>
      <td align="center">1</td>
      <td align="center"><strong>$64\times$ Reduction</strong></td>
      <td align="right"><strong>4.88 GB</strong></td>
    </tr>
    <tr>
      <td><strong>Multi-Head Latent Attention (MLA)</strong></td>
      <td align="center">128</td>
      <td align="center">Latent $d_c=512$</td>
      <td align="center"><strong>Low-Rank Latent</strong></td>
      <td align="right"><strong>~11.20 GB</strong> (DeepSeek-V2/V3)</td>
    </tr>
  </tbody>
</table>

Under traditional Multi-Head Attention, a single user requesting 128k context would require **312.5 GB of RAM solely for the KV cache**—more than three full 80GB NVIDIA A100 GPUs just to store the memory of a single conversation!

---

### 5. Architectural Compression: MQA, GQA, and DeepSeek MLA

To prevent GPU memory exhaustion, researchers invented three generations of compression architectures:

#### A. Grouped-Query Attention (GQA)
Instead of allocating a private Key and Value head to every Query head ($H_{\text{kv}} = H_q$), GQA clusters $H_q$ query heads into $H_{\text{kv}}$ groups of size $G = H_q / H_{\text{kv}}$:

$$
\text{kv\_idx}(h) = \left\lfloor \frac{h}{G} \right\rfloor
$$

This reduces cache memory strictly by a factor of $G$ ($8\times$ in LLaMA-3) while maintaining full multi-head expressive capacity in query representations.

#### B. DeepSeek Multi-Head Latent Attention (MLA)
In DeepSeek-V2 and DeepSeek-V3, researchers eliminated independent multi-head KV caching entirely by projecting hidden state $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ into a shared, low-rank compressed latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ (where $d_c \ll H_q \cdot d_{\text{head}}$):

$$
\mathbf{c}_t^{KV} = \mathbf{x}_t \mathbf{W}_{DKV} \in \mathbb{R}^{1 \times d_c}
$$

During generation, the serving engine caches **only** the low-dimensional vector $\mathbf{c}_t^{KV}$ (along with a decoupled 64-dimensional RoPE key vector $\mathbf{k}_t^R$). The full multi-head keys and values are mathematically absorbed into the query projection via matrix associativity:

$$
\mathbf{q}_{t, h} \mathbf{k}_{\tau, h}^\top = \mathbf{q}_{t, h} (\mathbf{c}_\tau^{KV} \mathbf{W}_{UK, h})^\top = (\mathbf{q}_{t, h} \mathbf{W}_{UK, h}^\top) \mathbf{c}_\tau^{KV \top}
$$

By caching only the latent vector $\mathbf{c}_t^{KV}$, DeepSeek MLA compresses KV Cache memory consumption to a tiny fraction of MHA without any loss in model capacity.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2019">2019</time> &mdash; <strong>Multi-Query Attention (MQA)</strong> (<cite>Noam Shazeer, Google</cite>)</dt>
  <dd>Legendary Transformer co-inventor Noam Shazeer published <em>"Fast Transformer Decoding: One Write-Head is All You Need"</em>, proving that all query heads could share a single Key and Value head ($H_{\text{kv}} = 1$), slashing decode memory bandwidth by 95%.</dd>
  <dt><time datetime="2023">2023</time> &mdash; <strong>Grouped-Query Attention (GQA)</strong> (<cite>Ainslie et al., Google Research</cite>)</dt>
  <dd>Introduced as the golden trade-off between MHA (high quality, high memory) and MQA (low memory, minor quality degradation). GQA was rapidly adopted as the universal standard in LLaMA-2-70B, LLaMA-3, Mistral, and Gemma 2.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>Multi-Head Latent Attention (MLA)</strong> (<cite>DeepSeek-AI</cite>)</dt>
  <dd>DeepSeek introduced low-rank KV compression in DeepSeek-V2 and DeepSeek-V3, decoupling cache storage from head count and enabling 128k context serving at unprecedented low memory costs.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace step-by-step KV cache accumulation and memory consumption with tiny matrices.

### The Toy Model Specifications
- Layers: $n_{\text{layers}} = 1$
- Query Heads: $H_q = 2$
- Key/Value Heads: $H_{\text{kv}} = 1$ (Group size $G = 2$, MQA/GQA)
- Head dimension: $d_{\text{head}} = 2$
- Precision: 16-bit ($p = 2\text{ bytes}$)

---

### Step 1: Ingesting Prompt Token 1 ($t = 1$)
Suppose token 1 produces key and value vectors:
$$
\mathbf{k}_1 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 0.5 & 1.5 \end{bmatrix}
$$

We initialize the cache:
$$
\mathbf{K}_{1:1} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}, \quad \mathbf{V}_{1:1} = \begin{bmatrix} 0.5 & 1.5 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$
$$\text{Cache Elements} = 1 \text{ layer} \times 1 \text{ head} \times 2 \text{ dims} \times 1 \text{ token} \times 2 (\text{K and V}) = 4 \text{ numbers}$$
$$\text{Memory} = 4 \times 2 \text{ bytes} = 8\text{ Bytes}$$

---

### Step 2: Generating Token 2 ($t = 2$)
Token 2 arrives with its own projections:
$$
\mathbf{k}_2 = \begin{bmatrix} 3.0 & 0.0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2.0 & 1.0 \end{bmatrix}
$$
We append $\mathbf{k}_2$ and $\mathbf{v}_2$ to the cache:
$$
\mathbf{K}_{1:2} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}, \quad \mathbf{V}_{1:2} = \begin{bmatrix} 0.5 & 1.5 \\ 2.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$
$$\text{Memory} = (2 \text{ tokens} \times 2 \text{ dims} \times 2 \text{ tensors}) \times 2 \text{ bytes} = 16\text{ Bytes}$$

Now suppose Query Head 1 has query vector:
$$
\mathbf{q}_{2, 1} = \begin{bmatrix} 1.0 & 0.0 \end{bmatrix}
$$

We compute the attention scores across the cache:
$$
\mathbf{s} = \frac{\mathbf{q}_{2, 1} \mathbf{K}_{1:2}^\top}{\sqrt{2}} = \frac{\begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1.0 & 3.0 \\ 2.0 & 0.0 \end{bmatrix}}{\sqrt{2}} = \frac{\begin{bmatrix} 1.0 & 3.0 \end{bmatrix}}{\sqrt{2}} \approx \begin{bmatrix} 0.707 & 2.121 \end{bmatrix}
$$

We apply softmax:
$$
e^{0.707} \approx 2.028, \quad e^{2.121} \approx 8.339 \implies \sum = 10.367
$$
$$
\mathbf{a} = \operatorname{softmax}(\mathbf{s}) \approx \begin{bmatrix} \frac{2.028}{10.367} & \frac{8.339}{10.367} \end{bmatrix} \approx \begin{bmatrix} 0.196 & 0.804 \end{bmatrix}
$$

We aggregate over the cached values $\mathbf{V}_{1:2}$:
$$
\mathbf{o}_{2, 1} = \mathbf{a} \mathbf{V}_{1:2} = \begin{bmatrix} 0.196 & 0.804 \end{bmatrix} \begin{bmatrix} 0.5 & 1.5 \\ 2.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.098 + 1.608 & 0.294 + 0.804 \end{bmatrix} = \begin{bmatrix} 1.706 & 1.098 \end{bmatrix}
$$

Notice the magic: **We never recomputed token 1's key or value!** We retrieved them directly from the cache with zero wasted operations.

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>The <strong>KV Cache</strong> trades memory capacity for computational speed, slashing autoregressive generation cost from $O(T^2)$ to $O(T)$ operations per step.</p>
<p>However, because the cache scales linearly with context length $T_{\text{ctx}}$, long-context serving hits a catastrophic <strong>Memory Wall</strong>. Modern architectures overcome this wall through <strong>Grouped-Query Attention (GQA)</strong> and <strong>Multi-Head Latent Attention (MLA)</strong>, compressing the memory footprint by $8\times$ to $20\times$ without compromising model intelligence.</p>
</fieldset>
