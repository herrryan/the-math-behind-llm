# Chapter 28: Distributed Serving (Tensor Parallelism & Prefill-Decode Disaggregation)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine a restaurant kitchen with two completely different jobs:
> 
> 1. **The Heavy Chopping Station (The Prefill Chef)**:
>    A massive chef with giant cleavers. Whenever an order arrives, this chef chops 50 pounds of carrots and potatoes in one huge explosive burst of energy.
> 2. **The Teacup Serving Table (The Decode Waiter)**:
>    A delicate waiter holding a silver tray. Every 2 seconds, the waiter pours exactly one tiny teaspoon of hot tea into a customer's cup.
> 
> In traditional **Monolithic Serving**:
> You force the same person to chop 50 pounds of potatoes AND pour delicate teaspoons of tea at the exact same table! What happens? The cleaver shakes the table, tea spills everywhere, customers wait 10 minutes for their next sip of tea while the chef chops potatoes, and the chef gets distracted and drops their knife.
> 
> Then, a smart restaurateur invents **Disaggregated Serving**:
> - Move the heavy chopping chef into the back kitchen (**Dedicated Prefill Cluster**).
> - Move the delicate tea waiter into the quiet dining room (**Dedicated Decode Cluster**).
> - When the chopped vegetables are ready, a speedy conveyor belt (**Ultra-Fast RDMA Network**) whisks the ingredients from the kitchen to the dining room in a split second.
> 
> And what if a giant recipe is too big for any single chef's cutting board?
> You use **Tensor Parallelism (Slicing the Puzzle)**: Chef A chops the left half of the recipe, Chef B chops the right half of the recipe, and they clap their hands together to combine the dish!

---

## Step 2: The Bridging Question

How do we split giant weight matrices across multiple GPUs with minimal communication overhead, and how do we physically decouple compute-bound Prefill nodes from memory-bound Decode nodes across an enterprise cluster?

When serving frontier models like Llama-3-405B or DeepSeek-V3 (671B):
- The model weights alone require over **800 GB to 1.3 TB** of memory.
- No single GPU on Earth has enough HBM to hold the model.

Furthermore, as established in Chapter 20, Prefill and Decode have fundamentally antagonistic hardware requirements:
- **Prefill**: High Arithmetic Intensity ($I \gg I^*$), compute-bound, saturates Tensor Cores, benefits from high GPU count and massive matrix tile parallelization.
- **Decode**: Low Arithmetic Intensity ($I \ll I^*$), memory-bandwidth bound, starves Tensor Cores, demands massive HBM bandwidth and low inter-GPU latency.

Co-locating both phases on the same GPUs forces Decode requests to stall whenever a new Prefill request arrives, creating unacceptable latency jitter (tail TPOT spikes).

The bridging question is:
$$\text{How does Megatron Tensor Parallelism partition linear algebra operators with only two All-Reduce communications per layer, and how does Prefill-Decode Disaggregation physically separate serving clusters via high-speed RDMA KV cache transfer?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MEGATRON TENSOR PARALLELISM (MLP LAYER)                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Input Activation X [B x d_model] broadcast to both GPUs:                               │
│                                                                                        │
│   GPU 0 (Left Column / Top Row):                                                       │
│     h_1 = GeLU( X @ W_1,1 )  ──►  y_1 = h_1 @ W_2,1                                    │
│                                          │                                             │
│   GPU 1 (Right Column / Bottom Row):     │                                             │
│     h_2 = GeLU( X @ W_1,2 )  ──►  y_2 = h_2 @ W_2,2                                    │
│                                          │                                             │
│   NVLink All-Reduce (Sum):               ▼                                             │
│     Y = y_1 + y_2  (Mathematically identical to X @ W_1 @ W_2 with zero sync inside!)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 28.1:</strong> Megatron-LM conjugate Column-Row factorization. The intermediate activation is kept local on each GPU; only a single All-Reduce is required at the end of the block.</figcaption>
</figure>

### 1. Megatron Column-Row Parallel Factorization

Consider a standard two-layer Feed-Forward Network:
$$
\mathbf{Y} = \operatorname{GeLU}(\mathbf{X} \mathbf{W}_1) \mathbf{W}_2
$$
where $\mathbf{X} \in \mathbb{R}^{B \times d}$, $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, and $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$.

To split this computation across $N$ GPUs:

#### First Layer: Column Parallel GEMM
We slice $\mathbf{W}_1$ vertically into $N$ column blocks:
$$
\mathbf{W}_1 = \begin{bmatrix} \mathbf{W}_{1}^{(1)} & \mathbf{W}_{1}^{(2)} & \cdots & \mathbf{W}_{1}^{(N)} \end{bmatrix}, \quad \mathbf{W}_{1}^{(i)} \in \mathbb{R}^{d \times \frac{4d}{N}}
$$
Each GPU $i$ independently computes its local intermediate hidden representation:
$$
\mathbf{H}^{(i)} = \operatorname{GeLU}\left(\mathbf{X} \mathbf{W}_{1}^{(i)}\right) \in \mathbb{R}^{B \times \frac{4d}{N}}
$$
<mark>Key Architectural Insight:</mark> Because the GeLU activation function is strictly element-wise:
$$
\operatorname{GeLU}\left(\begin{bmatrix} \mathbf{A} & \mathbf{B} \end{bmatrix}\right) = \begin{bmatrix} \operatorname{GeLU}(\mathbf{A}) & \operatorname{GeLU}(\mathbf{B}) \end{bmatrix}
$$
**Zero inter-GPU communication is required after the first layer!** Each GPU holds its slice of the hidden state in local registers.

---

#### Second Layer: Row Parallel GEMM
We slice $\mathbf{W}_2$ horizontally into $N$ row blocks:
$$
\mathbf{W}_2 = \begin{bmatrix} \mathbf{W}_{2}^{(1)} \\ \mathbf{W}_{2}^{(2)} \\ \vdots \\ \mathbf{W}_{2}^{(N)} \end{bmatrix}, \quad \mathbf{W}_{2}^{(i)} \in \mathbb{R}^{\frac{4d}{N} \times d}
$$
Each GPU $i$ multiplies its local hidden state by its local row weight:
$$
\mathbf{Y}^{(i)} = \mathbf{H}^{(i)} \mathbf{W}_{2}^{(i)} \in \mathbb{R}^{B \times d}
$$

The mathematical result of the full block is the sum across all GPUs:
$$
\mathbf{Y} = \sum_{i=1}^N \mathbf{Y}^{(i)} = \sum_{i=1}^N \operatorname{GeLU}\left(\mathbf{X} \mathbf{W}_{1}^{(i)}\right) \mathbf{W}_{2}^{(i)} = \operatorname{GeLU}(\mathbf{X} \mathbf{W}_1) \mathbf{W}_2
$$

The GPUs perform a single hardware <dfn id="def-allreduce">All-Reduce (Sum)</dfn> collective across high-speed NVLink.

---

### 2. Multi-Head Attention Factorization &amp; Communication Cost

In the Self-Attention block:
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ are sliced **Column-wise** (splitting attention heads across GPUs).
- $\mathbf{W}_O$ is sliced **Row-wise**.
- One single All-Reduce combines the attention output before the residual addition.

Across an entire Transformer layer (Attention + MLP), exactly **2 All-Reduce operations** are required.
For a ring All-Reduce across $N$ GPUs with tensor shape $[B, T, d]$ and precision $p$ bytes:

$$
\text{Communication Volume} = 2 \times \left(2 \cdot \frac{N - 1}{N} \cdot B \cdot T \cdot d \cdot p\right) \text{ bytes/layer}
$$

Because of this frequent synchronization, Tensor Parallelism is strictly restricted to GPUs connected by ultra-high-speed NVLink ($900\text{ GB/s}$ on H100), typically scaling up to $N = 8$.

---

### 3. Prefill-Decode Disaggregation (PD Separation)

In modern web-scale architectures (DistServe, Mooncake, Splitwise), servers are physically partitioned into two distinct pools:

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PREFILL-DECODE DISAGGREGATED SERVING ARCHITECTURE               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ USER REQUEST ──► [ Global Router / Gateway ]                                          │
│                         │                                                              │
│                         ▼                                                              │
│ ┌──────────────────────────────────────────┐                                           │
│ │ PREFILL CLUSTER (Compute-Dense Engine):   │                                           │
│ │   - Optimized for Compute (GEMM)         │                                           │
│ │   - Large Tensor Parallelism (TP=4 or 8) │                                           │
│ │   - High batch sizes, full GPU saturation│                                           │
│ └──────────────────────────────────────────┘                                           │
│                         │                                                              │
│                         ▼  Ultra-High-Speed RDMA Network (RoCE v2 / InfiniBand)        │
│    KV CACHE TRANSFER ──► [ Transfer Layer: K, V tensors sent to decode node via RDMA ] │
│                         │                                                              │
│                         ▼                                                              │
│ ┌──────────────────────────────────────────┐                                           │
│ │ DECODE CLUSTER (Memory-Bandwidth Engine):│                                           │
│ │   - Optimized for Memory Bandwidth (GEMV)│                                           │
│ │   - Low Tensor Parallelism (TP=1 or 2)   │                                           │
│ │   - High token throughput, zero jitter   │                                           │
│ └──────────────────────────────────────────┘                                           │
│                         │                                                              │
│                         ▼                                                              │
│                STREAMED OUTPUT TOKENS                                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 28.2:</strong> Disaggregated serving isolates compute-bound Prefill nodes from memory-bandwidth-bound Decode nodes, transferring intermediate KV caches over RDMA.</figcaption>
</figure>

Let $T_{\text{ctx}}$ be the context length, $L$ layers, $H_{\text{kv}}$ heads, and $d$ head dimension.
The KV Cache transferred over the network is:
$$
\text{Size}_{\text{KV}} = 2 \times L \times H_{\text{kv}} \times d \times T_{\text{ctx}} \times p \text{ bytes}
$$

Transfer latency over an RDMA network with bandwidth $B_{\text{RDMA}}$ (e.g., $400 \text{ Gbps} = 50 \text{ GB/s}$) is:

$$
t_{\text{transfer}} = \frac{\text{Size}_{\text{KV}}}{B_{\text{RDMA}}}
$$

For a 4,000-token prompt on a 70B model with GQA ($H_{\text{kv}} = 8, d=128, L=80, p=2$), $\text{Size}_{\text{KV}} \approx 655 \text{ MB}$.
$$
t_{\text{transfer}} = \frac{0.655 \text{ GB}}{50 \text{ GB/s}} \approx 13 \text{ milliseconds}
$$

By paying a tiny $13\text{ ms}$ RDMA transfer tax, Decode nodes run with **zero interference**, stabilizing tail latency ($P_{99}$ TPOT) by over **$10\times$**!

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2019">2019</time> &mdash; <strong>Megatron-LM</strong> (<cite>Mohammad Shoeybi et al., NVIDIA</cite>)</dt>
  <dd>Pioneered Column-Row conjugate tensor parallelism, eliminating intermediate synchronizations and making it possible to train and serve multi-billion parameter models.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>DistServe &amp; Splitwise</strong> (<cite>Hao Zhong et al., OSDI 2024; Pratyush Patel et al., ISCA 2024</cite>)</dt>
  <dd>Demonstrated that coupling prefill and decode on the same hardware violates fundamental queuing theory, proposing physical cluster disaggregation.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>Mooncake (Kimi)</strong> (<cite>Qin et al., Moonshot AI</cite>)</dt>
  <dd>Deployed disaggregated serving at massive consumer scale, using a shared Kimi KV cache pool and chunked RDMA transport to power long-context reasoning.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace Megatron Column-Row Tensor Parallelism on 2 GPUs by hand.

### Inputs &amp; Weights
Let input activation be a single token ($B=1$, $d=2$):
$$
\mathbf{x} = [1.0, \; 2.0]
$$

Let the two MLP weight matrices be ($d=2 \to 4d=2 \to d=2$):
$$
\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$
(For simplicity of hand calculation, let activation be Identity: $\sigma(z) = z$).

---

### Single-GPU Ground Truth
$$
\mathbf{h} = \mathbf{x} \mathbf{W}_1 = [1, 2] \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = [1(1) + 2(3), \; 1(2) + 2(4)] = [7, \; 10]
$$
$$
\mathbf{y} = \mathbf{h} \mathbf{W}_2 = [7, 10] \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = [7(5) + 10(7), \; 7(6) + 10(8)] = [35 + 70, \; 42 + 80] = [\mathbf{105}, \; \mathbf{122}]
$$

---

### 2-GPU Tensor Parallel Execution

#### Step 1: Column Partitioning of $W_1$
- GPU 0 receives Column 1: $\mathbf{W}_{1,1} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$
- GPU 1 receives Column 2: $\mathbf{W}_{1,2} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

Each GPU multiplies $\mathbf{x} = [1, 2]$ locally:
- **GPU 0**: $h_1 = [1, 2] \begin{bmatrix} 1 \\ 3 \end{bmatrix} = 1(1) + 2(3) = 7$
- **GPU 1**: $h_2 = [1, 2] \begin{bmatrix} 2 \\ 4 \end{bmatrix} = 1(2) + 2(4) = 10$

Notice that $[h_1, h_2] = [7, 10]$. Zero communication occurred!

---

#### Step 2: Row Partitioning of $W_2$
- GPU 0 receives Row 1: $\mathbf{W}_{2,1} = \begin{bmatrix} 5 & 6 \end{bmatrix}$
- GPU 1 receives Row 2: $\mathbf{W}_{2,2} = \begin{bmatrix} 7 & 8 \end{bmatrix}$

Each GPU multiplies its scalar $h_i$ by its row vector:
- **GPU 0**: $\mathbf{y}_1 = 7 \times [5, 6] = [35, \; 42]$
- **GPU 1**: $\mathbf{y}_2 = 10 \times [7, 8] = [70, \; 80]$

---

#### Step 3: All-Reduce (Sum)
The two GPUs perform an All-Reduce addition across NVLink:
$$
\mathbf{y} = \mathbf{y}_1 + \mathbf{y}_2 = [35, 42] + [70, 80] = [35 + 70, \; 42 + 80] = [\mathbf{105}, \; \mathbf{122}]
$$

The result matches single-GPU ground truth **with mathematical perfection**, using only a single summation collective!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p><strong>Tensor Parallelism</strong> distributes giant matrices across NVLink-connected GPUs using conjugate Column-Row factorization, minimizing collective communications to just two All-Reduce calls per layer.</p>
<p>At the cluster level, <strong>Prefill-Decode Disaggregation</strong> solves the fundamental tension of serving by dedicating compute-optimized nodes to bursty prefill requests and memory-optimized nodes to steady decode streams, linked by ultra-fast RDMA KV cache migration.</p>
</fieldset>
