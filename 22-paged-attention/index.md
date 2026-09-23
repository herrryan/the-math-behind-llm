# Chapter 22: PagedAttention & Virtual Memory (vLLM & Zero Fragmentation)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine you run a hotel with 500 rooms.
> 
> A travel guide walks into the lobby and shouts: *"My tour group just arrived! We might need 5 rooms, or we might need all 500 rooms if all our relatives show up. And all our rooms MUST be in a single unbroken straight line on the exact same floor!"*
> 
> Because you must give them an unbroken line of rooms for their maximum possible size, you lock all 500 rooms on floors 1 through 5 just for them. But in reality, only 10 tourists ever show up! 490 hotel rooms sit completely empty and locked. Outside your hotel, a huge crowd of other tired travelers is waiting in the rain, but you have to turn them away because the rooms are "reserved." That is traditional **Contiguous Memory Allocation**.
> 
> Then, a clever hotel manager invents a smart computer keycard: **PagedAttention**.
> The manager tells the guide: *"You don't need to lock 500 rooms upfront, and your rooms don't need to be side by side! Here is room 102. When your next friend arrives, I will give them room 307. When the next friend arrives, I will give them room 412. My computer knows exactly who belongs to your group, and your friends can visit each other instantly."*
> 
> Suddenly, not a single room is wasted. Every empty room can take a guest, and the hotel accommodates 4 times as many travelers on the exact same budget!

---

## Step 2: The Bridging Question

How do we convert this hotel manager's dynamic keycard system into tensor operations that allow GPU attention kernels to read non-contiguous blocks of memory with zero overhead?

In standard deep learning frameworks (like native PyTorch), a tensor must reside in a **single contiguous chunk of physical memory**. If a user's request might generate up to $T_{\text{max}} = 2048$ tokens, the serving engine is forced to pre-allocate an unbroken memory buffer of shape $[B, 2, n_{\text{layers}}, n_{\text{kv\_heads}}, T_{\text{max}}, d_{\text{head}}]$.

Because the actual generation length is unknown in advance, this creates two devastating forms of memory waste:
1. **Internal Fragmentation**: Memory reserved for future tokens that the model never ended up generating.
2. **External Fragmentation**: Memory gaps between requests that are too small to fit a new $T_{\text{max}}$ allocation.

In production, **60% to 80% of GPU memory was wasted on empty padding**. The bridging question is:
$$\text{How can an attention kernel calculate } \operatorname{softmax}\left(\frac{\mathbf{q} \mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \text{ when the vectors of } \mathbf{K} \text{ and } \mathbf{V} \text{ are scattered across random, non-contiguous physical DRAM blocks?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PAGEDATTENTION VIRTUAL MEMORY ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Request 1 (Prompt: 7 tokens, Block Size B = 4)                                         │
│ Logical KV Blocks:                                                                     │
│   Logical Block 0: [ Token 0, Token 1, Token 2, Token 3 ]  (Full)                      │
│   Logical Block 1: [ Token 4, Token 5, Token 6,   ___   ]  (Offset = 3)                │
│                                                                                        │
│ Page Table (Request 1):                                                                │
│   Logical Block 0 ──► Physical Block 7                                                 │
│   Logical Block 1 ──► Physical Block 2                                                 │
│                                                                                        │
│ Physical Memory Pool (DRAM Frames in GPU HBM):                                         │
│   [ Block 0 ]   [ Block 1 ]   [ Block 2: Req 1 (Tokens 4-6) ]   [ Block 3 ]            │
│   [ Block 4 ]   [ Block 5 ]   [ Block 6 ]                       [ Block 7: Req 1 (0-3)]│
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 22.1:</strong> PagedAttention maps contiguous logical token sequences to non-contiguous physical blocks in GPU DRAM, eliminating memory fragmentation.</figcaption>
</figure>

### 1. Logical vs. Physical Memory Partitioning

Let the key-value sequence of a request be partitioned into fixed-size chunks called **Logical Blocks** of size $B$ tokens (typically $B = 16$ or $B = 32$).

For any token at sequence position $t \in \{0, 1, \dots, T - 1\}$:
- **Logical Block Index**:
  $$
  b_{\text{logical}} = \left\lfloor \frac{t}{B} \right\rfloor
  $$
- **Block Offset**:
  $$
  o = t \bmod B
  $$

Physical GPU memory is pre-allocated once into a flat pool of **Physical Blocks** $\mathcal{P} = \{P_0, P_1, \dots, P_{N-1}\}$. Each physical block has shape:
$$
P_i \in \mathbb{R}^{B \times 2 \times n_{\text{kv\_heads}} \times d_{\text{head}}}
$$

### 2. The Page Table Mapping Function

For each active request $r$, the serving runtime maintains a dynamic array called the <dfn id="def-page-table">Page Table</dfn> $\mathcal{T}_r$:

$$
\mathcal{T}_r(b_{\text{logical}}) = P_{\text{physical}} \in \mathcal{P}
$$

The physical memory address $\text{Addr}(t)$ of the Key or Value vector for token $t$ is calculated via:

$$
\text{Addr}(t) = \text{BaseAddr}\left(\mathcal{T}_r\left(\left\lfloor \frac{t}{B} \right\rfloor\right)\right) + (t \bmod B) \times S_{\text{token}}
$$

where $S_{\text{token}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times p$ is the byte stride per token.

---

### 3. The PagedAttention Kernel Execution

In standard attention, $\mathbf{q}_t$ multiplies a single contiguous matrix $\mathbf{K}$.
In PagedAttention, the GPU kernel loops over the request's page table entries, fetching blocks dynamically from arbitrary memory locations without copying them into a temporary buffer:

$$
\mathbf{o}_t = \sum_{j=0}^{\lceil t / B \rceil - 1} \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{P_j}^\top}{\sqrt{d_{\text{head}}}}\right) \mathbf{V}_{P_j}
$$

where $P_j = \mathcal{T}_r(j)$ is the physical block holding logical tokens $j \cdot B$ through $\min((j+1)B - 1, t)$.

---

### 4. Mathematical Elimination of Memory Fragmentation

Let $T$ be the actual sequence length of a request, and $B$ be the block size.
- **External Fragmentation**:
  Because any free physical block in $\mathcal{P}$ can be assigned to any request regardless of physical location, **external fragmentation is exactly $0\%$**.
- **Internal Fragmentation**:
  Internal fragmentation occurs exclusively in the very last logical block of a sequence:
  $$\text{Wasted Slots} = (B - 1) - (T - 1) \bmod B \le B - 1$$
  The fractional memory waste $W_{\text{internal}}$ is bounded by:
  $$
  W_{\text{internal}} < \frac{B}{T}
  $$

For a typical serving configuration with block size $B = 16$ and context length $T = 2048$:
$$
W_{\text{internal}} < \frac{16}{2048} \approx 0.78\%
$$

PagedAttention reduces memory waste from **over 60% down to under 1%**!

---

### 5. Dynamic Memory Sharing via Copy-on-Write (CoW)

In advanced generation workflows (such as parallel sampling, beam search, or system prompt prefix caching), multiple requests share identical prompt prefixes.

PagedAttention enables zero-memory copying via **Copy-on-Write**:
- Each physical block maintains a reference counter $\text{ref\_count}(P_i) \in \mathbb{N}$.
- When two requests share the same prompt tokens, their page tables point to the **exact same physical block**:
  $$\mathcal{T}_{r_1}(0) = \mathcal{T}_{r_2}(0) = P_7 \implies \text{ref\_count}(P_7) = 2$$
- If Request 1 later appends a new token to this block, the engine checks $\text{ref\_count}$. If $\text{ref\_count} > 1$, it allocates a new physical block $P_{\text{new}}$, copies the $B$ tokens over, decrements $\text{ref\_count}(P_7)$, and updates Request 1's page table.

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 22.1:</strong> Comparison of Contiguous Memory Serving vs. PagedAttention.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Metric</th>
      <th align="left">Contiguous Pre-allocation</th>
      <th align="left">PagedAttention (vLLM)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Allocation Strategy</strong></td>
      <td>Over-allocates for worst-case $T_{\text{max}}$</td>
      <td>Dynamically allocates blocks of size $B$ on demand</td>
    </tr>
    <tr>
      <td><strong>Internal Fragmentation</strong></td>
      <td>60% – 80% (massive padding waste)</td>
      <td><strong>&lt; 1%</strong> (bounded by $(B-1)/T$)</td>
    </tr>
    <tr>
      <td><strong>External Fragmentation</strong></td>
      <td>Severe (gaps between requests)</td>
      <td><strong>0%</strong> (arbitrary non-contiguous blocks)</td>
    </tr>
    <tr>
      <td><strong>Prefix Sharing</strong></td>
      <td>Requires full duplicate copy in memory</td>
      <td><strong>Zero copy</strong> (shared page table pointers)</td>
    </tr>
    <tr>
      <td><strong>Serving Throughput</strong></td>
      <td>1x (baseline)</td>
      <td><strong>2x – 4x higher throughput</strong></td>
    </tr>
  </tbody>
</table>

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="1962">1962</time> &mdash; <strong>Virtual Memory &amp; Paging</strong> (<cite>Ferranti Atlas Computer, Manchester</cite>)</dt>
  <dd>Tom Kilburn and his team invented paged virtual memory, separating an application's contiguous logical address space from non-contiguous physical core memory.</dd>
  <dt><time datetime="2023">2023</time> &mdash; <strong>PagedAttention &amp; vLLM</strong> (<cite>Woosuk Kwon et al., UC Berkeley, SOSP 2023</cite>)</dt>
  <dd>Recognizing that LLM KV cache management was suffering from the exact same memory fragmentation that operating systems solved 60 years earlier, Woosuk Kwon and the UC Berkeley Sky Computing Lab designed PagedAttention and built the <strong>vLLM</strong> open-source inference system, revolutionizing LLM serving industry-wide.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us walk through the dynamic memory allocation of two requests step-by-step.

### System Configuration
- Block size: $B = 2\text{ tokens}$
- Total Physical Memory Pool: 4 blocks $\{P_0, P_1, P_2, P_3\}$
- Each block stores 2 tokens of Key and Value tensors.

---

### Step 1: Request 1 Arrives with a 3-Token Prompt
Request 1 enters with prompt: `"the dog barked"` ($T = 3$).
- Number of required blocks: $\lceil 3 / 2 \rceil = 2\text{ blocks}$.
- Runtime allocates free blocks $P_1$ and $P_3$ from the pool.

The Page Table for Request 1 is initialized:
$$
\mathcal{T}_1 = [P_1, \; P_3]
$$

Physical storage breakdown:
- **$P_1$ (Logical Block 0)**: Stores token 0 (`"the"`) and token 1 (`"dog"`). **Full** (2/2 slots).
- **$P_3$ (Logical Block 1)**: Stores token 2 (`"barked"`). **Partial** (1/2 slots). 1 slot is free.

$$\text{Internal Fragmentation} = \frac{1\text{ empty slot}}{4\text{ allocated slots}} = 25\%$$

---

### Step 2: Request 1 Generates Token 3 (`"loudly"`)
The model generates token 3.
- Sequence position: $t = 3$.
- Logical block: $\lfloor 3 / 2 \rfloor = 1$.
- Offset: $3 \bmod 2 = 1$.

Logical block 1 is already mapped to $P_3$, and slot 1 is free!
The kernel writes token 3 directly into $P_3$ at offset 1 without allocating any new memory.
- $P_3$ is now completely full (2/2 slots).
$$\text{Internal Fragmentation} = 0\%$$

---

### Step 3: Request 1 Generates Token 4 (`"."`)
The model generates token 4.
- Sequence position: $t = 4$.
- Logical block: $\lfloor 4 / 2 \rfloor = 2$.

Logical block 2 does not exist in $\mathcal{T}_1$ yet!
The manager fetches the next free block from the pool ($P_0$) and appends it to Request 1's page table:
$$
\mathcal{T}_1 = [P_1, \; P_3, \; P_0]
$$
Token 4 is written into $P_0$ at offset 0.
The KV tensors are scattered across physical blocks $P_1 \to P_3 \to P_0$, yet the model attends across them seamlessly!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p><strong>PagedAttention</strong> solves the LLM memory crisis by borrowing the greatest triumph of operating system design: <strong>Virtual Memory Paging</strong>.</p>
<p>By mapping contiguous logical token sequences to non-contiguous physical DRAM blocks via page tables, PagedAttention slashes memory waste from 80% to under 1%, doubling to quadrupling serving capacity on identical hardware.</p>
</fieldset>
