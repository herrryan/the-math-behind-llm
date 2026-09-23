# Hands-on Lab 05: The Streaming KV & Continuous Batching Engine (vLLM Mini-Kernel in 200 Lines of Pure Python)

<fieldset id="evolution">
<legend><strong>The Python Brain Evolution Chain &bull; Inference Frontier (Stage 5 of 6)</strong></legend>
<p>In Lab 04, we assembled the complete interactive LLM with KV cache acceleration and nucleus sampling. However, that engine operated on a single user request at a time using contiguous linear memory arrays. In real-world production deployments, thousands of concurrent users send requests of drastically varying lengths at random arrival times.</p>
<p>In this fifth hands-on lab, we build <strong>The Streaming KV &amp; Continuous Batching Engine</strong> in ~200 lines of pure standard-library Python. Zero external dependencies: no PyTorch, no vLLM, no NumPy. We implement the two landmark engineering systems that power modern high-throughput serving: <strong>PagedAttention Virtual Memory Management</strong> (Chapters 21 &amp; 22) and <strong>Continuous Dynamic Batching</strong> (Chapter 25).</p>
<pre>
[The Python Brain Evolution Roadmap]
[Stage 1]  80 Lines Pure Python: Bengio 2003 MLP Language Model (Embeddings, Dense Layers, Manual Backprop)
       │
       ▼ (Amnesia flaw: 1-word context window)
[Stage 2] 140 Lines Pure Python: Attention Brain (Unlocking Q, K, V Projections &amp; Causal Attention)
       │
       ▼ (Instability flaw: vanishing gradients in deep networks)
[Stage 3] 220 Lines Pure Python: Modern Transformer Block (Pre-RMSNorm, Residual Highways, &amp; SwiGLU Gating)
       │
       ▼ (Sampling flaw: rigid greedy loops and O(T^2) redundant computation)
[Stage 4] 300 Lines Pure Python: Interactive LLM Engine (KV Cache &amp; Nucleus Sampling Suite)
       │
       ▼ (Multi-Tenant flaw: Static batching wastes 60%+ compute in bubbles; contiguous arrays fragment memory)
[Stage 5 (Current)] 200 Lines Pure Python: Streaming KV &amp; Continuous Batching Engine (PagedAttention &amp; Iteration Scheduling)
       │
       ▼ (Latency flaw: O(T) sequential decode latency bound by memory bandwidth)
[Stage 6 (Next)] 220 Lines Pure Python: Speculative Decoding &amp; INT4 Quantization Engine
</pre>
</fieldset>

---

## Step 1: 3-Year-Old Intuition (The Luggage Locker & The Revolving Carousel)

Imagine managing an airport baggage claim where passengers arrive at all hours:

<figure>
<pre>
[Traditional Static Batching: The Inflexible Tour Bus]
Passenger 1 (Fast: 1 bag)   ───► [Done at Min 1] ───► Must sit and wait for 59 mins!
Passenger 2 (Slow: 60 bags)  ───► Working...     ───► Working... ───► Finished at Min 60
Passenger 3 (Medium: 10 bags) ──► [Done at Min 10] ──► Must sit and wait for 50 mins!
                                └───────────────────────────────────────────────┘
                                       Bubble Waste: Over 60% Seats Idle!

[Continuous Batching: The Non-Stop Revolving Station]
Iteration 01: [Req A: Step 1] [Req B: Step 1] [Req C: Step 1]
Iteration 02: [Req A: DONE]   [Req B: Step 2] [Req C: Step 2]
                     │
                     ▼ (Instant Eviction &amp; Dynamic Slot Reclamation)
Iteration 03: [Req D: PREFILL][Req B: Step 3] [Req C: DONE]
                                                     │
                                                     ▼
Iteration 04: [Req D: Step 1] [Req B: Step 4] [Req E: PREFILL]  (Zero Idle Bubbles!)
</pre>
<figcaption><strong>Figure 25b.1:</strong> Continuous batching eliminates idle GPU bubble waste by scheduling on the discrete iteration level rather than the coarse batch level.</figcaption>
</figure>

1. **The Greedy Seat Reservation (Contiguous Memory Problem)**:
   In older systems, when a user opened a chat, the server asked: *"What is your maximum possible length?"* If the user said 4,096 words, the server cordoned off 4,096 contiguous empty seats in memory right away. If the user only asked *"What is 2+2?"* and received *"4"*, over 99% of those reserved seats sat completely empty, yet no other user was allowed to sit there!

2. **The Automatic Luggage Lockers (PagedAttention)**:
   Instead of reserving an entire hallway, the system buys small standardized 4-slot lockers (pages). As a sequence grows, it only claims a new 4-slot locker when its current locker is completely full. These lockers can sit anywhere in the warehouse (scattered physical memory), linked together by a small index card in the front office (the **Block Table**).

3. **The Non-Stop Revolving Carousel (Continuous Batching)**:
   Instead of waiting for every traveler on the tour bus to finish collecting luggage, the carousel rotates once every tick. If Passenger A grabs their single bag, they immediately leave the building. Their locker is wiped clean and handed to Passenger D, who just walked in the front door!

---

## Step 2: The Bridging Question

In standard PyTorch, multi-head attention expects a single contiguous tensor:

$$
\mathbf{K} \in \mathbb{R}^{B \times L \times d_k}, \quad \mathbf{V} \in \mathbb{R}^{B \times L \times d_v}
$$

How do we compute scaled dot-product attention when the historical tokens of a sequence are chopped into discrete pages and scattered randomly across non-contiguous physical memory blocks? And how does an iteration-level scheduler dynamically mix prompt prefill with ongoing token decoding?

---

## Step 3: The Exact Math & Formulas

### 1. PagedAttention Memory Address Translation

Let the sequence length be $L$ tokens. With a fixed block size $B$ (e.g., $B = 4$), the logical token index $t \in [0, L-1]$ is mapped to physical GPU memory via a block table lookup:

$$
\text{Block Index } i = \lfloor t / B \rfloor, \quad \text{Slot Offset } o = t \pmod B
$$

$$
\text{Physical Block ID } p = \text{BlockTable}[i]
$$

$$
\mathbf{k}_t = \text{PhysicalMemory}[p][\text{Key}][o], \quad \mathbf{v}_t = \text{PhysicalMemory}[p][\text{Value}][o]
$$

### 2. Paged Attention Kernel Over Scattered Blocks

For the current decoding step with query vector $\mathbf{q} \in \mathbb{R}^{1 \times d_k}$:

$$
S_t = \frac{\mathbf{q} \cdot \mathbf{k}_t}{\sqrt{d_k}} = \frac{\mathbf{q} \cdot \text{PhysicalMemory}[\text{BlockTable}[\lfloor t/B \rfloor]][\text{Key}][t \pmod B]}{\sqrt{d_k}}
$$

$$
\alpha_t = \frac{\exp(S_t - \max_j S_j)}{\sum_{\tau=0}^{L-1} \exp(S_\tau - \max_j S_j)}
$$

$$
\mathbf{o} = \sum_{t=0}^{L-1} \alpha_t \, \mathbf{v}_t
$$

### 3. Static Batching Bubble Waste Formula

If a batch of $N$ sequences runs under static batching until the longest sequence $L_{\max} = \max_i L_i$ completes, the total available compute slots is $N \cdot L_{\max}$. The bubble waste fraction $\eta_{\text{bubble}}$ is:

$$
\eta_{\text{bubble}} = 1 - \frac{\sum_{i=1}^N L_i}{N \cdot \max_{1 \le j \le N} L_j}
$$

Under continuous batching, completed sequences are evicted at the exact step they emit $\langle \text{eos} \rangle$, driving $\eta_{\text{bubble}} \to 0$.

---

## Step 4: Where Did It Come From?

In 2022, Yu et al. (UC Berkeley) introduced **Orca**, demonstrating that iteration-level scheduling unlocks up to $36\times$ higher throughput than static batching. However, Orca still struggled with memory fragmentation because it pre-allocated continuous memory buffers.

In 2023, Kwon et al. published **PagedAttention** and released **vLLM**. Inspired by virtual memory paging in operating systems (introduced in 1962 for the Atlas computer), vLLM decoupled logical sequence representation from physical GPU RAM. This reduced KV cache memory waste from $60\% - 80\%$ down to under $4\%$, enabling serving engines to support $2\times$ to $4\times$ higher concurrency on identical hardware.

---

## Step 5: Concrete Toy Example & Code Walkthrough

Let us trace 3 client requests with block size $B = 2$ and 4 total physical blocks:

<figure>
<pre>
[Physical Memory State &bull; 4 Blocks of Capacity 2]
Block 0: [ free ] [ free ]
Block 1: [ free ] [ free ]
Block 2: [ free ] [ free ]
Block 3: [ free ] [ free ]

Req 1 arrives: prompt = ["cat", "sat"] (needs 2 slots)
  - Allocates Block 0 -> BlockTable[Req 1] = [0]
  - Physical Block 0: ["cat", "sat"]

Req 2 arrives: prompt = ["the", "big", "dog"] (needs 3 slots)
  - Allocates Block 1 (slots: "the", "big")
  - Allocates Block 2 (slots: "dog", empty)
  - BlockTable[Req 2] = [1, 2]

Iteration 1 Decode:
  - Req 1 decodes: "on" -> Block 0 full! Allocates Block 3 -> BlockTable[Req 1] = [0, 3]
  - Req 2 decodes: "barks" -> Placed in remaining slot of Block 2!
</pre>
<figcaption><strong>Figure 25b.2:</strong> Physical blocks are dynamically attached only when needed, avoiding contiguous pre-allocation.</figcaption>
</figure>

### File Structure for Lab 05

```
25b-lab-streaming-engine/
├── streaming_engine.py          # Complete reference engine (~200 lines)
├── streaming_engine_exercise.py # Guided exercise with TODOs & unit tests
├── index.md                     # English curriculum guide
└── index.zh.md                  # Chinese curriculum guide
```

### Running the Reference Engine

```bash
python3 25b-lab-streaming-engine/streaming_engine.py
```

Output:
```
=====================================================================
Lab 05: Streaming KV & Continuous Batching Engine Simulation
=====================================================================

--- [Iteration 01] ---
  [ADMIT & PREFILL] Req 'Req-A': prompt ['the', 'cat'] -> blocks [0]
  [ADMIT & PREFILL] Req 'Req-B': prompt ['dog', 'sat', 'on'] -> blocks [1]
  [ADMIT & PREFILL] Req 'Req-C': prompt ['sun'] -> blocks [2]
  [DECODE STEP] Req 'Req-A' -> 'sat' (Blocks: [0])
  [DECODE STEP] Req 'Req-B' -> 'dog' (Blocks: [1])
  [DECODE STEP] Req 'Req-C' -> 'dog' (Blocks: [2])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

--- [Iteration 02] ---
  [FINISH & EVICT] Req 'Req-A' produced '<eos>'. Blocks freed! Total output: ['sat']
  [DECODE STEP] Req 'Req-B' -> 'sun' (Blocks: [1, 3])
  [DECODE STEP] Req 'Req-C' -> 'sun' (Blocks: [2])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

--- [Iteration 03] ---
  [ADMIT & PREFILL] Req 'Req-D': prompt ['the', 'cat', 'slept'] -> blocks [4]
  [DECODE STEP] Req 'Req-B' -> 'dog' (Blocks: [1, 3])
  [FINISH & EVICT] Req 'Req-C' produced '<eos>'. Blocks freed! Total output: ['dog', 'sun']
  [DECODE STEP] Req 'Req-D' -> 'the' (Blocks: [4])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

=====================================================================
Final Performance Audit & Bubble Waste Analysis
=====================================================================
Total Requests Completed: 4
  * Req-A: Prompt=2 tok, Gen=1 tok, Latency=1 iters
  * Req-C: Prompt=1 tok, Gen=2 tok, Latency=2 iters
  * Req-D: Prompt=3 tok, Gen=1 tok, Latency=1 iters
  * Req-B: Prompt=3 tok, Gen=5 tok, Latency=5 iters

Static Batching Theoretical Slot Usage:     20 slot-steps
Actual Useful Compute Tokens:               9 tokens
Static Batching Bubble Waste:               55.0%
Continuous Batching Bubble Waste:           0.0% (Slots instantly recycled!)
=====================================================================
```

### Running the Guided Exercise

```bash
python3 25b-lab-streaming-engine/streaming_engine_exercise.py
```

---

## Step 6: Core Takeaway

Serving LLMs at scale is primarily an **operating systems and memory scheduling challenge**: by breaking contiguous memory into standardized physical blocks (**PagedAttention**) and scheduling on the granularity of individual forward passes (**Continuous Batching**), we eliminate memory fragmentation and reclaim up to 60% of GPU compute that was previously wasted in padding bubbles.
