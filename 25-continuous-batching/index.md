# Chapter 25: Cellular Scheduling (Continuous Batching & Chunked Prefill)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine a theme park roller coaster with 4 seats.
> 
> Four riders climb in. Rider 1 only wants a short 1-minute ride. Rider 2 wants a 2-minute ride. But Rider 4 is an adrenaline junkie who bought a ticket for a 50-minute marathon ride!
> 
> In traditional **Static Batching**:
> The coaster takes off. After 1 minute, Rider 1 is dizzy and ready to leave. But the safety bars are locked! Rider 1 has to sit strapped into their seat for the remaining 49 minutes doing nothing. Meanwhile, outside the gate, 100 excited kids are waiting in line in the hot sun, but the operator refuses to let anyone on until Rider 4 finishes all 50 minutes. Almost every seat sits empty or wasted for 90% of the day.
> 
> Then, a clever ride designer invents **Continuous Batching**:
> Every single time the coaster loops past the platform, it slows down for just 1 second.
> - Whoever is finished unbuckles and walks out.
> - The next kid in line immediately jumps into that empty seat and clicks their seatbelt.
> - The coaster accelerates back onto the track with **all 4 seats full on every single loop**!
> 
> And what if a giant elephant arrives with 10 suitcases?
> Instead of stopping the whole roller coaster for an hour to load the elephant, the operator uses **Chunked Prefill**: load 2 suitcases on loop 1, 2 suitcases on loop 2, and 2 on loop 3. The coaster never stops, and nobody in line ever gets stuck waiting.

---

## Step 2: The Bridging Question

How do we convert this roller coaster platform into a mathematical scheduling state machine that operates at the granularity of individual token iterations, and how does combining Prefill and Decode tokens in the same forward pass solve the memory wall?

In machine learning training, batches are completely static: tensors have a fixed rectangular shape $[B, T]$, and every sample executes for the exact same forward and backward pass.

In production LLM serving, incoming requests exhibit three forms of extreme dynamism:
1. **Arrival Times**: User requests arrive unpredictably as a Poisson process.
2. **Prompt Lengths**: Prompt sizes vary wildly from 5 tokens to 32,000 tokens.
3. **Output Lengths**: Generation terminates non-deterministically whenever the model predicts the special `<eos>` (End-of-Sequence) token.

If an engine groups requests into static batches, early-terminating requests produce massive **execution bubbles**: GPU Tensor Cores waste compute multiplying zeros and padding masks.

The bridging question is:
$$\text{How can an inference scheduler update its batch membership after every single token generation step, and how can chunking long prompts allow memory-bound decode requests to piggyback on compute-dense prefill operations?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STATIC BATCHING VS. CONTINUOUS BATCHING                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STATIC BATCHING (Coarse-Grained Request Level):                                         │
│   Slot 0: [ Req A (3 tokens) ] [ PADDING PADDING PADDING PADDING ] &lt;── 60% Bubble Waste│
│   Slot 1: [ Req B (5 tokens) ] [ PADDING PADDING ]                                     │
│   Slot 2: [ Req C (8 tokens - Longest) ]                                               │
│   ──► Slot 0 and Slot 1 sit idle until Slot 2 finishes! New requests blocked in queue.  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CONTINUOUS BATCHING (Fine-Grained Iteration Level):                                    │
│   Slot 0: [ Req A (3 tokens) ] ──► EVICT ──► [ Req D arrives and starts instantly! ]   │
│   Slot 1: [ Req B (5 tokens) ] ──► EVICT ──► [ Req E arrives and starts instantly! ]   │
│   Slot 2: [ Req C continues generating... ]                                           │
│   ──► Zero bubble padding! 100% GPU slot occupancy across every single forward pass.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 25.1:</strong> Static batching locks GPU slots until the longest request completes, generating massive bubble waste. Continuous batching evicts finished requests and inserts new requests on every iteration.</figcaption>
</figure>

### 1. The Bubble Waste Equation in Static Batching

Let a static batch contain $B$ requests with generation lengths $T_{\text{gen}}^{(1)}, T_{\text{gen}}^{(2)}, \dots, T_{\text{gen}}^{(B)}$.
Because the batch cannot finish until the longest request completes, total wall-clock iterations is:
$$
T_{\text{max}} = \max_{i=1}^B T_{\text{gen}}^{(i)}
$$

The total number of token generation slots allocated is $B \times T_{\text{max}}$, while the number of genuinely useful generated tokens is $\sum_{i=1}^B T_{\text{gen}}^{(i)}$.

The fractional <dfn id="def-bubble">Bubble Waste</dfn> $\eta_{\text{bubble}}$ is defined as:

$$
\eta_{\text{bubble}} = 1 - \frac{\sum_{i=1}^B T_{\text{gen}}^{(i)}}{B \times \max_{i=1}^B T_{\text{gen}}^{(i)}}
$$

In production deployments with diverse user queries, $\eta_{\text{bubble}}$ routinely exceeds **70%**, meaning over two-thirds of GPU serving capacity is spent processing empty padding!

---

### 2. Continuous (Iteration-Level) Batching State Transition

Let $\mathcal{R}_{\text{waiting}}$ denote the FIFO queue of pending requests, and $\mathcal{B}_k = \{r_1, r_2, \dots, r_{B_k}\}$ denote the active batch executing at iteration $k$.

At each discrete iteration $k \to k + 1$:
1. **Execution**: The GPU runs exactly one forward step for all requests in $\mathcal{B}_k$.
2. **Termination Check**: For each request $r_i \in \mathcal{B}_k$, let $y_k^{(i)}$ be its generated token:
   $$\text{Finished}(r_i) = \left(y_k^{(i)} = \langle\text{eos}\rangle\right) \lor \left(\text{len}(r_i) \ge T_{\text{limit}}\right)$$
3. **Dynamic Eviction &amp; Injection**:
   $$\mathcal{B}_{\text{surviving}} = \left\{r_i \in \mathcal{B}_k \mid \neg \text{Finished}(r_i)\right\}$$
   Available slots $N_{\text{free}} = B_{\text{max}} - |\mathcal{B}_{\text{surviving}}|$ are immediately populated by dequeuing from $\mathcal{R}_{\text{waiting}}$:
   $$
   \mathcal{B}_{k+1} = \mathcal{B}_{\text{surviving}} \cup \operatorname{Dequeue}\left(\mathcal{R}_{\text{waiting}}, N_{\text{free}}\right)
   $$

GPU slots are refilled continuously on every token clock, driving bubble waste $\eta_{\text{bubble}} \to 0$.

---

### 3. Chunked Prefill &amp; The Piggyback Math

A severe problem remains: when a new request enters $\mathcal{B}_{k+1}$, its entire prompt must be prefilled. If a prompt is 8,000 tokens long, running its prefill monopolizes the GPU for hundreds of milliseconds, causing existing decode requests to suffer massive latency spikes (<dfn id="def-tpot">Time Per Output Token</dfn> degradation).

**Chunked Prefill** (introduced in Sarathi-Serve) caps the maximum number of tokens processed in any iteration to a budget $K_{\text{budget}}$ (e.g., $K_{\text{budget}} = 512$).
If a prompt has length $T_{\text{prompt}} > K_{\text{budget}}$, it is split into sequential chunks:
$$
C = \min\left(T_{\text{remaining}}, \; K_{\text{budget}} - N_{\text{decode}}\right)
$$

Now, consider the operational intensity of a **Mixed Batch** containing $N_{\text{decode}}$ single decode tokens and $C$ chunked prefill tokens:
- **Total Compute**:
  $$\text{FLOPs} \approx 2 \times (N_{\text{decode}} + C) \times P$$
  (where $P$ is total model parameter count).
- **Total Memory Transferred**:
  Model weights are streamed from HBM **once** per layer:
  $$\text{Bytes} \approx 2 \times P \times p$$
- **Arithmetic Intensity**:
  $$
  I_{\text{mixed}} = \frac{2 \times (N_{\text{decode}} + C) \times P}{2 \times P \times p} = \frac{N_{\text{decode}} + C}{p} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

When $N_{\text{decode}} = 32$ and $C = 480$ with 16-bit precision ($p = 2$ bytes):
$$
I_{\text{mixed}} = \frac{32 + 480}{2} = 256 \text{ FLOPs/Byte} \approx I^*
$$

<mark>The Magic of Mixed Serving:</mark> The memory-bandwidth-bound decode tokens **piggyback** on the weights already being loaded into SRAM for the compute-heavy prefill chunk! Decode latency drops to near-zero marginal cost, while prefill runs without causing latency jitter.

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 25.1:</strong> Serving paradigm comparison.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Paradigm</th>
      <th align="center">Scheduling Granularity</th>
      <th align="center">Bubble Waste</th>
      <th align="center">Decode Latency Jitter</th>
      <th align="right">Throughput Multiplier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Static Batching</strong></td>
      <td align="center">Request-level (Coarse)</td>
      <td align="center">50% – 80%</td>
      <td align="center">High</td>
      <td align="right">1.0x (Baseline)</td>
    </tr>
    <tr>
      <td><strong>Continuous Batching (Orca)</strong></td>
      <td align="center">Iteration-level (Fine)</td>
      <td align="center"><strong>&lt; 5%</strong></td>
      <td align="center">Moderate (prefill pauses decodes)</td>
      <td align="right"><strong>2.5x – 3.5x</strong></td>
    </tr>
    <tr>
      <td><strong>Chunked Prefill (Sarathi)</strong></td>
      <td align="center">Iteration + Token Chunk</td>
      <td align="center"><strong>&lt; 1%</strong></td>
      <td align="center"><strong>Near Zero (Stable TPOT)</strong></td>
      <td align="right"><strong>4.0x+</strong></td>
    </tr>
  </tbody>
</table>

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2022">2022</time> &mdash; <strong>Orca: Iteration-Level Scheduling</strong> (<cite>Gyeong-In Yu et al., Seoul National University &amp; Microsoft Research, OSDI 2022</cite>)</dt>
  <dd>Introduced the foundational concept of Continuous Batching (iteration-level scheduling). By breaking the rigid request-level batch boundary, Orca demonstrated a 36x throughput improvement over standard static batching systems.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>Sarathi-Serve &amp; Chunked Prefill</strong> (<cite>Amey Agrawal et al., Microsoft &amp; Georgia Tech, OSDI 2024</cite>)</dt>
  <dd>Recognized that unconstrained prefills in continuous batching caused severe decode latency spikes. Sarathi introduced Chunked-Prefill and piggybacking, co-scheduling prefill and decode tokens to create a uniform compute profile across all iterations.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace 3 requests on a GPU cluster with a batch capacity of $B_{\text{max}} = 2$ slots.

### Request Arrival Specs
- **Request A**: Arrives at $t=0$, needs 1 token.
- **Request B**: Arrives at $t=0$, needs 3 tokens.
- **Request C**: Arrives at $t=1$, needs 2 tokens.

---

### Method 1: Static Batching
At $t=0$, system forms a batch with Request A and Request B:
- **Iteration 0**: Req A computes token 1 (Done!). Req B computes token 1.
- **Iteration 1**: Req A is finished, but sits locked in padding. Req B computes token 2.
- **Iteration 2**: Req A sits locked in padding. Req B computes token 3 (Done!).

Batch finishes at $t=3$.
Total slots used = $2 \times 3 = 6$ slots.
Actual useful tokens = $1 + 3 = 4$ tokens.
$$\text{Bubble Waste} = 1 - \frac{4}{6} = 33.3\%$$

Request C had to wait in the queue until $t=3$ just to begin!

---

### Method 2: Continuous Batching
- **Iteration 0 ($t=0$)**:
  - Active: [Req A, Req B].
  - Req A generates token 1 $\to$ Finished!
  - Req B generates token 1.
- **Iteration 1 ($t=1$)**:
  - Req A is **evicted immediately**.
  - Req C (which just arrived) is **injected immediately** into Slot 0!
  - Active: [Req C, Req B].
  - Req C generates token 1.
  - Req B generates token 2.
- **Iteration 2 ($t=2$)**:
  - Active: [Req C, Req B].
  - Req C generates token 2 $\to$ Finished!
  - Req B generates token 3 $\to$ Finished!

All 3 requests are 100% complete by $t=3$!
Total slots used = 6 slots.
Actual useful tokens = $1 (\text{A}) + 3 (\text{B}) + 2 (\text{C}) = 6\text{ tokens}$.
$$\text{Bubble Waste} = 1 - \frac{6}{6} = 0\%$$
Zero wasted cycles, and Request C finished in half the time!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p><strong>Continuous Batching</strong> transforms LLM serving from rigid, request-level batching into a fluid, iteration-level state machine that evicts finished sequences and admits new requests on every single token cycle.</p>
<p>When combined with <strong>Chunked Prefill</strong>, it mathematically fuses compute-dense prompt chunks with memory-bound decode tokens in the same forward pass, driving bubble waste to zero and unlocking stable, low-latency interactive serving.</p>
</fieldset>
