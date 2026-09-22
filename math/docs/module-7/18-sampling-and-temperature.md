# Chapter 18: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling)


## Step 1: 3-Year-Old Intuition (The Three Engines of Storytelling) {: #step-1 }

!!! note "3-Year-Old Intuition: From Reading the Recipe to Rolling the Dice"
    Imagine you have an intelligent storytelling music box. To tell a story, the box goes through three distinct mechanical stages:

    1. **The Reading Phase (<dfn id="def-prefill">Pre-fill Phase</dfn>) &mdash; Reading the Whole Recipe in One Glance**:
       - When you ask a question or give the box a prompt, you hand it a card with words on it: *"Once upon a time in a magical..."*
       - The box does not read word-by-word like a slow beginner. Instead, its eyes scan **every single word on the card at the exact same fraction of a second**.
       - It memorizes all relationships, lays out all puzzle pieces on its work table, and prepares its memory for writing. This is **Pre-fill**: high-speed, parallel ingestion of your entire prompt.

    2. **The Writing Phase (<dfn id="def-decoding">Decoding Phase</dfn>) &mdash; The Solitary Falling Domino**:
       - Now the box begins to create new words. But it cannot write tomorrow's word until today's word exists!
       - It writes word #1: <kbd>"forest"</kbd>.
       - Then it lifts its pen, looks back at its entire memory, and writes word #2: <kbd>"there"</kbd>.
       - Then it looks back again and writes word #3: <kbd>"lived"</kbd>.
       - Each new word is like a single domino falling: strictly sequential, one by one. This is **Decoding**: autoregressive, token-by-token generation.

    3. **The Repair Trick (<dfn id="def-midfill">Mid-fill / Fill-in-the-Middle</dfn>) &mdash; The Missing Bridge Span**:
       - What if you have already written the beginning of a story and the ending, but left a blank hole in the middle?
       - A normal music box only knows how to write forward into empty space.
       - **Mid-fill** is a clever packing trick: You tape the ending right behind the beginning, and say: *"Now fill the gap between them!"* The box can look at both sides while writing the missing middle.

    4. **The Temperature Dial (<dfn id="def-temperature">Temperature, $T$</dfn>) & The Velvet Rope Bouncers**:
       - When the box is about to speak each word, how does it choose?
       - **Turned to Freezing ($T \to 0$)**: The box strictly picks the single most predictable word every time. It becomes a frozen calculator.
       - **Turned to Warm Room Temperature ($T = 0.7$)**: The box picks mostly smart words, with occasional lively, creative metaphors.
       - **Turned to Boiling Fire ($T = 5.0$)**: Chaos explodes; it spits random gibberish.
       - **The Bouncers (Top-$k$ and Top-$p$)**: Velvet rope bouncers standing at the door to bar ridiculous, absurd tokens (like <kbd>"toaster"</kbd>) from ever being spoken!

<figure>
<pre>
The Complete End-to-End LLM Inference Pipeline:

Input Prompt: "The capital of France is"
              │
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: PRE-FILL PHASE (Parallel Context Ingestion)                   │
│ • Input: All S prompt tokens simultaneously [S × d]                    │
│ • Hardware: Compute-Bound GEMM (Tensor Cores 100% Saturated)           │
│ • Key Output: Initial Key-Value (KV) Cache populated in GPU VRAM       │
│ • Latency Metric: Time To First Token (TTFT)                           │
└────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: DECODING PHASE (Sequential Autoregressive Generation)         │
│ • Input: Exactly 1 new token vector [1 × d]                            │
│ • Hardware: Memory-Bandwidth-Bound GEMV (Streaming Cache from VRAM)    │
│ • Action: Appends new (K, V) to Cache; computes new Query against all  │
│ • Latency Metric: Time Per Output Token (TPOT)                         │
└────────────────────────────────────────────────────────────────────────┘
              │
              ▼ Produces Raw Logits z in R^|V|
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: SAMPLING & FILTERING (Creative Control)                       │
│ • Scale logits by Temperature: z_i / T                                 │
│ • Exponentiate via Softmax: p_i = exp(z_i/T) / sum exp(z_j/T)          │
│ • Truncate tails: Top-k (fixed pool) or Top-p (dynamic nucleus)        │
│ • Sample Final Token: "Paris" ──► Appended to Input ──► Repeat Phase 2!│
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 18.1:</strong> The complete LLM inference lifecycle: parallel Pre-fill constructs the KV Cache, sequential Decoding advances one token at a time, and Sampling reshapes the logits.</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: From Parallel Ingestion to Sequential Generation: The Hardware Divide"
    When you send a prompt of 2,000 words to an LLM, two vastly different computational problems occur in sequence:

    1. **The Pre-fill Divide**: Why can the GPU ingest all 2,000 prompt words in just ~15 milliseconds, yet generating the next 100 words takes 2,000 milliseconds (2 full seconds)?
    2. **The Memory Wall**: Why do multi-thousand-dollar GPUs with massive tensor cores sit over 85% idle during token generation, starved of work while waiting for memory buses?
    3. **The Infilling Dilemma**: Standard autoregressive models only generate from left to right. How can an LLM generate missing code in the **middle** of a file without rewriting the model architecture?
    4. **The Generation Trap**: Once output logits $\mathbf{z} \in \mathbb{R}^{|V|}$ are computed at each decoding step, why can't we just pick the #1 most probable word ($\arg\max$)?

    In 2019, Ari Holtzman and his co-authors proved (<cite>"The Curious Case of Neural Text Degeneration"</cite>) that pure **Greedy Decoding** causes language models to repeat themselves in infinite deterministic loops ($A \to B \to A \to B$). Human speech is neither completely predictable nor purely random.

    *"How do we mathematically formulate the Pre-fill and Decoding phases, how do we adapt causal models to fill in the middle, and how do we use statistical thermodynamics to sample natural, creative language?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Pre-fill Phase (Full-Context Ingestion & KV Cache Construction)

In the Pre-fill phase, the entire prompt sequence of length $S$ is provided up front as an embedding matrix $\mathbf{X} \in \mathbb{R}^{S \times d}$.

The Transformer computes Queries, Keys, and Values for all $S$ tokens in a single dense matrix multiplication:



$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{S \times d_k}, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{S \times d_k}, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{S \times d_v}
$$



The self-attention score matrix is calculated across the entire prompt simultaneously:



$$
\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \in \mathbb{R}^{S \times S}
$$





$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{S \times d_v}
$$



where $\mathbf{M} \in \mathbb{R}^{S \times S}$ is the lower-triangular causal mask.

#### The Fundamental Outcome: Initializing the KV Cache
Because all $S$ prompt tokens are known, we do not discard their Keys and Values. Instead, we write them into GPU High-Bandwidth Memory (<abbr title="High Bandwidth Memory">HBM</abbr>) as the initial **KV Cache**:



$$
\mathbf{K}_{\text{cache}}^{(0)} = \mathbf{K} \in \mathbb{R}^{S \times d_k}, \quad \mathbf{V}_{\text{cache}}^{(0)} = \mathbf{V} \in \mathbb{R}^{S \times d_v}
$$



#### Hardware Characteristics of Pre-fill:
- **Operation**: General Matrix-Matrix Multiplication (<abbr title="General Matrix Multiply">GEMM</abbr>).
- **Arithmetic Intensity**: $\mathcal{O}(S \cdot d)$ floating-point operations per byte loaded. Because $S$ is large (hundreds or thousands of tokens), arithmetic intensity is high ($\gg 100$ FLOPs/byte).
- **Bottleneck**: **Compute-Bound**. GPU Tensor Cores are fully utilized.
- **Key Benchmark Metric**: **Time To First Token (<abbr title="Time To First Token">TTFT</abbr>)**.

---

### 2. The Decoding Phase (Incremental Autoregressive Stepping)

Once the prompt is pre-filled, the model enters the sequential Decoding phase to generate tokens $t = S+1, S+2, \dots$ one by one.

At step $t$, the input is strictly **a single token vector**:



$$
\mathbf{x}_t \in \mathbb{R}^{1 \times d}
$$



We project only this single token into Query, Key, and Value:



$$
\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k}, \quad
\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times d_k}, \quad
\mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times d_v}
$$



#### Incremental KV Cache Update:
We do not recompute past tokens. We append the new $(\mathbf{k}_t, \mathbf{v}_t)$ to the cache:



$$
\mathbf{K}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{K}_{\text{cache}} \\ \mathbf{k}_t \end{bmatrix} \in \mathbb{R}^{t \times d_k}, \quad
\mathbf{V}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{V}_{\text{cache}} \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{t \times d_v}
$$



#### Single-Query Attention:
The single Query vector $\mathbf{q}_t$ attends across all $t$ historical keys:



$$
\mathbf{a}_t = \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{\text{cache}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times t}
$$





$$
\mathbf{o}_t = \mathbf{a}_t \mathbf{V}_{\text{cache}} \in \mathbb{R}^{1 \times d_v}
$$



#### Hardware Characteristics of Decoding (The Memory Wall):
- **Operation**: General Matrix-Vector Multiplication (<abbr title="General Matrix Vector Multiply">GEMV</abbr>).
- **Arithmetic Intensity**: Extremely low ($\approx 1\text{--}2$ FLOPs per byte loaded). To calculate the attention for just 1 token, the GPU must load all model weights ($\approx 140\text{ GB}$ for a 70B parameter model) and all cached Keys and Values from VRAM into registers!
- **Bottleneck**: **Memory-Bandwidth-Bound**. The Tensor Cores sit mostly idle waiting for data to crawl across the memory bus.
- **Key Benchmark Metric**: **Time Per Output Token (<abbr title="Time Per Output Token">TPOT</abbr>)** / Inter-Token Latency.

---

### 3. Mid-fill (Fill-in-the-Middle / FIM) Inference Pipeline

When a user asks an LLM to complete a missing section in existing text or code:
- **Prefix ($P$)**: The text preceding the cursor.
- **Suffix ($S$)**: The text succeeding the cursor.
- **Middle ($M$)**: The missing text to be generated.

In a standard causal decoder, generating $M$ after $P$ cannot see $S$ because causal masking hides future positions.

**The Inference Execution Protocol for Mid-fill:**
1. **Prompt Rearrangement**: The inference system wraps the inputs with special delimiters into Prefix-Suffix-Middle format:


   $$
   \mathbf{X}_{\text{infill}} = \langle\text{PRE}\rangle \circ P \circ \langle\text{SUF}\rangle \circ S \circ \langle\text{MID}\rangle
   $$


2. **Pre-fill Stage**: The entire sequence $\mathbf{X}_{\text{infill}}$ is pre-filled simultaneously in parallel. All Keys and Values for both $P$ and $S$ are stored in the KV Cache.
3. **Decoding Stage**: Starting at the position right after $\langle\text{MID}\rangle$, the model generates the middle tokens $m_1, m_2, \dots$ one by one. Because $P$ and $S$ are located earlier in the physical sequence, every generated middle token can freely attend to both anchors!
4. **Termination**: Decoding terminates when the model emits the $\langle\text{EOT}\rangle$ (End-of-Transmission) token. The inference engine extracts the generated middle string and inserts it directly at the user's cursor.

---

### 4. Temperature Scaling (Boltzmann Distribution)

Let $\mathbf{z} = [z_1, z_2, \dots, z_{|V|}]^\top \in \mathbb{R}^{|V|}$ be the raw output logits.
The **Temperature-Scaled Softmax** defines the probability of token $i$ as:



$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}
$$



where $T > 0$ is the **temperature parameter**.

#### Mathematical Properties Across Temperature Regimes:
1. **As $T \to 0^+$ (Argmax / Greedy Limit)**:
   The gap between the largest logit $z_{\max}$ and all other logits approaches infinity:


   $$
   \lim_{T \to 0^+} p_i(T) = \begin{cases} 1 & \text{if } z_i = \max_j z_j \\ 0 & \text{otherwise} \end{cases}
   $$


   The probability collapses into a deterministic **one-hot Dirac delta distribution**.

2. **Standard Temperature ($T = 1.0$)**:
   Recovers the pure mathematical Softmax distribution used during cross-entropy training:


   $$
   p_i(1.0) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
   $$



3. **As $T \to \infty$ (Uniform Noise Limit)**:
   All scaled logits approach zero ($z_i / T \to 0$), meaning $\exp(z_i / T) \to 1$:


   $$
   \lim_{T \to \infty} p_i(T) = \frac{1}{|V|}
   $$


   The distribution becomes completely flat (maximum entropy / pure white noise).

---

### 5. Top-$k$ Truncation Filtering (Fan et al., 2018)

Top-$k$ filtering restricts the sampling pool to the $k$ tokens with the largest logits, setting all other logits to negative infinity:



$$
z'_i = \begin{cases} z_i & \text{if } z_i \ge z_{(k)} \\ -\infty & \text{otherwise} \end{cases}
$$



where $z_{(k)}$ is the $k$-th largest logit in the vocabulary.
After masking, Softmax is recomputed over the remaining $k$ tokens:



$$
p'_i = \frac{\exp(z'_i / T)}{\sum_{j=1}^{|V|} \exp(z'_j / T)}
$$



---

### 6. Top-$p$ Truncation Filtering (Nucleus Sampling; Holtzman et al., 2019)

Top-$k$ has a major flaw: a fixed $k=50$ is far too large when the model is confident (admitting 49 junk tokens), and too small when the context is genuinely ambiguous (excluding valid creative options).

**Top-$p$ (Nucleus) Sampling** solves this by dynamically adapting the cutoff threshold:
1. Sort all vocabulary tokens in descending order of probability:


   $$
   p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(|V|)}
   $$


2. Find the smallest index $k^*$ such that the cumulative distribution function (<abbr title="Cumulative Distribution Function">CDF</abbr>) reaches threshold $p \in (0, 1]:


   $$
   k^* = \min \left\{ k : \sum_{i=1}^k p_{(i)} \ge p \right\}
   $$


3. Define the nucleus subset $V^{(p)} = \{ (1), (2), \dots, (k^*) \}$.
4. Re-normalize the probabilities strictly over $V^{(p)}$:


   $$
   p'_i = \begin{cases} \frac{p_i}{\sum_{j \in V^{(p)}} p_j} & \text{if } i \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}
   $$



<figure>
<pre>
Top-p Nucleus Truncation Dynamics:

Case A: Highly Confident Context ("The capital of France is...")
  Tokens:       [ Paris (0.97) │ Lyon (0.01) │ Marseille (0.01) │ ... ]
  CDF:            0.97 >= 0.90 ──► Cutoff! Pool size = 1 token!

Case B: Open Creative Context ("She looked outside and saw a...")
  Tokens:       [ bird (0.18) │ tree (0.15) │ cat (0.12) │ car (0.10) ... ]
  CDF:            0.18 + 0.15 + 0.12 + 0.10 + ... >= 0.90 ──► Pool = 18 tokens!
</pre>
<figcaption><strong>Figure 18.2:</strong> Nucleus sampling dynamically contracts to 1 token when confident, and expands to dozens when ambiguous.</figcaption>
</figure>

---

## Step 4: Where Did It Come From? (Hardware Bottlenecks & Sampling Theory) {: #step-4 }

<dl>
  <dt><time datetime="1877">1877</time> &mdash; <strong>Ludwig Boltzmann</strong></dt>
  <dd>Formulated statistical mechanics, showing that the probability of a physical system occupying microstate $i$ with energy $E_i$ at thermodynamic temperature $T$ follows $p_i \propto \exp(-E_i / k_B T)$. In LLMs, the negative logit $-z_i$ acts as the particle energy state.</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Ashish Vaswani et al.</strong> (<cite>"Attention Is All You Need"</cite>)</dt>
  <dd>Eliminated temporal recurrence in favor of spatial self-attention, creating the dual-phase inference reality: instantaneous parallel <strong>Pre-fill</strong> ($O(1)$ sequential steps) versus strictly serial autoregressive <strong>Decoding</strong> ($O(T)$ sequential steps).</dd>

  <dt><time datetime="2018">2018</time> &mdash; <strong>Angela Fan, Mike Lewis, & Yann Dauphin</strong> (<cite>"Hierarchical Neural Story Generation"</cite>)</dt>
  <dd>Introduced Top-$k$ random sampling for neural text generation at Meta AI, proving that truncating the unreliable probability tail dramatically eliminated repetition and gibberish compared to unconstrained sampling.</dd>

  <dt><time datetime="2019">2019</time> &mdash; <strong>Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, & Yejin Choi</strong> (<cite>"The Curious Case of Neural Text Degeneration"</cite>)</dt>
  <dd>Discovered that human language does not maximize probability, exposed the fatal flaws of both greedy decoding and fixed Top-$k$, and invented Nucleus (Top-$p$) Sampling, which remains the default decoding algorithm in ChatGPT, Claude, and Gemini.</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>Mohammad Bavarian et al.</strong> (<cite>"Efficient Training of Language Models to Fill in the Middle"</cite>)</dt>
  <dd>Invented Fill-in-the-Middle (FIM) at OpenAI, proving that permuting prefix, middle, and suffix during training allows standard causal LLMs to seamlessly perform in-place code editing and infilling with zero architectural changes.</dd>

  <dt><time datetime="2023">2023</time> &mdash; <strong>Woosuk Kwon et al.</strong> (<cite>"Efficient Memory Management for Large Language Model Serving with PagedAttention"</cite>)</dt>
  <dd>Created vLLM and PagedAttention to conquer the Decoding Memory Wall, virtualizing the KV Cache like operating system virtual memory to eliminate fragmentation during multi-tenant token generation.</dd>
</dl>

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

### 1. Pre-fill vs. Decoding Trace (Tensor Dimensions & Hardware Arithmetic)

Consider a miniature language model with embedding dimension $d = 4$ and $d_k = 4$.
We provide a 3-token prompt: <kbd>"who are you"</kbd> ($S = 3$).

```
Prompt tokens:   x_1="who", x_2="are", x_3="you"  (S = 3)
Model dimension: d = 4
```

#### Phase 1: Pre-fill (Prompt Processing)
1. **Input Matrix**: $\mathbf{X} \in \mathbb{R}^{3 \times 4}$.
2. **Dense Parallel Projection**:


   $$
   \mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{3 \times 4}, \quad
   \mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{3 \times 4}, \quad
   \mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{3 \times 4}
   $$


3. **Full Attention Score Matrix** (GEMM):


   $$
   \mathbf{S}_{\text{attn}} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{4}} + \mathbf{M} \in \mathbb{R}^{3 \times 3}
   $$


   All 9 pairs of attention interactions are computed **simultaneously in 1 clock cycle**.
4. **KV Cache Initialization**:
   We store $\mathbf{K} \in \mathbb{R}^{3 \times 4}$ and $\mathbf{V} \in \mathbb{R}^{3 \times 4}$ in GPU VRAM:




   $$
   \text{Cache size} = 2 \times (3 \times 4) = 24 \text{ numbers}.
   $$




5. **Output Token #1**: The final row $\mathbf{O}[3]$ produces logits; we sample <kbd>"I"</kbd>.

#### Phase 2: Decoding (Generating Token #2)
To generate the next token after <kbd>"I"</kbd> ($t = 4$):
1. **Input Vector**: Strictly a single vector $\mathbf{x}_4 \in \mathbb{R}^{1 \times 4}$ for token <kbd>"I"</kbd>.
2. **Single Projection**:


   $$
   \mathbf{q}_4 = \mathbf{x}_4 \mathbf{W}_Q \in \mathbb{R}^{1 \times 4}, \quad
   \mathbf{k}_4 = \mathbf{x}_4 \mathbf{W}_K \in \mathbb{R}^{1 \times 4}, \quad
   \mathbf{v}_4 = \mathbf{x}_4 \mathbf{W}_V \in \mathbb{R}^{1 \times 4}
   $$


3. **Cache Append**:


   $$
   \mathbf{K}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{K}_{\text{cache}} \\ \mathbf{k}_4 \end{bmatrix} \in \mathbb{R}^{4 \times 4}, \quad
   \mathbf{V}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{V}_{\text{cache}} \\ \mathbf{v}_4 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
   $$


4. **Incremental Attention (GEMV)**:


   $$
   \mathbf{a}_4 = \operatorname{softmax}\left(\frac{\mathbf{q}_4 \mathbf{K}_{\text{cache}}^\top}{\sqrt{4}}\right) \in \mathbb{R}^{1 \times 4}
   $$


   $\mathbf{q}_4$ is a single row ($1 \times 4$). It multiplies the $4 \times 4$ key matrix.
   To do this tiny dot product, the GPU had to load **all** weight matrices and the **entire** cache from memory across the bus!

---

### 2. Mid-fill (FIM) Inference Trace

Suppose we want the model to complete the function logic:
- **Prefix ($P$)**: `def square(x):`
- **Suffix ($S$)**: `return y`
- **Target Middle ($M$)**: `y = x * x`

1. The inference system wraps the text into the PSM format:
   ```
   [PRE] def square(x):\n [SUF] return y [MID]
   ```
2. **Pre-fill Phase**: Ingests all 7 prompt tokens in parallel, constructing the initial KV Cache for both `def square(x):` and `return y`.
3. **Decoding Phase**: The model generates `y`, then `=`, then `x`, then `*`, then `x`, attending to both the function signature and the return statement.
4. **Termination**: The model emits `[EOT]`. The system extracts `y = x * x` and inserts it cleanly into the file.

---

### 3. Sampling Probability Calculations ($T$, Top-$k$, Top-$p$)

Let us calculate the exact probabilities for a miniature 4-word vocabulary under different temperatures, apply Top-$k$, and apply Top-$p$.

#### Miniature Vocabulary & Raw Logits
Consider vocabulary $V = \{\text{cat}, \text{dog}, \text{fish}, \text{toaster}\}$ with raw output logits:



$$
\mathbf{z} = [z_{\text{cat}} = 4.0, \; z_{\text{dog}} = 2.0, \; z_{\text{fish}} = 1.0, \; z_{\text{toaster}} = -1.0]
$$



---

#### Temperature Comparison ($T = 0.5$ vs. $T = 1.0$ vs. $T = 2.0$)

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step A:</strong> Scale logits by temperature $\tilde{z}_i = z_i / T$.</p>
<p><input type="checkbox" checked disabled> <strong>Step B:</strong> Compute exponentials $e^{\tilde{z}_i}$ and sum.</p>
<p><input type="checkbox" checked disabled> <strong>Step C:</strong> Normalize probabilities $p_i = e^{\tilde{z}_i} / \sum e^{\tilde{z}_j}$.</p>
<p><input type="checkbox" checked disabled> <strong>Step D:</strong> Apply Top-$k = 2$ filter and re-normalize.</p>
<p><input type="checkbox" checked disabled> <strong>Step E:</strong> Apply Top-$p = 0.90$ nucleus filter and re-normalize.</p>
</fieldset>

#### Case A: Neutral Temperature ($T = 1.0$)
- Scaled logits: $\mathbf{z} / 1.0 = [4.0, 2.0, 1.0, -1.0]$
- Exponentials:
  - $e^{4.0} \approx 54.5982$
  - $e^{2.0} \approx 7.3891$
  - $e^{1.0} \approx 2.7183$
  - $e^{-1.0} \approx 0.3679$
  - Sum $= 54.5982 + 7.3891 + 2.7183 + 0.3679 = \mathbf{65.0735}$
- Probabilities:
  - $p(\text{cat}) = 54.5982 / 65.0735 = \mathbf{0.8390} \; (83.9\%)$
  - $p(\text{dog}) = 7.3891 / 65.0735 = \mathbf{0.1136} \; (11.4\%)$
  - $p(\text{fish}) = 2.7183 / 65.0735 = \mathbf{0.0418} \; (4.2\%)$
  - $p(\text{toaster}) = 0.3679 / 65.0735 = \mathbf{0.0057} \; (0.6\%)$

#### Case B: Cold Temperature ($T = 0.5$ &mdash; Sharpening)
- Scaled logits: $\mathbf{z} / 0.5 = [8.0, 4.0, 2.0, -2.0]$
- Exponentials:
  - $e^{8.0} \approx 2980.9580$
  - $e^{4.0} \approx 54.5982$
  - $e^{2.0} \approx 7.3891$
  - $e^{-2.0} \approx 0.1353$
  - Sum $= 2980.9580 + 54.5982 + 7.3891 + 0.1353 = \mathbf{3043.0806}$
- Probabilities:
  - $p(\text{cat}) = 2980.9580 / 3043.0806 = \mathbf{0.9796} \; (98.0\%)$
  - $p(\text{dog}) = 54.5982 / 3043.0806 = \mathbf{0.0179} \; (1.8\%)$
  - $p(\text{fish}) = 7.3891 / 3043.0806 = \mathbf{0.0024} \; (0.2\%)$
  - $p(\text{toaster}) = 0.1353 / 3043.0806 = \mathbf{0.00004} \; (0.004\%)$

<mark>At $T = 0.5$, <kbd>"cat"</kbd> surged from $83.9\%$ to $98.0\%$. The distribution became crisp and near-deterministic.</mark>

#### Case C: Hot Temperature ($T = 2.0$ &mdash; Flattening)
- Scaled logits: $\mathbf{z} / 2.0 = [2.0, 1.0, 0.5, -0.5]$
- Exponentials:
  - $e^{2.0} \approx 7.3891$
  - $e^{1.0} \approx 2.7183$
  - $e^{0.5} \approx 1.6487$
  - $e^{-0.5} \approx 0.6065$
  - Sum $= 7.3891 + 2.7183 + 1.6487 + 0.6065 = \mathbf{12.3626}$
- Probabilities:
  - $p(\text{cat}) = 7.3891 / 12.3626 = \mathbf{0.5977} \; (59.8\%)$
  - $p(\text{dog}) = 2.7183 / 12.3626 = \mathbf{0.2199} \; (22.0\%)$
  - $p(\text{fish}) = 1.6487 / 12.3626 = \mathbf{0.1334} \; (13.3\%)$
  - $p(\text{toaster}) = 0.6065 / 12.3626 = \mathbf{0.0491} \; (4.9\%)$

<mark>At $T = 2.0$, <kbd>"toaster"</kbd> gained an alarming $4.9\%$ chance of being selected!</mark>

---

#### Truncation in Action: Top-$k$ and Top-$p$ at $T = 1.0$

Starting with the $T = 1.0$ probabilities:
- $\text{cat}: 0.8390$
- $\text{dog}: 0.1136$
- $\text{fish}: 0.0418$
- $\text{toaster}: 0.0057$

##### Applying Top-$k = 2$:
1. Retain only the top 2 candidates: $\{\text{cat}, \text{dog}\}$. Discard $\{\text{fish}, \text{toaster}\}$.
2. Sum of retained probabilities: $0.8390 + 0.1136 = \mathbf{0.9526}$.
3. Re-normalize:
   - $p'(\text{cat}) = 0.8390 / 0.9526 = \mathbf{0.8808} \; (88.1\%)$
   - $p'(\text{dog}) = 0.1136 / 0.9526 = \mathbf{0.1192} \; (11.9\%)$
   - $p'(\text{fish}) = \mathbf{0.0000}$
   - $p'(\text{toaster}) = \mathbf{0.0000}$

##### Applying Top-$p = 0.90$ (Nucleus):
1. Accumulate probabilities in descending order:
   - Candidate 1: $\text{cat} \to \text{Cumulative} = 0.8390 < 0.90$ (continue)
   - Candidate 2: $\text{dog} \to \text{Cumulative} = 0.8390 + 0.1136 = 0.9526 \ge 0.90$ (**Threshold reached! Truncate pool here!**)
2. The nucleus contains exactly $\{\text{cat}, \text{dog}\}$.
3. Re-normalized distribution:
   - $p'(\text{cat}) = \mathbf{88.1\%}$
   - $p'(\text{dog}) = \mathbf{11.9\%}$
   - The absurd tail token <kbd>"toaster"</kbd> has been completely and safely eradicated!

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: The Punchline of LLM Inference"
    **Pre-fill consumes context in parallel to build the memory; Decoding steps forward sequentially one domino at a time; Mid-fill bridges gaps between anchors; and Sampling transforms raw mathematics into creative, coherent human intelligence.**

    By balancing compute-bound ingestion with memory-bound token stepping, and shaping logits via thermodynamic temperature and nucleus filters, modern inference engines achieve the delicate balance between speed, fluency, and spontaneity.
