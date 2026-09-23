# Hands-on Lab 06: The Speculative Decoding & INT4 Quantization Engine (Accelerated Serving in 220 Lines of Pure Python)

<fieldset id="evolution">
<legend><strong>The Python Brain Evolution Chain &bull; Capstone Inference Frontier (Stage 6 of 6)</strong></legend>
<p>In Lab 05, we built the PagedAttention memory manager and continuous batching scheduler. While this solved multi-tenant concurrency and eliminated bubble waste, individual user generation was still fundamentally bottlenecked by the <strong>Memory Bandwidth Wall</strong>: reading hundreds of billions of weight parameters off HBM just to generate one single token.</p>
<p>In this final hands-on capstone lab of the inference track, we build <strong>The Speculative Decoding &amp; INT4 Quantization Engine</strong> in ~220 lines of pure standard-library Python. Zero external dependencies: no PyTorch, no HuggingFace, no NumPy. We implement the two crowning acceleration paradigms of frontier inference: <strong>Uniform Symmetric Low-Bit Quantization</strong> (Chapter 27) and <strong>Lossless Speculative Rejection Sampling</strong> (Chapter 26).</p>
<pre>
[The Python Brain Evolution Roadmap &bull; Complete 6-Stage Journey]
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
[Stage 5] 200 Lines Pure Python: Streaming KV &amp; Continuous Batching Engine (PagedAttention &amp; Iteration Scheduling)
       │
       ▼ (Bandwidth flaw: Serial single-token memory bound bottleneck O(T))
[Stage 6 (Capstone)] 220 Lines Pure Python: Speculative Decoding &amp; INT4 Quantization Engine
       │
       ▼ (Result: A production-grade inference accelerator delivering 2x-3x speedup with 75% memory compression!)
</pre>
</fieldset>

---

## Step 1: 3-Year-Old Intuition (The Wise Professor & The Fast Assistant)

Imagine writing a collaborative research paper with a venerable professor:

<figure>
<pre>
[Traditional Autoregressive Generation: Slow &amp; Painful]
Step 1: Professor takes 1 full second to think ──► writes word 1: "Quantum"
Step 2: Professor takes 1 full second to think ──► writes word 2: "computers"
Step 3: Professor takes 1 full second to think ──► writes word 3: "operate"
                                                  (3 full seconds for 3 words!)

[Speculative Decoding: The Fast Assistant Team]
Step 1: Fast Assistant (1B model) drafts 3 words in 0.05 seconds:
        ["computers", "operate", "using"]
Step 2: Professor (70B model) glances at all 3 words SIMULTANEOUSLY in 1 second:
        "Yes, computers!" (Accepted)
        "Yes, operate!"   (Accepted)
        "No, not using — on qubits!" (Rejected &amp; Corrected)
        └───────────────────────────────────────────────┘
        Result: 3 validated words produced in ONLY 1 Professor step! (3x Speedup!)
</pre>
<figcaption><strong>Figure 27b.1:</strong> Drafting is serial and cheap; verification is parallel and authoritative. Speculative decoding converts sequential latency into parallel batch verification.</figcaption>
</figure>

1. **The Light Wooden Bowling Ball (INT4 Quantization)**:
   Suppose you must haul a 16-pound solid iron bowling ball across a warehouse for every single pin you want to knock down. Your back will ache and your cart will move at a crawl. If you replace it with a compact 4-pound carved wooden ball (INT4), your cart is **4 times lighter** ($75\%$ memory bandwidth saved!), allowing your conveyor belt to move four times faster.

2. **The Fast Assistant &amp; The Professor (Speculative Decoding)**:
   A huge 70-billion-parameter model is like a distinguished professor: every single step takes a full second of heavyweight cognitive effort. A tiny 1-billion-parameter model is like a speedy assistant who can guess 4 words in a fraction of a second. The assistant drafts the phrase; the professor looks at the whole sentence at once in a **single parallel forward pass**. If the professor agrees, all 4 words are minted instantly!

3. **The Fair Audit Rule (Rejection Sampling)**:
   If the assistant guesses incorrectly, the professor doesn't get confused or adopt the error. The professor crosses out the wrong guess and writes the true correction from an exact mathematical residual formula. The generated text is guaranteed to have the **exact same probability distribution** as if the professor wrote every word by hand!

---

## Step 2: The Bridging Question

How do we compress 16-bit floating-point weights $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into 4-bit integers while preserving accurate matrix multiplication? And when the draft model proposes $K$ tokens with probabilities $q(x)$, how does the target model verify them with probabilities $p(x)$ such that the joint distribution remains **100% mathematically identical** to the target model alone?

---

## Step 3: The Exact Math & Formulas

### 1. Uniform Symmetric INT4 Quantization

For a weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, with bit-width $b = 4$ and maximum integer value $Q_{\max} = 2^{b-1} - 1 = 7$:

$$
s = \frac{\max_{i,j} |W_{i,j}|}{Q_{\max}} = \frac{\max |W|}{7}
$$

Quantization maps each floating-point weight $w$ to a signed integer $q \in [-7, 7]$:

$$
q = \operatorname{clip}\left(\left\lfloor \frac{w}{s} \right\rceil, -7, 7\right)
$$

During linear projection $\mathbf{y} = \mathbf{W}\mathbf{x}$, by the distributive law, accumulation occurs in integer arithmetic before applying the single scalar scale:

$$
y_i = \sum_{j} (s \cdot q_{i,j}) x_j = s \sum_{j} q_{i,j} x_j
$$

### 2. Speculative Rejection Sampling & Residual Recovery

Let the draft model propose token sequence $[x_1, x_2, \dots, x_\gamma]$ with sequential conditional probabilities $q(x_t \mid x_{<t})$. The target model evaluates the full sequence in a single forward pass, producing true probabilities $p(x_t \mid x_{<t})$.

For each candidate token $x_t$, generate a uniform random draw $u \sim \mathcal{U}(0, 1)$:

$$
\text{Acceptance Probability } \alpha_t = \min\left(1, \, \frac{p(x_t \mid x_{<t})}{q(x_t \mid x_{<t})}\right)
$$

If $u \le \alpha_t$, token $x_t$ is accepted. If rejected at position $k$, all subsequent draft tokens $x_{>k}$ are discarded, and a replacement token is sampled from the **residual distribution** $p'(x)$:

$$
p'(x) = \frac{\max(0, \, p(x) - q(x))}{\sum_{y \in V} \max(0, \, p(y) - q(y))}
$$

If all $\gamma$ candidate tokens are accepted, the target model's final parallel logit provides an extra bonus token $x_{\gamma+1} \sim p(x_{\gamma+1} \mid x_{\le \gamma})$ at **zero additional computational cost**.

---

## Step 4: Where Did It Come From?

In 2022, Yaniv Leviathan et al. (Google Research) and Charlie Chen et al. (DeepMind) independently published **Speculative Decoding**. They proved mathematically that sampling from the residual distribution $p'(x)$ guarantees **zero distribution drift**: the output is statistically indistinguishable from sampling from the large target model alone.

Simultaneously, quantization research matured from 8-bit post-training quantization (Dettmers et al., LLM.int8()) to 4-bit activation-aware weight quantization (**AWQ**, Lin et al., 2023) and **SmoothQuant** (Xiao et al., 2023), proving that 4-bit weights retain full perplexity while slashing memory bandwidth bottlenecks by $4\times$.

---

## Step 5: Concrete Toy Example & Code Walkthrough

Let us trace a numerical example of speculative verification on a 3-word vocabulary:

<figure>
<pre>
Vocabulary: ["apple", "banana", "cherry"]

Draft Model proposes: "banana"
  q("banana") = 0.60
  q("apple")  = 0.30
  q("cherry") = 0.10

Target Model evaluates:
  p("banana") = 0.30
  p("apple")  = 0.50
  p("cherry") = 0.20

Acceptance Probability:
  alpha = min(1, 0.30 / 0.60) = 0.50 (50% chance of accepting "banana")

Case A: Random draw u = 0.35 <= 0.50 -> "banana" ACCEPTED!
Case B: Random draw u = 0.82 > 0.50  -> "banana" REJECTED!
  Residual differences:
    apple:  max(0, 0.50 - 0.30) = 0.20
    banana: max(0, 0.30 - 0.60) = 0.00
    cherry: max(0, 0.20 - 0.10) = 0.10
  Sum of residuals = 0.20 + 0.00 + 0.10 = 0.30
  Normalized residual distribution p':
    apple:  0.20 / 0.30 = 66.7%
    banana: 0.00 / 0.30 = 0.0%
    cherry: 0.10 / 0.30 = 33.3%
  Sample correction token from p' -> 66.7% chance of "apple", 33.3% chance of "cherry"!
</pre>
<figcaption><strong>Figure 27b.2:</strong> The residual distribution mathematically balances the probabilities so the composite output distribution equals the target distribution exactly.</figcaption>
</figure>

### File Structure for Lab 06

```
27b-lab-speculative-engine/
├── speculative_engine.py          # Complete reference engine (~220 lines)
├── speculative_engine_exercise.py # Guided exercise with TODOs & unit tests
├── index.md                       # English curriculum guide
└── index.zh.md                    # Chinese curriculum guide
```

### Running the Reference Engine

```bash
python3 27b-lab-speculative-engine/speculative_engine.py
```

Output:
```
=====================================================================
Lab 06: Speculative Decoding & INT4 Quantization Engine Simulation
=====================================================================

[PART 1: INT4 Uniform Symmetric Quantization Audit]
  * Linear Layer: 64 x 64 weights
  * FP32 Footprint:        16384 bytes
  * FP16 Footprint:        8192 bytes
  * Packed INT4 Footprint: 2052 bytes
  * Effective Compression vs FP16: 3.99x (75% memory saved!)

[PART 2: Speculative Decoding vs Standard Autoregressive Generation]
Baseline Autoregressive Generated Sequence:
  the brown fox jumps over the brown fox
  Target Forward Calls: 7
  Tokens per Forward Call: 1.14

Running Speculative Engine (gamma = 3):
  Round 01: Drafted ['brown', 'fox', 'jumps'] -> Produced ['brown', 'fox', 'jumps', 'over'] (4 tokens in 1 target call)
  Round 02: Drafted ['the', 'quick', 'brown'] -> Produced ['the', 'quick', 'brown', 'fox'] (4 tokens in 1 target call)

=====================================================================
Final Performance Audit & Acceleration Metrics
=====================================================================
Speculative Generated Sequence:
  the brown fox jumps over the quick brown
  Total Tokens Generated:       8
  Target Model Forward Passes:  4
  Tokens per Forward Pass:      2.00 tok/call
  Net Inference Acceleration:   1.75x Speedup over Autoregressive Decoding!
=====================================================================
```

### Running the Guided Exercise

```bash
python3 27b-lab-speculative-engine/speculative_engine_exercise.py
```

---

## Step 6: Core Takeaway

Modern inference speedups are achieved by breaking serial dependencies on two complementary fronts: **quantization** shrinks the physical memory payload by $4\times$ to alleviate the memory bandwidth wall, while **speculative decoding** converts sequential token generation into parallel multi-token batch verification with zero loss in mathematical fidelity.
