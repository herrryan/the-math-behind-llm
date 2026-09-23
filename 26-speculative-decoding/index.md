# Chapter 26: Breaking the Sequential Barrier (Speculative Decoding & Verification Math)

> [!INTUITION] Step 1: 3-Year-Old Intuition
> Imagine a brilliant, world-famous professor and an energetic, fast-talking young apprentice.
> 
> The professor (**Target Model - 70B**) is a genius, but is very old and walks very slowly. Every single word the professor says takes a full minute of deep contemplation.
> 
> The apprentice (**Draft Model - 1B**) is not quite as wise, but speaks at 100 miles per hour. In just 1 second, the apprentice blurts out 4 quick guesses:
> *"The... cat... sat... on..."*
> 
> Now, instead of thinking about one word at a time, the professor takes one single slow breath, looks at all 4 guesses together in parallel, and checks them:
> - Word 1 (*"The"*): *"Yes, I agree."*
> - Word 2 (*"cat"*): *"Yes, I agree."*
> - Word 3 (*"sat"*): *"Yes, I agree."*
> - Word 4 (*"on"*): *"Yes, I agree! And I will add the next word: 'the'."*
> 
> In that one single minute where the professor normally produces only 1 word, the team just produced **5 verified words**!
> 
> And here is the magic trick: If the apprentice makes a mistake, the professor catches it, crosses it out, and writes the correct word. The final published essay is **100% mathematically indistinguishable from an essay written by the genius professor working alone**.

---

## Step 2: The Bridging Question

How do we convert this apprentice-professor partnership into parallel tensor verification, and what exact probability formula guarantees that accepting and rejecting speculative tokens produces the **exact same probability distribution** as the massive target model?

Autoregressive generation is strictly sequential by definition:
$$x_t \sim P(X_t \mid x_1, \dots, x_{t-1})$$
You cannot evaluate step $t+1$ until token $x_t$ is chosen. This sequential dependency is why decode is bounded by memory bandwidth.

However, **verification is completely parallel**!
If a small, cheap draft model speculatively drafts $\gamma$ tokens ahead:
$$\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_\gamma$$
the large target model can ingest all $\gamma$ tokens in a single parallel **Prefill-style forward pass**!

The bridging question is:
$$\text{How do we formulate an acceptance/rejection rule such that after filtering the draft tokens, the sampled sequence has the exact probability distribution } p(x) \text{ of the target model with zero quality degradation?}$$

---

## Step 3: The Exact Math & Formula

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SPECULATIVE DECODING VERIFICATION PIPELINE                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Draft Model (1B, Fast):                                                                │
│   Autoregressively generates γ = 4 draft tokens: [ x_1, x_2, x_3, x_4 ]                │
│   Draft probabilities: q(x_1), q(x_2), q(x_3), q(x_4)                                  │
│                                                                                        │
│ Target Model (70B, Slow &amp; Powerful):                                                  │
│   Runs 1 single parallel forward pass over all 4 tokens simultaneously (Prefill-style) │
│   Target probabilities: p(x_1), p(x_2), p(x_3), p(x_4)                                  │
│                                                                                        │
│ Speculative Rejection Sampler:                                                         │
│   Token 1: Accept with prob min(1, p(x_1)/q(x_1)) ──► ACCEPTED!                        │
│   Token 2: Accept with prob min(1, p(x_2)/q(x_2)) ──► ACCEPTED!                        │
│   Token 3: Accept with prob min(1, p(x_3)/q(x_3)) ──► REJECTED!                        │
│            ──► Sample correction token x_3* from max(0, p - q)                         │
│            ──► Discard remaining draft token x_4                                       │
│ RESULT: 3 tokens accepted/generated in the time of 1 target forward pass!              │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 26.1:</strong> The Speculative Decoding pipeline. A small draft model speculates γ tokens, which the large target model evaluates in parallel. Rejected tokens are corrected via residual sampling.</figcaption>
</figure>

### 1. The Speculative Acceptance Criterion

Let:
- $M_p$: The large target model with output distribution $p(x) = P_{M_p}(x \mid x_{<t})$.
- $M_q$: The small draft model with output distribution $q(x) = P_{M_q}(x \mid x_{<t})$.

For a speculative candidate token $x \sim q(x)$, draw a uniform random number $u \sim U(0, 1)$.
The token is **accepted** if:

$$
u \le \min\left(1, \; \frac{p(x)}{q(x)}\right)
$$

### 2. The Residual Recovery Distribution

If candidate token $x$ is rejected at index $i$, all subsequent draft tokens $x_{i+1}, \dots, x_\gamma$ are discarded.
We do not waste the forward pass! We sample a replacement token $x^*$ directly from the normalized residual distribution:

$$
p'(x) = \frac{\max\left(0, \; p(x) - q(x)\right)}{\sum_{x'} \max\left(0, \; p(x') - q(x')\right)}
$$

---

### 3. Mathematical Proof of Exact Lossless Distribution

We now prove that the probability of generating token $x$ under this speculative rejection protocol is **identically equal to $p(x)$**.

The output token $X$ is produced either by being drafted and accepted, or by being sampled from $p'(x)$ upon rejection:

$$
P(X = x) = P(\text{Accepted } x) + P(\text{Rejected}) \cdot p'(x)
$$

#### Term 1: Acceptance Probability
$$
P(\text{Accepted } x) = q(x) \cdot \min\left(1, \; \frac{p(x)}{q(x)}\right) = \min\left(q(x), \; p(x)\right)
$$

#### Term 2: Rejection Probability
The probability of rejecting the draft token is:
$$
P(\text{Rejected}) = 1 - \sum_{x'} \min\left(q(x'), \; p(x')\right) = \sum_{x'} \left(p(x') - \min(q(x'), p(x'))\right)
$$
Using the identity $a - \min(b, a) = \max(0, a - b)$:
$$
P(\text{Rejected}) = \sum_{x'} \max\left(0, \; p(x') - q(x')\right)
$$

#### Term 3: Combining the Terms
Notice that the denominator of $p'(x)$ cancels perfectly with $P(\text{Rejected})$:
$$
P(\text{Rejected}) \cdot p'(x) = \left(\sum_{x'} \max(0, p(x') - q(x'))\right) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))} = \max\left(0, \; p(x) - q(x)\right)
$$

Adding Term 1 and Term 2:
$$
P(X = x) = \min\left(q(x), \; p(x)\right) + \max\left(0, \; p(x) - q(x)\right)
$$
Using the algebraic identity $\min(a, b) + \max(0, b - a) = b$:
$$
P(X = x) = p(x) \quad \blacksquare
$$

The mathematical output distribution of speculative decoding is **strictly identical to native sampling from the target model**!

---

### 4. Expected Tokens and Speedup Formulation

Let $\beta$ denote the <dfn id="def-acceptance">Expected Acceptance Rate</dfn>:
$$
\beta = \mathbb{E}_{x \sim q}\left[\min\left(1, \; \frac{p(x)}{q(x)}\right)\right] = 1 - \frac{1}{2} \sum_{x} |p(x) - q(x)| = 1 - D_{\text{TV}}(p, q)
$$
where $D_{\text{TV}}(p, q)$ is the Total Variation Distance between the draft and target models.

If the draft model proposes $\gamma$ tokens, the expected number of generated tokens per cycle $\mathbb{E}[\tau]$ is:

$$
\mathbb{E}[\tau] = \frac{1 - \beta^{\gamma + 1}}{1 - \beta}
$$

Let $t_q$ be the execution time of one draft forward pass, and $t_p$ be the time of one target forward pass. The theoretical speedup $S$ is:

$$
S = \frac{\mathbb{E}[\tau]}{\frac{t_q}{t_p} \cdot \gamma + 1}
$$

When $\beta \approx 0.8$ and $\gamma = 5$ with a small draft model ($t_q / t_p \approx 0.05$):
$$\mathbb{E}[\tau] \approx 3.7 \text{ tokens}, \quad S = \frac{3.7}{(0.05 \times 5) + 1} = \frac{3.7}{1.25} \approx \mathbf{2.96\times \text{ Wall-Clock Speedup!}}$$

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2023">2023</time> &mdash; <strong>Speculative Decoding</strong> (<cite>Yaniv Leviathan, Matan Kalman, Yossi Matias, Google Research, ICML 2023</cite>)</dt>
  <dd>Introduced speculative sampling with rigorous proof of lossless distribution preservation, demonstrating 2x-3x speedups on T5 and PaLM.</dd>
  <dt><time datetime="2023">2023</time> &mdash; <strong>Speculative Sampling</strong> (<cite>Charlie Chen et al., DeepMind</cite>)</dt>
  <dd>Independently developed speculative sampling with identical mathematical proofs for Chinchilla 70B.</dd>
  <dt><time datetime="2024">2024</time> &mdash; <strong>Medusa &amp; Eagle: Draft-Model-Free Speculation</strong> (<cite>Cai et al., Together AI</cite>)</dt>
  <dd>Eliminated the need for a separate draft model by training lightweight multi-head prediction layers directly on top of the target model's final hidden states.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace exact numbers on a tiny 3-word vocabulary: $V = \{\text{"apple"}, \text{"banana"}, \text{"cherry"}\}$.

### Model Probabilities
- Draft Model $q$:
  $$q(\text{"apple"}) = 0.60, \quad q(\text{"banana"}) = 0.30, \quad q(\text{"cherry"}) = 0.10$$
- Target Model $p$:
  $$p(\text{"apple"}) = 0.40, \quad p(\text{"banana"}) = 0.50, \quad p(\text{"cherry"}) = 0.10$$

---

### Step 1: Draft Candidate Selection
Suppose the draft model samples candidate: $x = \text{"apple"}$.

We look up the probabilities:
$$q(\text{"apple"}) = 0.60, \quad p(\text{"apple"}) = 0.40$$

### Step 2: Compute Acceptance Probability
$$
\alpha = \min\left(1, \; \frac{p(\text{"apple"})}{q(\text{"apple"})}\right) = \min\left(1, \; \frac{0.40}{0.60}\right) = \frac{2}{3} \approx 0.667
$$

- We roll a random number $u \in [0, 1]$.
- If $u \le 0.667$: **Accepted!** We output `"apple"`.
- If $u > 0.667$: **Rejected!** We proceed to residual recovery.

---

### Step 3: Compute the Residual Distribution $p'(x)$
Suppose $u = 0.85$ (Rejected!).
We compute $\max(0, p(x) - q(x))$ for all words in vocabulary:
- `"apple"`: $\max(0, 0.40 - 0.60) = \max(0, -0.20) = 0$
- `"banana"`: $\max(0, 0.50 - 0.30) = +0.20$
- `"cherry"`: $\max(0, 0.10 - 0.10) = 0$

$$\text{Sum} = 0 + 0.20 + 0 = 0.20$$

We normalize to create $p'(x)$:
- $p'(\text{"apple"}) = \frac{0}{0.20} = 0$
- $p'(\text{"banana"}) = \frac{0.20}{0.20} = 1.00$
- $p'(\text{"cherry"}) = \frac{0}{0.20} = 0$

Upon rejecting `"apple"`, the model is mathematically forced to output `"banana"` with 100% certainty!

---

### Step 4: Verification of Total Marginal Probability
Let us calculate the net probability of generating `"apple"`:
$$
P(X = \text{"apple"}) = q(\text{"apple"}) \cdot \alpha + P(\text{Reject}) \cdot p'(\text{"apple"}) = 0.60 \times \frac{0.40}{0.60} + 0 = 0.40 = p(\text{"apple"})!
$$
And for `"banana"`:
$$
P(X = \text{"banana"}) = q(\text{"banana"}) \cdot \alpha_{\text{banana}} + P(\text{Reject}) \cdot p'(\text{"banana"}) = 0.30 \times 1.0 + (0.20 \times 1.00) = 0.50 = p(\text{"banana"})!
$$

The final distribution matches the target model with absolute mathematical perfection!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p><strong>Speculative Decoding</strong> breaks the strict sequential curse of autoregressive decoding by exploiting a fundamental asymmetry in computation: <strong>Drafting is sequential, but Verification is completely parallel</strong>.</p>
<p>By pairing a fast draft model with a mathematically exact residual acceptance sampler, modern inference engines generate 2 to 4 tokens per target forward pass with zero loss in model accuracy.</p>
</fieldset>
