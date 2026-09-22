# Chapter 18: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling)


## Step 1: 3-Year-Old Intuition (Freezing Ice, Bouncing Gas, & The Dynamic Club Bouncer) {: #step-1 }

!!! note "3-Year-Old Intuition: The Thermometer Knob and the Velvet Rope Bouncer"
    Imagine you have a music box that sings stories one word at a time. On the front of the box are two special control dials:

    1. **The Temperature Dial (<dfn id="def-temperature">Temperature, $T$</dfn>)**:
       - **Turned all the way down to Freezing ($T \to 0$)**:
         - The air inside the box turns to solid ice. Every particle freezes in place.
         - The music box has zero spontaneity: it strictly picks the single most predictable, safe word every single time.
         - Ask it a riddle, and it repeats the same sentence forever in an infinite loop: *"The dog ran, and the dog ran, and the dog ran..."*
       - **Turned to Pleasant Room Temperature ($T = 0.7$)**:
         - The ice melts into pleasant, warm air. Particles bounce with gentle, lively energy.
         - The music box picks mostly smart, logical words, but occasionally sprinkles in a delightful, creative metaphor. Its storytelling is natural, varied, and captivating!
       - **Turned to a Roaring Campfire ($T = 5.0$)**:
         - The air boils into superheated steam. Particles fly randomly in all directions.
         - The music box babbles chaotic nonsense: *"The purple bicycle slept upside-down pancake dinosaur!"*

    2. **The Club Bouncers (Top-$k$ and Top-$p$ Filters)**:
       - Even with a nice temperature, there is always a tiny risk that an absurd, nonsensical word (like <kbd>"toaster"</kbd> after *"Once upon a..."*) might slip through by pure statistical luck.
       - **The Strict Top-$k$ Bouncer**: Only allows the top $k$ most popular candidates into the VIP room (e.g. exactly the top 4 words). Everyone else is barred at the door.
       - **The Dynamic Top-$p$ Bouncer (Nucleus Sampling)**: Instead of counting people, this bouncer counts the crowd's total energy!
         - When the situation is obvious (like *"The capital of France is..."* where <kbd>"Paris"</kbd> has 99% of the vote), the VIP room shrinks to just **1 word** (<kbd>"Paris"</kbd>).
         - When the situation is wide open (like *"She opened the gift and saw a..."* where dozens of toys have small chances), the VIP room automatically widens to hold **40 words**!

    In Large Language Models, **Temperature, Top-$k$, and Top-$p$** are the mathematical filters that transform raw token probabilities into fluent, creative, and safe human conversation!

<figure>
<pre>
Token Logits to Final Sample Flow:

Raw Logits:           [ z_apple = 4.0, z_banana = 2.0, z_car = -3.0 ]
                               │
                               ▼  Divide by Temperature T
Scaled Logits:        [ z_i / T ]
                               │
                               ▼  Softmax Exponentiation
Probabilities:        [ p_apple = 0.84, p_banana = 0.15, p_car = 0.01 ]
                               │
                               ▼  Top-k / Top-p Truncation
Candidate Pool:       [ p_apple = 0.85, p_banana = 0.15 ]  (car is cut!)
                               │
                               ▼  Weighted Random Draw
Final Output Token:   "apple"
</pre>
<figcaption><strong>Figure 18.1:</strong> Logits are scaled by temperature $T$, converted to probabilities via Softmax, truncated by Top-$k$/Top-$p$ filters, and sampled.</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: Why Can't an LLM Just Pick the #1 Most Probable Word?"
    At every step of generation, the model computes an unembedding projection producing logits $\mathbf{z} \in \mathbb{R}^{|V|}$ across 32,000+ vocabulary tokens.

    The most obvious strategy is **Greedy Decoding**:

    $$
    w_{t}^* = \arg\max_{w \in V} z_w
    $$

    Why doesn't every LLM simply use Greedy Decoding?

    In 2019, Ari Holtzman and his co-authors at the University of Washington proved a startling mathematical discovery (\lt cite>"The Curious Case of Neural Text Degeneration"</cite>):
    1. **Greedy Text is Unnatural**: Natural human language does *not* consist of the highest-probability words at every step. Real human writers weave between high-probability predictable words and lower-probability creative, surprising words.
    2. **The Repetition Trap**: In deep autoregressive loops, greedy decoding gets trapped in deterministic loops: because word $A$ predicts $B$, and $B$ predicts $A$, the model gets caught in an infinite cycle ($A \to B \to A \to B$).

    *"How do we mathematically reshape the logit distribution using statistical thermodynamics, and how do we truncate the unreliable long tail of 30,000 improbable tokens without killing creativity?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. Temperature Scaling (Boltzmann Distribution)

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

### 2. Top-$k$ Truncation Filtering (Fan et al., 2018)

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

### 3. Top-$p$ Truncation Filtering (Nucleus Sampling; Holtzman et al., 2019)

Top-$k$ has a major flaw: a fixed $k=50$ is far too large when the model is confident (admitting 49 junk tokens), and too small when the context is genuinely ambiguous (excluding valid creative options).

**Top-$p$ (Nucleus) Sampling** solves this by dynamically adapting the cutoff threshold:
1. Sort all vocabulary tokens in descending order of probability:
   $$
   p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(|V|)}
   $$
2. Find the smallest index $k^*$ such that the cumulative distribution function (\lt abbr title="Cumulative Distribution Function">CDF</abbr>) reaches threshold $p \in (0, 1]$:
   $$
   k^* = \min \left\{ k : \sum_{i=1}^k p_{(i)} \ge p \right\}
   $$
3. Define the nucleus subset $V^{(p)} = \{ (1), (2), \dots, (k^*) \}$.
4. Re-normalize the probabilities strictly over $V^{(p)}$:
   $$
   p'_i = \begin{cases} \frac{p_i}{\sum_{j \in V^{(p)}} p_j} & \text{if } i \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}
   $$

\lt figure>
\lt pre>
Top-p Nucleus Truncation Dynamics:

Case A: Highly Confident Context ("The capital of France is...")
  Tokens:       [ Paris (0.97) │ Lyon (0.01) │ Marseille (0.01) │ ... ]
  CDF:            0.97 >= 0.90 ──► Cutoff! Pool size = 1 token!

Case B: Open Creative Context ("She looked outside and saw a...")
  Tokens:       [ bird (0.18) │ tree (0.15) │ cat (0.12) │ car (0.10) ... ]
  CDF:            0.18 + 0.15 + 0.12 + 0.10 + ... >= 0.90 ──► Pool = 18 tokens!
</pre>
\lt figcaption>\lt strong>Figure 18.2:</strong> Nucleus sampling dynamically contracts to 1 token when confident, and expands to dozens when ambiguous.</figcaption>
</figure>

---

## Step 4: Where Did It Come From? (Boltzmann, Fan, & Holtzman) {: #step-4 }

\lt dl>
  \lt dt>\lt time datetime="1877">1877</time> &mdash; \lt strong>Ludwig Boltzmann</strong></dt>
  \lt dd>Formulated statistical mechanics, showing that the probability of a physical system occupying microstate $i$ with energy $E_i$ at thermodynamic temperature $T$ follows $p_i \propto \exp(-E_i / k_B T)$. In LLMs, the negative logit $-z_i$ acts as the particle energy state.</dd>

  \lt dt>\lt time datetime="2018">2018</time> &mdash; \lt strong>Angela Fan, Mike Lewis, & Yann Dauphin</strong> (\lt cite>"Hierarchical Neural Story Generation"</cite>)</dt>
  \lt dd>Introduced Top-$k$ random sampling for neural text generation at Meta AI, proving that truncating the unreliable probability tail dramatically eliminated repetition and gibberish compared to unconstrained sampling.</dd>

  \lt dt>\lt time datetime="2019">2019</time> &mdash; \lt strong>Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, & Yejin Choi</strong> (\lt cite>"The Curious Case of Neural Text Degeneration"</cite>)</dt>
  \lt dd>Discovered that human language does not maximize probability, exposed the fatal flaws of both greedy decoding and fixed Top-$k$, and invented Nucleus (Top-$p$) Sampling, which remains the default decoding algorithm in ChatGPT, Claude, and Gemini.</dd>
</dl>

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

Let us calculate the exact probabilities for a miniature 4-word vocabulary under different temperatures, apply Top-$k$, and apply Top-$p$.

### 1. Miniature Vocabulary & Raw Logits
Consider vocabulary $V = \{\text{cat}, \text{dog}, \text{fish}, \text{toaster}\}$ with raw output logits:

$$
\mathbf{z} = [z_{\text{cat}} = 4.0, \; z_{\text{dog}} = 2.0, \; z_{\text{fish}} = 1.0, \; z_{\text{toaster}} = -1.0]
$$

---

### 2. Temperature Comparison ($T = 0.5$ vs. $T = 1.0$ vs. $T = 2.0$)

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

### 3. Truncation in Action: Top-$k$ and Top-$p$ at $T = 1.0$

Starting with the $T = 1.0$ probabilities:
- $\text{cat}: 0.8390$
- $\text{dog}: 0.1136$
- $\text{fish}: 0.0418$
- $\text{toaster}: 0.0057$

#### Applying Top-$k = 2$:
1. Retain only the top 2 candidates: $\{\text{cat}, \text{dog}\}$. Discard $\{\text{fish}, \text{toaster}\}$.
2. Sum of retained probabilities: $0.8390 + 0.1136 = \mathbf{0.9526}$.
3. Re-normalize:
   - $p'(\text{cat}) = 0.8390 / 0.9526 = \mathbf{0.8808} \; (88.1\%)$
   - $p'(\text{dog}) = 0.1136 / 0.9526 = \mathbf{0.1192} \; (11.9\%)$
   - $p'(\text{fish}) = \mathbf{0.0000}$
   - $p'(\text{toaster}) = \mathbf{0.0000}$

#### Applying Top-$p = 0.90$ (Nucleus):
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

!!! tip "Key Insight: The Punchline of Sampling Strategies"
    **Temperature controls the shape of the mountain, while Top-$k$ and Top-$p$ erect fences to keep the hiker away from dangerous cliffs.**

    By cooling or heating the logits via the Boltzmann distribution and dynamically trimming the improbable long tail, sampling transforms deterministic pattern matchers into natural, creative, and coherent conversational partners.
