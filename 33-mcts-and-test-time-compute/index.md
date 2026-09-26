# Chapter 33: Search, MCTS & Test-Time Compute Scaling

---

## Step 1: 3-Year-Old Intuition (The Chess Grandmaster Looking Ahead)

Imagine two different people playing a high-stakes game of chess:

1. **The Impulsive Novice (Standard Greedy LLM Generation)**:
   - The novice looks at the board and immediately grabs the piece that looks flashiest.
   - They don't look ahead at all. They just say: *"Ooh, my queen can move forward!"*
   - Five seconds later, their opponent traps the queen and the game is over.
   - This is standard LLM autoregressive greedy decoding: the model outputs the very first token that has the highest instantaneous probability, with zero ability to look into the future.

2. **The Grandmaster Closing Their Eyes (Search & MCTS)**:
   - The chess grandmaster sits with closed eyes, breathing slowly. Before touching a single piece, they explore a mental tree of possibilities:
     - **Branch A**: *"If I move my knight to E5, my opponent can pin my bishop. That leads to a dangerous bind."* $\to$ **Mental Red Flag! Discard Branch A.**
     - **Branch B**: *"If I push my pawn to D4, my opponent can sacrifice a rook and force a draw."* $\to$ **Discard Branch B.**
     - **Branch C**: *"If I trade bishops first, and then move my rook to C1, my opponent has no defense."* $\to$ **Golden Path!**
   - The grandmaster tests multiple moves, evaluates intermediate board positions using gut intuition (the Process Reward Model from Chapter 32), backs up from dead ends, and only plays the physical move once the entire path is verified.

3. **Trading Thinking Time for Superhuman Intelligence**:
   - If you give the grandmaster **1 second** to move (blitz chess), they make decent, intuitive moves.
   - If you give the grandmaster **10 minutes** to think, they find a stunning, counter-intuitive sacrifice that wins the world championship!
   - Their physical brain didn't get bigger in those 10 minutes. What scaled was **Test-Time Compute** &mdash; spending more computational effort thinking, exploring, and verifying before committing to an answer.

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                   THE MCTS TEST-TIME SEARCH PARADIGM                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                  [ Root State s_0 (Prompt) ]                           │
│                          /         \                                   │
│                         /           \                                  │
│              Step a_1  /             \  Step a_2 (Chosen via PUCT)     │
│                       ▼               ▼                                │
│                   [ Node s_1 ]     [ Node s_2 ]                        │
│                                      /    \                            │
│                          Step a_21  /      \  Step a_22                │
│                                    ▼        ▼                          │
│                               [ Leaf s_L ]  [ Dead End ] (PRM = 0.04)  │
│                                    │                                   │
│                                    ▼                                   │
│                           PRM Evaluation: V(s_L)                       │
│                                    │                                   │
│                                    ▼                                   │
│                  Backpropagate Value &amp; Update Counts:                  │
│                     N(s, a) += 1,  Q(s, a) += ΔQ                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 33.1:</strong> Monte Carlo Tree Search on language reasoning: Balances exploration and exploitation via PUCT, evaluates intermediate states via PRMs, and updates tree statistics.</figcaption>
</figure>

---

## Step 2: The Bridging Question

In Chapters 00 to 28, the capabilities of Large Language Models were dictated by the **Pre-training Scaling Laws** (Kaplan et al., Chinchilla): to make a model smarter, one had to double the model parameters $N$ or quadruple the pre-training tokens $D$.
However, pre-training scaling is colliding with physical reality:
1. **The Web Data Exhaustion Wall**: Human society has already fed almost all high-quality public text, books, and code into frontier models.
2. **The Gigawatt Power Wall**: Training clusters are consuming hundreds of megawatts of electrical power, requiring hundreds of millions of dollars per run.

The bridging question is:
$$\text{How do we mathematically scale inference-time compute (FLOPs) &mdash; through Best-of-}N\text{, Majority Voting, and Monte Carlo Tree Search (MCTS) &mdash; to achieve orders-of-magnitude reasoning gains without retraining the base model parameters?}$$

---

## Step 3: The Exact Math & Formula

### 1. The Test-Time Compute Scaling Law

Let inference compute $C_{\text{test}}$ be defined as the total floating-point operations invested into exploring reasoning trajectories for a single query $x$:

$$
C_{\text{test}} \approx N_{\text{rollouts}} \times T_{\text{length}} \times \text{FLOPs}_{\text{model}}
$$

Empirical research (Snell et al., OpenAI o1) demonstrates that reasoning benchmark accuracy scales log-linearly with test-time compute expenditure:

$$
\text{Error Rate}(C_{\text{test}}) \propto C_{\text{test}}^{-\alpha}
$$

where $\alpha > 0$ is the test-time scaling exponent. A 7-billion parameter model given sufficient test-time compute can soundly defeat a 70-billion parameter model running single-pass greedy decoding!

---

### 2. Self-Consistency and Majority Voting

The simplest form of test-time search is **Self-Consistency (Majority Voting)**:
1. Sample $N$ independent reasoning paths $\{y_1, y_2, \dots, y_N\} \sim \pi_{\boldsymbol{\theta}}(x)$ using temperature sampling ($\tau \approx 0.7$).
2. Extract the final discrete answer $\operatorname{ans}(y_i) \in \mathcal{Y}$.
3. Select the answer with the highest frequency:
   $$
   \hat{y} = \arg\max_{a \in \mathcal{Y}} \sum_{i=1}^N \mathbb{I}\left(\operatorname{ans}(y_i) = a\right)
   $$

#### Mathematical Bounds via the Binomial Distribution
Suppose the base model has a per-sample probability $p$ of reaching the correct discrete answer. If $p > 0.5$, the probability that the majority of $N$ independent odd samples ($N = 2m + 1$) is correct follows the regularized incomplete beta function:

$$
P_N(\text{Majority Correct}) = \sum_{k=\frac{N+1}{2}}^N \binom{N}{k} p^k (1-p)^{N-k} = I_p\left(\frac{N+1}{2}, \frac{N+1}{2}\right)
$$

By the Condorcet Jury Theorem, as $N \to \infty$:

$$
\lim_{N \to \infty} P_N(\text{Majority Correct}) = 1.0 \quad (\text{for any } p > 0.5)
$$

The error rate decays exponentially with sample budget $N$:

$$
P_N(\text{Error}) \le \exp\left( -2 N \left(p - \frac{1}{2}\right)^2 \right)
$$

---

### 3. Monte Carlo Tree Search (MCTS) on Language Reasoning Steps

When problems require dozens of sequential deductive steps, independent sampling wastes enormous compute regenerating identical prefixes.
We structure test-time search as a tree $(\mathcal{V}, \mathcal{E})$ where each node represents a partial reasoning state $s_t = (x, s_1, \dots, s_t)$ and each edge represents a reasoning step $a_t = s_{t+1}$.

Each state-action edge $(s, a)$ maintains four statistics:
- **$N(s, a)$**: Visit count.
- **$W(s, a)$**: Total accumulated value.
- **$Q(s, a)$**: Mean state-action value: $Q(s, a) = \frac{W(s, a)}{N(s, a)}$.
- **$P(s, a)$**: Prior policy probability: $P(s, a) = \pi_{\boldsymbol{\theta}}(a \mid s)$.

MCTS iterates through four fundamental phases:

#### Phase 1: Selection via PUCT
Starting at root node $s_0$, descend the tree by choosing action $a^*$ that maximizes the **Predictor Upper Confidence Bound applied to Trees (<abbr title="Predictor Upper Confidence Bound applied to Trees">PUCT</abbr>)**:

$$
a^* = \arg\max_a \left[ Q(s, a) + U(s, a) \right]
$$

where the exploration bonus $U(s, a)$ is defined as:

$$
U(s, a) = c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_{b} N(s, b)}}{1 + N(s, a)}
$$

where $c_{\text{puct}} \approx 1.5$ balances exploitation (high $Q$) and exploration (high prior $P$, low visits $N$).

#### Phase 2: Expansion
Upon reaching a leaf node $s_L$, generate $K$ candidate next reasoning steps using the policy model:

$$
\{a_1, a_2, \dots, a_K\} \sim \pi_{\boldsymbol{\theta}}(\cdot \mid s_L)
$$

Initialize each new edge with $N(s_L, a_k) = 0, W(s_L, a_k) = 0, Q(s_L, a_k) = 0$.

#### Phase 3: Evaluation
Evaluate the quality of the leaf node $s_L$ using a Process Reward Model (Chapter 32) or terminal rollout verifier:

$$
V(s_L) = \sigma\left(f_{\boldsymbol{\psi}}(s_L)\right) \in [0, 1]
$$

#### Phase 4: Backpropagation
Ascend the tree from $s_L$ back to root $s_0$. For every traversed edge $(s, a)$, update its visit count and value:

$$
N(s, a) \leftarrow N(s, a) + 1, \quad W(s, a) \leftarrow W(s, a) + V(s_L), \quad Q(s, a) = \frac{W(s, a)}{N(s, a)}
$$

---

### 4. Autonomous Reasoning Phenomena in Extended Thinking

When models are trained under search-driven reinforcement learning (such as DeepSeek-R1 or OpenAI o1), two remarkable phenomena emerge during test-time generation:

1. **Thinking Token Expansion**: The model autonomously allocates thousands of internal tokens inside `<think>...</think>` tags to explore, simulate, and verify alternative hypotheses before producing the final `<answer>`.
2. **Spontaneous Metacognitive Backtracking**: Without human demonstrations, the policy generates self-monitoring trigger phrases:
   - <samp>"Wait, let me double check that assumption..."</samp>
   - <samp>"Actually, plugging x=3 back into equation 1 yields a contradiction. Let me rethink..."</samp>
   These markers indicate that the model is performing internal tree search directly in token space!

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2006">2006</time> &mdash; <strong>Rémi Coulom, Levente Kocsis, & Csaba Szepesvári</strong></dt>
  <dd>Introduced <em>Monte Carlo Tree Search (MCTS)</em> and the <em>UCT</em> algorithm, revolutionizing computer Go by replacing brute-force minimax search with statistical exploration-exploitation bounds.</dd>

  <dt><time datetime="2016–2017">2016–2017</time> &mdash; <strong>David Silver et al. (AlphaGo &amp; AlphaZero, DeepMind)</strong></dt>
  <dd>Formulated the modern <strong>PUCT</strong> equation, marrying deep policy networks (for action priors $P(s,a)$) with value networks ($Q(s,a)$) and tree search, achieving superhuman mastery over Go, Chess, and Shogi.</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>Xuezhi Wang et al. (Google Research)</strong></dt>
  <dd>Introduced <em>Self-Consistency</em> in Chain-of-Thought reasoning, proving that sampling diverse reasoning paths and taking the majority vote delivered double-digit accuracy leaps across mathematical benchmarks.</dd>

  <dt><time datetime="2024–2025">2024–2025</time> &mdash; <strong>The Inference Compute Paradigm Shift (OpenAI o1 &amp; DeepSeek-R1)</strong></dt>
  <dd>Marked the transition from pure pre-training scaling to <strong>Test-Time Compute Scaling</strong>. DeepSeek-R1 proved that pure reinforcement learning (GRPO) directly incentivizes models to expand internal search trees, solving Olympiad-level mathematics and competitive coding.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace a concrete PUCT selection and backpropagation calculation by hand with exact numbers.

### The Search Tree State at Root $s_0$
Suppose our search tree is at root state $s_0$ (the initial problem prompt).
- Exploration hyperparameter: $c_{\text{puct}} = 1.414$ ($\approx \sqrt{2}$).
- Total root visit count: $\sum_{b} N(s_0, b) = 10$.
- The model policy proposed two candidate next steps:
  - **Candidate Step $a_1$**: High prior probability, heavily visited.
    - Prior: $P(s_0, a_1) = 0.60$
    - Visit count: $N(s_0, a_1) = 8$
    - Accumulated value: $W(s_0, a_1) = 5.60$
    - Mean value: $Q(s_0, a_1) = \frac{5.60}{8} = 0.70$
  - **Candidate Step $a_2$**: Lower prior probability, lightly explored.
    - Prior: $P(s_0, a_2) = 0.40$
    - Visit count: $N(s_0, a_2) = 2$
    - Accumulated value: $W(s_0, a_2) = 1.30$
    - Mean value: $Q(s_0, a_2) = \frac{1.30}{2} = 0.65$

Which branch will the PUCT tree policy select for the next simulation?

---

### Phase 1: Evaluating the PUCT Selection Formula

The parent visit root term is:
$$
\sqrt{\sum_b N(s_0, b)} = \sqrt{10} \approx 3.1623
$$

#### 1. PUCT Score for Candidate Step $a_1$:
$$
\begin{aligned}
U(s_0, a_1) &= c_{\text{puct}} \cdot P(s_0, a_1) \cdot \frac{\sqrt{10}}{1 + N(s_0, a_1)} \\
&= 1.414 \times 0.60 \times \frac{3.1623}{1 + 8} \\
&= 0.8484 \times \frac{3.1623}{9} \\
&= 0.8484 \times 0.3514 \approx \mathbf{0.2981}
\end{aligned}
$$
Total selection score:
$$
\operatorname{PUCT}(s_0, a_1) = Q(s_0, a_1) + U(s_0, a_1) = 0.70 + 0.2981 = \mathbf{0.9981}
$$

#### 2. PUCT Score for Candidate Step $a_2$:
$$
\begin{aligned}
U(s_0, a_2) &= c_{\text{puct}} \cdot P(s_0, a_2) \cdot \frac{\sqrt{10}}{1 + N(s_0, a_2)} \\
&= 1.414 \times 0.40 \times \frac{3.1623}{1 + 2} \\
&= 0.5656 \times \frac{3.1623}{3} \\
&= 0.5656 \times 1.0541 \approx \mathbf{0.5962}
\end{aligned}
$$
Total selection score:
$$
\operatorname{PUCT}(s_0, a_2) = Q(s_0, a_2) + U(s_0, a_2) = 0.65 + 0.5962 = \mathbf{1.2462}
$$

<fieldset>
<legend><strong>The Power of PUCT</strong></legend>
<p>Look at the selection decision:</p>
<p>$\operatorname{PUCT}(s_0, a_2) = \mathbf{1.2462} > \operatorname{PUCT}(s_0, a_1) = \mathbf{0.9981}$.</p>
<p>Even though Candidate Step $a_1$ has a higher prior probability ($0.60 > 0.40$) and a higher mean value ($0.70 > 0.65$), the search tree chooses <strong>Candidate Step $a_2$</strong> because its low visit count ($N=2$) creates a massive exploration bonus ($0.5962$ vs $0.2981$)! The algorithm refuses to prematurely lock into familiar paths.</p>
</fieldset>

---

### Phase 2: Leaf Evaluation & Backpropagation

Following edge $(s_0, a_2)$ leads to a leaf node $s_L$.
The Process Reward Model (PRM) evaluates $s_L$ and discovers that it represents a brilliant mathematical insight:

$$
V(s_L) = \mathbf{0.90}
$$

We backpropagate this evaluation up the tree to update edge $(s_0, a_2)$:
1. Increment visit count:
   $$
   N_{\text{new}}(s_0, a_2) = N_{\text{old}} + 1 = 2 + 1 = \mathbf{3}
   $$
2. Update accumulated value:
   $$
   W_{\text{new}}(s_0, a_2) = W_{\text{old}} + V(s_L) = 1.30 + 0.90 = \mathbf{2.20}
   $$
3. Update mean state-action value:
   $$
   Q_{\text{new}}(s_0, a_2) = \frac{W_{\text{new}}}{N_{\text{new}}} = \frac{2.20}{3} \approx \mathbf{0.7333}
   $$

<mark>$Q(s_0, a_2)$ jumped from $0.6500$ to $0.7333$, now surpassing Candidate Step $a_1$ ($0.7000$)!</mark>
By exploring an under-visited branch, the search algorithm discovered a superior reasoning pathway that greedy decoding would have permanently ignored.

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>Test-Time Compute Scaling represents the second scaling frontier of artificial intelligence: by replacing single-pass greedy token generation with Monte Carlo Tree Search (<abbr title="Monte Carlo Tree Search">MCTS</abbr>) and majority verification, Large Language Models convert additional inference FLOPs into superhuman reasoning &mdash; exploring branching hypotheses, pruning dead ends, and autonomously discovering verified truths.</p>
</fieldset>
