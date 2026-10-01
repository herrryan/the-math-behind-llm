# Chapter 32: Process Reward Models (PRMs) & Step-Level Verification

---

## Step 1: 3-Year-Old Intuition (The Master Chef Inspecting the Soup at Every Step)

Imagine an apprentice chef cooking a complex French soup that requires four hours and twenty delicate steps:

1. **The Outcome-Only Food Critic (The ORM)**:
   - The food critic sits in the dining room and refuses to look inside the kitchen.
   - Four hours later, the apprentice brings out the finished bowl. The critic takes one tiny sip, grimaces, and shouts: *"Horrible! 0 out of 10!"*
   - Think about what happens now:
     - Did the apprentice burn the onions in Step 1?
     - Did they add spoiled cream in Step 18?
     - Or did they do nineteen steps brilliantly, but accidentally drop in a teaspoon of sand at the very end?
   - The apprentice has no idea! Under **Outcome-supervised Reward Models (<abbr title="Outcome Reward Model">ORM</abbr>)**, all twenty cooking steps receive the exact same devastating blame.

2. **The Master Chef Standing by the Pot (The PRM)**:
   - Now imagine a world-class master chef standing directly beside the stove:
     - **Step 1 (Caramelizing onions)**: The master chef tastes the pan $\to$ *"Golden, sweet, and perfectly cooked! Step 1 passed (Score: 0.98)!"*
     - **Step 2 (Deglazing with white wine)**: Tastes the reduction $\to$ *"Vibrant acidity, alcohol evaporated! Step 2 passed (Score: 0.95)!"*
     - **Step 3 (Adding stock)**: Tastes the broth $\to$ *"Wait! You used cheap, over-salted bouillon cubes instead of fresh veal stock! Step 3 failed (Score: 0.04)!"*
   - The apprentice knows **instantly and precisely** where the recipe derailed!
   - Steps 1 and 2 keep their gold stars. The apprentice does not throw away the whole kitchen; they simply dump Step 3, pour in fresh veal stock, and continue cooking on the winning path.

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│             OUTCOME REVISION (ORM) vs. PROCESS REVISION (PRM)          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Prompt [x] ──► Step 1 ──► Step 2 ──► Step 3 (Error!) ──► Step 4      │
│                    │          │              │               │         │
│                    │          │              │               ▼         │
│                    │          │              │          Final Answer   │
│                    │          │              │               │         │
│                    ▼          ▼              ▼               ▼         │
│   ORM Evaluation:  ?          ?              ?          Score = 0.0    │
│   (All steps penalized equally; zero diagnostic guidance)             │
│                                                                        │
│   PRM Evaluation: r_1=0.98   r_2=0.95     r_3=0.04        r_4=0.10     │
│   (Pinpoints exact step of failure; enables search pruning &amp; rewind!)  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 32.1:</strong> Process Reward Models evaluate intermediate reasoning tokens, isolating logical fallacies at the exact step they occur.</figcaption>
</figure>

---

## Step 2: The Bridging Question

In Chapters 30 and 31, our reward signals were evaluated purely on the final terminal output ($R(x, y)$).
However, for multi-step reasoning (e.g., 30-step mathematical proofs, 100-line software algorithms), outcome supervision suffers from two fatal mathematical vulnerabilities:

1. **The Credit Assignment Dilemma**: If a model generates 25 correct deductive steps, makes an arithmetic sign slip at Step 26, and reaches the wrong boxed answer, the terminal reward is $R = 0$. Outcome supervision assigns negative credit to all 25 flawless logical deductions, actively unlearning valid mathematical principles!
2. **The False-Positive Illusion (Lucky Bugs)**: A model can make two compounding, illegal logical errors that accidentally cancel out, arriving at the correct final integer by sheer luck. An Outcome Reward Model assigns this solution a perfect $+1.0$ score, strongly reinforcing hallucinated logic!

The central bridging question is:
$$\text{How do we mathematically formulate step-level verification, estimate intermediate validity without expensive per-step human annotations, and use Process Reward Models to prune search trees?}$$

---

## Step 3: The Exact Math & Formula

### 1. Step-Level Reasoning Decomposition

Let an input question be $x$. A generative model produces a chain-of-thought completion $y$ partitioned into $K$ distinct sequential reasoning steps separated by newline delimiters ($\texttt{\textbackslash n\textbackslash n}$):

$$
y = (s_1, s_2, \dots, s_K)
$$

where $s_k$ is the sequence of tokens comprising the $k$-th reasoning step.

---

### 2. Outcome Reward Model (ORM) vs. Process Reward Model (PRM)

- **Outcome Reward Model ($R_{\text{ORM}}$)**: A scalar classifier operating exclusively on the complete sequence:
  $$
  r_{\text{ORM}}(x, y) = \sigma\left( f_{\boldsymbol{\theta}}(x, s_1, \dots, s_K) \right) \in [0, 1]
  $$
- **Process Reward Model ($R_{\text{PRM}}$)**: A classifier parameterized by weights $\boldsymbol{\psi}$ that evaluates the correctness of each individual step $s_k$ conditioned on the question and all preceding steps:
  $$
  r_k = \sigma\left( f_{\boldsymbol{\psi}}(x, s_1, s_2, \dots, s_k) \right) \in [0, 1]
  $$

The **Joint Validity** of the complete reasoning path under the independence assumption is:

$$
P(\text{Path Valid}) = \prod_{k=1}^K r_k
$$

If a single catastrophic error occurs at step $j$ ($r_j \approx 0$), the entire path validity collapses to zero:

$$
r_j \to 0 \implies \prod_{k=1}^K r_k \to 0
$$

---

### 3. Automated Step Annotation via Monte Carlo Rollouts (Math-Shepherd)

Hiring human mathematicians to annotate millions of individual intermediate steps is prohibitively expensive.
The **Math-Shepherd** algorithm automates step supervision using pure counting and Monte Carlo rollouts:

For a given problem $x$ and prefix $s_{1:k} = (s_1, \dots, s_k)$, the system freezes the first $k$ steps on the chalkboard. It then asks the generator model to sample $M$ independent completions to the very end:

$$
\{c_1, c_2, \dots, c_M\} \sim \pi_{\text{gen}}\left(\cdot \mid x, s_{1:k}\right)
$$

Each completed rollout is checked by an automated deterministic verifier $v(c_m) \in \{0, 1\}$ (such as Python's `math` module or SymPy checking the boxed integer).

<fieldset>
<legend><strong>The Feynman View: Counting Winning Marbles from Where You Stand</strong></legend>
<p>Imagine you are hiking up a mountain trail and reach a fork in the road (Step $k$):</p>
<ul>
  <li>You freeze where you stand and send $M = 10$ scouts running down the mountain from your exact location.</li>
  <li>If $8$ scouts successfully reach the summit, then $8$ out of $10$ paths succeeded! The empirical quality of your current location on the number line is:
    $$
    V^*(x, s_{1:k}) = \frac{8}{10} = 0.80
    $$
  </li>
  <li>Now take one step forward along Fork A ($s_{k+1}$). Send $10$ scouts again. $9$ scouts reach the summit ($V^* = 0.90$). The path is healthy!</li>
  <li>Take one step forward along Fork B ($s'_{k+1}$, where you wrote $2x = 8 \implies x = 5$). Send $10$ scouts. <strong>$0$ out of $10$ scouts can ever reach the summit</strong> because the arithmetic is mathematically broken! Its empirical value collapses: $V^* = \frac{0}{10} = 0.00$.</li>
</ul>
<p>The difference $\Delta V^* = 0.00 - 0.80 = -0.80$ immediately exposes the fatal blunder. No human expert is needed &mdash; just simple counting of successful arrivals!</p>
</fieldset>

The empirical value of state $s_{1:k}$ is the fraction of winning rollouts:

$$
V^*(x, s_{1:k}) = \frac{1}{M} \sum_{m=1}^M v(c_m) \in [0, 1]
$$

The step ground-truth label $y_k$ is assigned by comparing the change in value across steps:

$$
y_k = \begin{cases}
1 & \text{if } V^*(x, s_{1:k}) \ge \tau \quad (\text{promising step that preserves solvability}) \\
0 & \text{if } V^*(x, s_{1:k}) < \tau \quad (\text{fatal step leading to dead ends})
\end{cases}
$$

where $\tau \in (0, 1)$ is a validity threshold (typically $\tau = 0.5$).

---

### 4. PRM Training Objective

The PRM is trained using Binary Cross-Entropy (<abbr title="Binary Cross-Entropy">BCE</abbr>) loss applied to the representation of the terminal step token (the newline or step delimiter) across all steps:

$$
\mathcal{L}_{\text{PRM}}(\boldsymbol{\psi}) = -\sum_{k=1}^K \left[ y_k \log r_k + (1 - y_k) \log(1 - r_k) \right]
$$

---

### 5. PRM-Guided Best-of-$N$ Selection Metrics

When sampling $N$ candidate solutions at test time, how do we rank them?
There are three primary PRM ranking aggregators:

1. **Minimum Step Score (Bottleneck Pruning)**:
   $$
   \operatorname{Score}_{\min}(y) = \min_{1 \le k \le K} r_k
   $$
   *A chain is only as strong as its weakest link! If even one step is flawed, the solution is discarded.*
2. **Product of Step Scores (Joint Probability)**:
   $$
   \operatorname{Score}_{\text{prod}}(y) = \prod_{k=1}^K r_k = \exp\left( \sum_{k=1}^K \log r_k \right)
   $$
3. **Length-Normalized Average Score**:
   $$
   \operatorname{Score}_{\text{mean}}(y) = \frac{1}{K} \sum_{k=1}^K r_k
   $$

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2022">2022</time> &mdash; <strong>Jonathan Uesato et al. (DeepMind)</strong></dt>
  <dd>Published <cite>"Solving math word problems with process- and outcome-based feedback"</cite>, formally demonstrating that step-level human feedback produces superior reasoning reliability compared to terminal outcome rewards alone.</dd>

  <dt><time datetime="2023-05">May 2023</time> &mdash; <strong>Hunter Lightman et al. (OpenAI)</strong></dt>
  <dd>Published the landmark paper <cite>"Let's Verify Step by Step"</cite> alongside the <strong>PRM800K</strong> dataset (800,000 human-annotated math step labels). Proved that PRM-guided Best-of-$N$ search crushed ORMs on the high-school MATH benchmark (78.2% vs. 69.6%), specifically eliminating false-positive logic leaps.</dd>

  <dt><time datetime="2023-12">December 2023</time> &mdash; <strong>Peiyi Wang et al. (Math-Shepherd)</strong></dt>
  <dd>Overcame the human annotation bottleneck by demonstrating that Monte Carlo completion rollouts could autonomously synthesize high-quality PRM labels at scale, paving the way for fully automated RL verification loops.</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace step verification by hand on two competing candidate derivations for a high school algebra problem.

### The Problem
<kbd>"Find x if 2x + 6 = 16"</kbd>. (Ground truth: $x = 5$).

---

### Candidate Solution A: Valid Step-by-Step Derivation

- **Step 1 ($s_1$)**: <samp>"Subtract 6 from both sides: 2x = 16 - 6 = 10\n\n"</samp>
  - PRM evaluation: Correct algebraic manipulation $\to$ <mark>$r_1 = 0.98$</mark>.
- **Step 2 ($s_2$)**: <samp>"Divide both sides by 2: x = 10 / 2 = 5\n\n"</samp>
  - PRM evaluation: Flawless division $\to$ <mark>$r_2 = 0.96$</mark>.
- **Step 3 ($s_3$)**: <samp>"Therefore, the solution is 5.\n\n"</samp>
  - PRM evaluation: Direct logical conclusion $\to$ <mark>$r_3 = 0.99$</mark>.

#### Metrics for Solution A:
- Outcome Verifier (ORM): Answer is 5 $\to$ $R_{\text{ORM}} = 1.0$.
- PRM Minimum: $\operatorname{Score}_{\min} = \min(0.98, 0.96, 0.99) = \mathbf{0.96}$.
- PRM Product: $\operatorname{Score}_{\text{prod}} = 0.98 \times 0.96 \times 0.99 = \mathbf{0.9314}$.

---

### Candidate Solution B: Flawed Logic with Canceling Lucky Error

- **Step 1 ($s_1$)**: <samp>"Subtract 6 from both sides: 2x = 16 + 6 = 22\n\n"</samp>
  - Notice the fatal sign error ($+6$ instead of $-6$)!
  - PRM evaluation: Arithmetic hallucination $\to$ <del>$r_1 = 0.04$</del>.
- **Step 2 ($s_2$)**: <samp>"Divide both sides by 2: x = 22 / 2 = 11\n\n"</samp>
  - PRM evaluation: Mechanically follows Step 1, but premise is false $\to$ $r_2 = 0.85$.
- **Step 3 ($s_3$)**: <samp>"Subtract 6 to get final answer: x = 11 - 6 = 5. Final answer: 5.\n\n"</samp>
  - Random unmotivated subtraction that accidentally hits the number 5!
  - PRM evaluation: Unjustified operation $\to$ $r_3 = 0.10$.

#### Metrics for Solution B:
- **Outcome Verifier (ORM)**: Final boxed answer is $5$ $\to$ <mark>$R_{\text{ORM}} = 1.0$ (Fatal False Positive!)</mark>
- **PRM Minimum**:
  $$
  \operatorname{Score}_{\min} = \min(0.04, 0.85, 0.10) = \mathbf{0.04}
  $$
- **PRM Product**:
  $$
  \operatorname{Score}_{\text{prod}} = 0.04 \times 0.85 \times 0.10 = \mathbf{0.0034}
  $$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 32.1:</strong> Comparison of ORM vs. PRM scoring on Candidate B (Lucky Error).</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Evaluation System</th>
      <th align="right">Assigned Score</th>
      <th align="left">Verdict</th>
      <th align="left">Consequence on LLM Training</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Outcome Reward Model (ORM)</strong></td>
      <td align="right"><strong>1.0000</strong></td>
      <td>Accepted as ground truth</td>
      <td><mark>Fatal</mark>: Reinforces sign error and random arithmetic tricks.</td>
    </tr>
    <tr>
      <td><strong>PRM Minimum Aggregator</strong></td>
      <td align="right"><strong>0.0400</strong></td>
      <td>Instantly rejected</td>
      <td><strong>Success</strong>: Discards path due to Step 1 fallacy.</td>
    </tr>
    <tr>
      <td><strong>PRM Joint Product Aggregator</strong></td>
      <td align="right"><strong>0.0034</strong></td>
      <td>Severely penalized</td>
      <td><strong>Success</strong>: Solution pruned from search tree.</td>
    </tr>
  </tbody>
</table>

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>Process Reward Models (PRMs) solve the foundational credit assignment failure of Large Language Models: by evaluating the mathematical validity of every single intermediate reasoning step rather than just the final answer, PRMs expose lucky logical fallacies and provide granular value gradients that enable search algorithms to prune doomed branches the exact millisecond they occur.</p>
</fieldset>
