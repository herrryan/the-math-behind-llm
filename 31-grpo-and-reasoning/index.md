# Chapter 31: Group Relative Policy Optimization (GRPO & DeepSeek-R1)

---

## Step 1: 3-Year-Old Intuition (The Classroom Pop Quiz)

Imagine a classroom where four children are asked to solve the exact same tricky riddle:

1. **The Old, Expensive Way (PPO with a Private Tutor)**:
   - In Chapter 30, we saw that PPO hires a private tutor (the Critic network $V_{\boldsymbol{\phi}}$) to sit beside a single student.
   - Every time the student writes down a single letter, the tutor leans over and whispers: *"Hmm, that comma looks like a 7 out of 10... that number looks like an 8 out of 10..."*
   - That private tutor is enormous &mdash; taking up half the seats in the room (50% of your GPU memory) &mdash; and on long math problems, the tutor often guesses wrong anyway!

2. **The DeepSeek-R1 Way (The Group Pop Quiz)**:
   - DeepSeek asked a radical question: **What if we fire the private tutor completely?**
   - Instead, the teacher writes the riddle on the chalkboard and asks **4 students** to write their full solutions on the board at the same time:
     - **Student 1**: Shows full step-by-step work and finds the right answer $\to$ **100 points!**
     - **Student 2**: Writes messy work but also gets the right answer $\to$ **80 points!**
     - **Student 3**: Makes an arithmetic slip at the end $\to$ **20 points!**
     - **Student 4**: Writes random nonsense $\to$ **0 points!**

3. **Grading Relative to the Class Average**:
   - The class average on this riddle is:
     $$
     \text{Average} = \frac{100 + 80 + 20 + 0}{4} = 50 \text{ points}
     $$
   - Student 1 scored $+50$ points above average $\to$ **Positive Advantage!** We praise their reasoning steps.
   - Student 2 scored $+30$ points above average $\to$ **Positive Advantage!**
   - Student 3 scored $-30$ points below average $\to$ **Negative Advantage!** We gently discourage their mistake.
   - Student 4 scored $-50$ points below average $\to$ **Strong Penalty!**

Zero private tutor needed. Zero value parameters taking up GPU memory. The students act as their own mutual baselines!

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                   GRPO: CRITIC-FREE GROUP OPTIMIZATION                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Prompt [q] ─────────────► [ Actor Policy π_θ ]                       │
│                                  │                                     │
│            ┌─────────────────────┼─────────────────────┐               │
│            ▼                     ▼                     ▼               │
│       Rollout o_1           Rollout o_2           Rollout o_G          │
│            │                     │                     │               │
│            ▼                     ▼                     ▼               │
│     [ Rule Verifier ]     [ Rule Verifier ]     [ Rule Verifier ]      │
│      Acc + Format          Acc + Format          Acc + Format          │
│            │                     │                     │               │
│            ▼                     ▼                     ▼               │
│        Reward R_1            Reward R_2            Reward R_G          │
│            └─────────────────────┬─────────────────────┘               │
│                                  ▼                                     │
│                 Group Mean μ_q &amp; Standard Deviation σ_q                │
│                                  │                                     │
│                                  ▼                                     │
│            Normalized Relative Advantages: A_i = (R_i - μ_q) / σ_q     │
│                                  │                                     │
│                                  ▼                                     │
│                 PPO-Clipped Token Policy Update                       │
│                  (Zero Critic! Zero Value Model!)                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 31.1:</strong> The GRPO architecture: Samples a group of outputs per prompt, computes rule-based rewards, normalizes relative advantages across the group, and updates the actor with zero critic network.</figcaption>
</figure>

---

## Step 2: The Bridging Question

In Chapter 30, we saw that Proximal Policy Optimization (<abbr title="Proximal Policy Optimization">PPO</abbr>) relies on a Critic network $V_{\boldsymbol{\phi}}$ to estimate token-level advantages $\hat{A}_t$.
However, when training modern reasoning models (such as DeepSeek-R1 or OpenAI o1/o3) on mathematical proofs and code synthesis, the Critic becomes a fatal bottleneck:

1. **The GPU VRAM Explosion (The 70B Twin Problem)**:
   - The Critic is a full Transformer backbone with parameters comparable in size to the Actor ($\approx 70\text{B}$ parameters).
   - In distributed training, Adam stores two 32-bit floating-point states (momentum and variance) per parameter, requiring $16$ bytes per parameter for optimizer states alone.
   - Hosting a 70B Critic alongside a 70B Actor effectively **doubles the entire GPU cluster size** required for post-training, burning millions of dollars in compute.

2. **The Mid-Proof Hallucination Trap**:
   - In a 15,000-token mathematical derivation, predicting whether step 4,210 will eventually lead to a correct proof is virtually impossible for a neural network.
   - If the Critic guesses incorrectly and marks a brilliant intermediate step with a negative value, its erroneous value gradients misguide the Actor policy, causing training to diverge.

3. **Goodhart's Law and Reward Hacking**:
   - In classic RLHF with neural reward models, Actor models quickly learn to exploit model blind spots: writing verbose, polite, flattering paragraphs that score high on subjective ratings without solving the actual problem.

The bridging question is:
$$\text{How do we mathematically eliminate the Critic network entirely, while computing low-variance, statistically valid advantages across a group of rollouts using unhackable rule-based verifiers?}$$

---

## Step 3: The Exact Math & Formula

### 1. Group Rollout Sampling

For each input query $q$ drawn from prompt distribution $\mathcal{P}(Q)$, the old policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$ generates a cohort of $G$ independent candidate completions:

$$
\{o_1, o_2, \dots, o_G\} \sim \pi_{\boldsymbol{\theta}_{\text{old}}}(q)
$$

where $G$ is the group size (typically $G \in [4, 16]$ in DeepSeek-Math and DeepSeek-R1).

---

### 2. Rule-Based Verifiable Rewards (RLVR)

Instead of relying on a fallible neural reward model, **Reinforcement Learning with Verifiable Rewards (<abbr title="Reinforcement Learning with Verifiable Rewards">RLVR</abbr>)** evaluates each completion $o_i$ using deterministic, non-gameable rule functions:

$$
R_i = r_{\text{acc}}(o_i) + r_{\text{format}}(o_i)
$$

1. **Accuracy Reward ($r_{\text{acc}}$)**: Evaluates whether the final boxed answer strictly matches the ground truth:
   $$
   r_{\text{acc}}(o_i) = \begin{cases} 1.0 & \text{if mathematical answer or code unit tests pass} \\ 0.0 & \text{otherwise} \end{cases}
   $$
2. **Format Reward ($r_{\text{format}}$)**: Enforces architectural discipline by verifying that the model wraps its reasoning chain in explicit structural tags:
   $$
   r_{\text{format}}(o_i) = \begin{cases} 1.0 & \text{if output strictly follows } \texttt{<think>...</think><answer>...</answer>} \\ 0.0 & \text{otherwise} \end{cases}
   $$

<fieldset>
<legend><strong>Why RLVR Is Immune to Goodhart's Law</strong></legend>
<p><em>Goodhart's Law</em> states: <strong>"When a measure becomes a target, it ceases to be a good measure."</strong></p>
<p>When training against a human preference reward model, LLMs learn that writing pleasant, repetitive filler phrases tricks the evaluator. But under RLVR, the evaluator is a cold, deterministic Python script (e.g., checking if <code>assert solve() == 42</code> or parsing LaTeX <code>\boxed{...}</code>). You cannot flatter a compiler! Either the mathematical logic works, or it fails.</p>
</fieldset>

---

### 3. Group-Relative Advantage Normalization (The Chalkboard Average)

For prompt $q$, let $\{R_1, R_2, \dots, R_G\}$ be the scalar rewards assigned to the $G$ sampled completions.
The empirical mean $\mu_q$ and standard deviation $\sigma_q$ of the group are:

$$
\mu_q = \frac{1}{G} \sum_{i=1}^G R_i, \quad \sigma_q = \sqrt{\frac{1}{G} \sum_{i=1}^G \left( R_i - \mu_q \right)^2 + \epsilon}
$$

where $\epsilon = 10^{-4}$ prevents division by zero.
The <dfn id="def-grpo-advantage">Group-Relative Advantage</dfn> $A_i$ for the $i$-th completion is:

$$
A_i = \frac{R_i - \mu_q}{\sigma_q}
$$

<fieldset>
<legend><strong>Feynman's View: How Four Chalk Marks Slayed the 70B Critic Robot</strong></legend>
<p>Recall Piece 5 from our Feynman primer: In Chapter 30, we spent half our GPU memory building a second giant 70-billion-parameter neural network (the Critic) just to guess what the baseline score should be!</p>
<p>DeepSeek's insight was almost laughably simple: <strong>Instead of guessing what the baseline is, just generate 4 answers to the exact same prompt and look at them on the number line!</strong></p>
<pre>
                                           Group Mean (μ = 50)
                                                    ▼
<───────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────>
           R_4 = 0             R_3 = 20            μ = 50              R_2 = 80            R_1 = 100
         (A_4 = -1.26)       (A_3 = -0.76)        (Baseline)          (A_2 = +0.76)       (A_1 = +1.26)
</pre>
<ol>
  <li><strong>The Zero-Sum Center</strong>: Look at what happens when you sum the differences:
    $$
    (+50) + (+30) + (-30) + (-50) = 0
    $$
    By definition:
    $$
    \sum_{i=1}^G (R_i - \mu_q) = \sum_{i=1}^G R_i - G \cdot \left( \frac{1}{G}\sum_{i=1}^G R_i \right) = \sum_{i=1}^G R_i - \sum_{i=1}^G R_i \equiv 0
    $$
    The group creates its own perfectly balanced zero-sum pivot on the number line. Winning solutions get positive nudges ($A_i > 0$); losing solutions get negative nudges ($A_i < 0$).
  </li>
  <li><strong>Dividing by $\sigma$: Measuring in Universal Strides</strong>:
    Subtracting $\mu$ centers the marks at zero. Dividing by the standard deviation $\sigma_q$ standardizes the scale: an advantage of $+1.26$ means this attempt was $1.26$ full standard deviations better than its classmates.
  </li>
  <li><strong>Self-Tuning Gradient Attenuation</strong>:
    <ul>
      <li>If a question is <em>too easy</em> and all $G$ outputs succeed ($R_i = 1$), then $\mu_q = 1$ and $R_i - \mu_q = 0 \implies A_i = 0$.</li>
      <li>If a question is <em>too hard</em> and all $G$ outputs fail ($R_i = 0$), then $\mu_q = 0$ and $R_i - \mu_q = 0 \implies A_i = 0$.</li>
      <li><strong>Result</strong>: The model automatically produces <strong>zero gradient updates</strong> on questions it has already mastered or questions completely beyond its capability, focusing 100% of its learning capacity on the frontier of its abilities!</li>
    </ul>
  </li>
</ol>
<p>Zero extra parameters. Zero extra GPU memory. Two lines of chalk on the board replaced a 70-billion-parameter Critic model!</p>
</fieldset>

---

### 4. The Full GRPO Objective Function

The objective optimized by GRPO over policy parameters $\boldsymbol{\theta}$ is:

$$
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\boldsymbol{\theta}) = \mathbb{E}_{\substack{q \sim \mathcal{P}(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\boldsymbol{\theta}_{\text{old}}}(q)}} \Bigg[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \bigg( &\min\left( \rho_{i,t}(\boldsymbol{\theta}) A_i, \; \operatorname{clip}\left(\rho_{i,t}(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon\right) A_i \right) \\
&- \beta D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}\right) \bigg) \Bigg]
\end{aligned}
$$

where:
- $\rho_{i,t}(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ is the token importance sampling ratio.
- $|o_i|$ is the sequence length of the $i$-th candidate output. Dividing by $|o_i|$ is crucial: it prevents long, verbose answers from dominating the gradient update simply because they contain more tokens!
- $\epsilon$ is the PPO clipping parameter ($\epsilon \approx 0.2$).
- $\beta$ controls the strength of the reference regularization.

---

### 5. The Schulman Unbiased Non-Negative KL Estimator

Rather than computing standard asymmetric sample KL divergence $\log(\pi / \pi_{\text{ref}})$ which can yield negative values on individual samples and destabilize training, DeepSeek adopts John Schulman's (2020) non-negative unbiased estimator:

$$
D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}\right) \approx \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})} - \log \left( \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})} \right) - 1
$$

<fieldset>
<legend><strong>First-Principles Proof: Why Is This Estimator Strictly Non-Negative?</strong></legend>
<p>Let $u = \frac{\pi_{\text{ref}}}{\pi_{\boldsymbol{\theta}}} > 0$. Consider the scalar function:</p>
$$
g(u) = u - \log u - 1
$$
<ol>
  <li>At $u = 1$ (when policies match): $g(1) = 1 - \log(1) - 1 = 1 - 0 - 1 = 0$.</li>
  <li>Take the first derivative with respect to $u$:
    $$
    g'(u) = 1 - \frac{1}{u}
    $$
  </li>
  <li>Analyze the critical points:
    <ul>
      <li>When $u > 1$: $g'(u) > 0 \implies$ the function is strictly increasing.</li>
      <li>When $u < 1$: $g'(u) < 0 \implies$ the function is strictly decreasing.</li>
    </ul>
  </li>
  <li>Therefore, $u = 1$ is the unique <strong>global minimum</strong> of $g(u)$, and:
    $$
    g(u) \ge 0 \quad \text{for all } u > 0
    $$
  </li>
</ol>
<p>This elementary calculus proof guarantees that the KL penalty is <strong>strictly non-negative ($\ge 0$) at every single token</strong>, ensuring reference regularization never accidentally acts as a reward subsidy!</p>
</fieldset>

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2024-02">February 2024</time> &mdash; <strong>Zhihong Shao et al. (DeepSeek-Math)</strong></dt>
  <dd>Introduced <em>Group Relative Policy Optimization (GRPO)</em> as a specialized mathematical framework for mathematical reasoning. Proved that by replacing the Critic with group normalization, training memory dropped by 50% while achieving state-of-the-art results on GSM8K (88.2%) and MATH (51.7%).</dd>

  <dt><time datetime="2025-01">January 2025</time> &mdash; <strong>DeepSeek-AI (DeepSeek-R1 &amp; R1-Zero)</strong></dt>
  <dd>Applied pure GRPO directly to a base model without prior Supervised Fine-Tuning (<abbr title="Supervised Fine-Tuning">SFT</abbr>). Observed the historic spontaneous emergence of extended reasoning chains: the model learned on its own to allocate thousands of "thinking tokens", perform self-verification, correct prior mistakes, and exhibit human-like "Aha!" moments solely driven by rule-based verifiable rewards.</dd>
</dl>

### Architectural Comparison: PPO vs. DPO vs. GRPO

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 31.1:</strong> Structural comparison of leading LLM post-training reinforcement algorithms.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Dimension</th>
      <th align="left">PPO (Chapter 30)</th>
      <th align="left">DPO (Chapter 19)</th>
      <th align="left">GRPO (DeepSeek-R1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Models in VRAM</strong></td>
      <td>4 (Actor, Critic, Ref, RM)</td>
      <td><strong>2</strong> (Actor, Ref)</td>
      <td><strong>2</strong> (Actor, Ref)</td>
    </tr>
    <tr>
      <td><strong>Critic Network</strong></td>
      <td>Full 70B parameter model</td>
      <td>None (Eliminated)</td>
      <td><strong>None (Group Statistics)</strong></td>
    </tr>
    <tr>
      <td><strong>Exploration Capacity</strong></td>
      <td>Active online sampling</td>
      <td><mark>Zero (Offline static pairs)</mark></td>
      <td><strong>Full online group sampling</strong></td>
    </tr>
    <tr>
      <td><strong>Reward Source</strong></td>
      <td>Neural Reward Model</td>
      <td>Pairwise human preference</td>
      <td><strong>Verifiable Rule Engine (RLVR)</strong></td>
    </tr>
    <tr>
      <td><strong>Reasoning Evolution</strong></td>
      <td>Struggles on long chains</td>
      <td>Cannot discover new steps</td>
      <td><strong>Spontaneous "Aha!" moments</strong></td>
    </tr>
  </tbody>
</table>

---

## Step 5: Concrete Toy Example

Let us trace a complete GRPO group advantage step by hand with exact numbers.

### The Problem Setup
- Prompt $q$: <kbd>"Solve 2 + 3 * 4"</kbd>.
- Ground truth answer: $14$.
- Group size: $G = 4$ independent completions sampled from $\pi_{\boldsymbol{\theta}_{\text{old}}}$.
- Reward rules:
  - Format Reward: $+0.5$ if `<think>...</think><answer>...</answer>` tags exist.
  - Accuracy Reward: $+1.0$ if answer is $14$.
  - Total maximum reward: $1.5$.

---

### Phase 1: Evaluating the Group Completions

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 31.2:</strong> Group completions and evaluated reward scores.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center">Candidate</th>
      <th align="left">Model Output Text</th>
      <th align="center">Format ($r_{\text{fmt}}$)</th>
      <th align="center">Accuracy ($r_{\text{acc}}$)</th>
      <th align="right">Total Reward ($R_i$)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">$o_1$</td>
      <td><samp>&lt;think&gt;3*4=12, 12+2=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_2$</td>
      <td><samp>&lt;think&gt;Order of ops: 2+12=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_3$</td>
      <td><samp>&lt;think&gt;2+3=5, 5*4=20&lt;/think&gt;&lt;answer&gt;20&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">0.0 (+0.0)</td>
      <td align="right"><strong>0.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_4$</td>
      <td><samp>The answer is 14.</samp></td>
      <td align="center">0.0 (+0.0)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.0</strong></td>
    </tr>
  </tbody>
</table>

Rewards across the group: $\mathbf{R} = [1.5, \; 1.5, \; 0.5, \; 1.0]$.

---

### Phase 2: Computing Group Statistics & Advantages

#### 1. Group Mean ($\mu_q$)
$$
\mu_q = \frac{1.5 + 1.5 + 0.5 + 1.0}{4} = \frac{4.5}{4} = \mathbf{1.125}
$$

#### 2. Group Variance & Standard Deviation ($\sigma_q$)
Deviations from the mean:
- $(1.5 - 1.125)^2 = (+0.375)^2 = 0.140625$
- $(1.5 - 1.125)^2 = (+0.375)^2 = 0.140625$
- $(0.5 - 1.125)^2 = (-0.625)^2 = 0.390625$
- $(1.0 - 1.125)^2 = (-0.125)^2 = 0.015625$

Variance:
$$
\sigma_q^2 = \frac{0.140625 + 0.140625 + 0.390625 + 0.015625}{4} = \frac{0.6875}{4} = 0.171875
$$
Standard deviation:
$$
\sigma_q = \sqrt{0.171875} \approx \mathbf{0.4146}
$$

#### 3. Normalized Group Advantages ($A_i = \frac{R_i - \mu_q}{\sigma_q}$)
- $A_1 = \frac{1.5 - 1.125}{0.4146} = \frac{+0.375}{0.4146} \approx \mathbf{+0.9045}$ (<mark>High Positive Reinforcement</mark>)
- $A_2 = \frac{1.5 - 1.125}{0.4146} = \frac{+0.375}{0.4146} \approx \mathbf{+0.9045}$ (<mark>High Positive Reinforcement</mark>)
- $A_3 = \frac{0.5 - 1.125}{0.4146} = \frac{-0.625}{0.4146} \approx \mathbf{-1.5077}$ (<del>Strong Penalty for Wrong Math</del>)
- $A_4 = \frac{1.0 - 1.125}{0.4146} = \frac{-0.125}{0.4146} \approx \mathbf{-0.3015}$ (Mild Penalty for Missing Formatting)

Check sum of advantages: $+0.9045 + 0.9045 - 1.5077 - 0.3015 = -0.0002 \approx \mathbf{0}$!

---

### Phase 3: Token Clipped Policy Gradient Step

Consider the token <kbd>"12"</kbd> in completion $o_1$:
- $\pi_{\text{old}}("12" \mid \dots) = 0.20$.
- Current policy after an update: $\pi_{\boldsymbol{\theta}}("12" \mid \dots) = 0.23$.
- Importance ratio:
  $$
  \rho_{1, t} = \frac{0.23}{0.20} = \mathbf{1.15}
  $$
- Clipping bounds with $\epsilon = 0.2$: $[1-\epsilon, 1+\epsilon] = [0.80, 1.20]$.
- Since $1.15 \le 1.20$, clipping is not triggered.
- Surrogate term:
  $$
  \rho_{1, t} A_1 = 1.15 \times (+0.9045) = \mathbf{+1.0402}
  $$
- The token <kbd>"12"</kbd> receives a positive gradient boost, teaching the model that computing $3 \times 4 = 12$ leads to high group-relative success!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>Group Relative Policy Optimization (GRPO) achieves frontier reasoning by banishing the complex, memory-hungry Critic network: by sampling a cohort of answers for each prompt and normalizing rewards across the group, the model acts as its own dynamic baseline.</p>
<p>Paired with verifiable rule-based rewards (RLVR), GRPO unlocks the self-reinforcing engine of test-time thought &mdash; allowing language models to discover complex reasoning, backtracking, and self-correction purely through trial and error.</p>
</fieldset>
