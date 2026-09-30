# The Math Behind LLM Reinforcement Learning: Master Curriculum

<nav aria-label="Table of Contents">
  <p>
    <strong>RL Track Navigation:</strong>
    <a href="#pedagogy">5-Step Pedagogy</a> &bull;
    <a href="#primer">Gentle Primer &amp; Dependency Ladder</a> &bull;
    <a href="#pipeline">RL Pipeline Overview</a> &bull;
    <a href="#foundations">Foundations &amp; Policy Gradients</a> &bull;
    <a href="#actor-critic">Actor-Critic &amp; PPO</a> &bull;
    <a href="#grpo">Critic-Free GRPO &amp; Reasoning</a> &bull;
    <a href="#prm">Process Reward Models (PRM)</a> &bull;
    <a href="#search">MCTS &amp; Test-Time Compute</a> &bull;
    <a href="#lab">Hands-on GRPO Lab</a>
  </p>
</nav>

<hr>

<fieldset>
<legend><strong>Core Mission: The Mathematics of LLM Reinforcement Learning &amp; Reasoning</strong></legend>
<p>Reinforcement Learning for Large Language Models has evolved from an alignment filter (RLHF) into the primary engine of modern artificial intelligence and machine reasoning (DeepSeek-R1, OpenAI o1/o3). While pre-training compresses human text into statistical distributions, reinforcement learning teaches models how to search, backtrack, verify, and discover novel reasoning strategies through autonomous trial, error, and credit assignment.</p>
<p>Every chapter in this curriculum answers two foundational questions:</p>
<ol>
  <li><em>What physical intuition makes this reinforcement learning mechanic instantly obvious to a child?</em></li>
  <li><em>Where does the exact mathematical equation come from, and why did researchers write it this way?</em></li>
</ol>
</fieldset>

---

<h2 id="primer">Gentle Primer: Transitioning from Pre-Training to Reinforcement Learning</h2>

<p>Many students find the mathematics of reinforcement learning intimidating because the equations look drastically different from standard deep learning. In pre-training and supervised fine-tuning, every mathematical step revolves around a simple, deterministic goal: <em>given an input, match the teacher's target word</em>. In reinforcement learning, that safety net disappears.</p>

<p>To ease your journey, this primer breaks down the three foundational paradigm shifts that govern all LLM reinforcement learning mathematics.</p>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table R.0:</strong> The three fundamental shifts from Supervised Learning (SFT) to Reinforcement Learning (RL).</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">Dimension</th>
      <th align="left" width="38%">Supervised Learning (SFT / Pre-training)</th>
      <th align="left" width="40%">Reinforcement Learning (RL / Reasoning)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. The Training Signal</strong></td>
      <td><strong>The Golden Answer Key</strong>: A human expert provides the exact ground-truth token $y_t^*$. The loss is standard Cross-Entropy: $-\log \pi_{\boldsymbol{\theta}}(y_t^* \mid x)$.</td>
      <td><strong>The Scorecard</strong>: Nobody provides the exact words. The model generates an entire sequence, and an external evaluator (Python interpreter, math grader, human) gives a single score $R$.</td>
    </tr>
    <tr>
      <td><strong>2. The Gradient Path</strong></td>
      <td><strong>Direct Backpropagation</strong>: The loss function is a smooth, continuous mathematical function of the model's logits: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{1}_{y^*}$. Gradients flow smoothly backwards.</td>
      <td><strong>The Discrete Wall</strong>: Words are selected via discrete sampling ($y_t \sim \operatorname{Categorical}(\mathbf{p})$) or $\operatorname{argmax}$. You cannot take the derivative of a sampled word or a Python test runner!</td>
    </tr>
    <tr>
      <td><strong>3. The Optimization Mechanism</strong></td>
      <td><strong>Imitation</strong>: Pull model parameters in the exact direction that increases the likelihood of the human demonstrator's words.</td>
      <td><strong>Trial, Error &amp; Dynamic Weighting</strong>: Model explores autonomously. The <em>Score Function Trick</em> turns the policy gradient into <strong>weighted cross-entropy on self-generated text</strong>, scaled by how much better the outcome was than expected.</td>
    </tr>
  </tbody>
</table>

<br>

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   THE UNBROKEN REINFORCEMENT LEARNING DEPENDENCY LADDER                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Supervised Fine-Tuning (Chapter 15)                                                  │
│   └── Loss: L_CE = -log π_θ(y*)  ──► Pulls towards teacher's exact token               │
│                                                                                        │
│   ▼ [Shift 1: No Teacher Token! Only a Score R(τ)]                                     │
│   REINFORCE &amp; Score-Function Trick (Chapter 29)                                        │
│   └── ∇J(θ) = E[ ∇log π_θ(a) · R(τ) ]  ──► Weighted SFT on model's own words!          │
│                                                                                        │
│   ▼ [Shift 2: Positive Rewards Cause High Variance!]                                   │
│   Baseline Subtraction (Chapter 29)                                                    │
│   └── ∇J(θ) = E[ ∇log π_θ(a) · (R(τ) - b) ]  ──► Only reward better-than-average text  │
│                                                                                        │
│   ▼ [Shift 3: 1,000-Token Essays Need Intermediate Credit!]                            │
│   Actor-Critic &amp; GAE (Chapter 30)                                                      │
│   └── TD Error: δ_t = r_t + γ V(s_t+1) - V(s_t)  ──► Token-level surprise scores       │
│                                                                                        │
│   ▼ [Shift 4: Unbounded Policy Steps Cause Model Collapse!]                            │
│   PPO Clipped Objective (Chapter 30)                                                   │
│   └── min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)  ──► Elastic safety tether prevents crash  │
│                                                                                        │
│   ▼ [Shift 5: Critic Models Double GPU VRAM &amp; Hallucinate!]                            │
│   Critic-Free GRPO (Chapter 31)                                                        │
│   └── Group Advantage: A_i = (R_i - μ_q) / σ_q  ──► Cohort acts as its own baseline!   │
│                                                                                        │
│   ▼ [Shift 6: Multi-Step Proofs Need Per-Step Verification!]                           │
│   Process Reward Models (Chapter 32)                                                   │
│   └── Step PRM: r_k = P(step k is sound)  ──► Pinpoints exact algebraic errors         │
│                                                                                        │
│   ▼ [Shift 7: Inference Thinking Compute Scaling!]                                     │
│   Test-Time Search &amp; MCTS (Chapter 33)                                                 │
│   └── PUCT: Q(s,a) + U(s,a)  ──► System-2 deliberative exploration at test time        │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure R.0:</strong> The conceptual dependency ladder connecting supervised pre-training to modern test-time reasoning search.</figcaption>
</figure>

---

<h2 id="pedagogy">The Mandatory 5-Step Pedagogy</h2>

Every chapter in the Reinforcement Learning Curriculum adheres to the unshakeable 6-step learning ladder:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table R.1:</strong> The 6-step pedagogical learning sequence for LLM reinforcement learning.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center" width="10%">Step</th>
      <th align="left" width="22%">Section Title</th>
      <th align="left" width="38%">What You Learn</th>
      <th align="left" width="30%">Tactile Physical Metaphor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Step 1</strong></td>
      <td><strong>3-Year-Old Intuition</strong></td>
      <td>Pure physical metaphor with zero mathematical jargon</td>
      <td>Blindfolded archers, classroom pop quizzes, soup taste-testing, maze scouts</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 2</strong></td>
      <td><strong>The Bridging Question</strong></td>
      <td>Translating physical reinforcement games into mathematical objectives</td>
      <td>Credit assignment, variance reduction, probability ratios, policy bounds</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 3</strong></td>
      <td><strong>The Exact Math &amp; Formula</strong></td>
      <td>The genuine equations governing modern LLM RL engines</td>
      <td>Policy gradient theorem, GAE advantage, clipped surrogates, GRPO group statistics, PUCT search</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 4</strong></td>
      <td><strong>Where Did It Come From?</strong></td>
      <td>Historical engineering origin and failure modes</td>
      <td>Why naive algorithms collapsed (reward hacking, high variance, GPU OOM) and what broke first</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 5</strong></td>
      <td><strong>Concrete Toy Example</strong></td>
      <td>Hand-calculated arithmetic with tiny numbers and small vocabularies</td>
      <td>Step-by-step additions, multiplications, log-ratios, and parameter updates</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 6</strong></td>
      <td><strong>Core Takeaway</strong></td>
      <td>1–2 sentence conceptual punchline</td>
      <td>The architectural anchor for production reinforcement learning systems</td>
    </tr>
  </tbody>
</table>

---

<h2 id="pipeline">The 5-Chapter RL &amp; Reasoning Curriculum Overview</h2>

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   THE LLM REINFORCEMENT LEARNING &amp; REASONING ROADMAP                   │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. POLICY GRADIENTS      │ 2. ACTOR-CRITIC &amp; PPO    │ 3. MODERN REASONING &amp; SEARCH     │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Ch 29: Language as MDP   │ Ch 30: Actor-Critic &amp;    │ Ch 31: Group Relative Policy     │
│        &amp; REINFORCE       │        PPO               │        Optimization (GRPO)       │
│        (Score Function   │        (4-Model Cluster, │        (DeepSeek-R1, Critic-Free │
│         &amp; Baselines)     │         GAE, Clipped Obj)│         Group Advantage)         │
│                          │                          │                                  │
│                          │                          │ Lab 07: GRPO Engine in Python    │
│                          │                          │                                  │
│                          │                          │ Ch 32: Process Reward Models     │
│                          │                          │        (PRM &amp; Step Verification) │
│                          │                          │                                  │
│                          │                          │ Ch 33: MCTS &amp; Test-Time Compute  │
│                          │                          │        (Search Scaling Laws)     │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
</pre>
<figcaption><strong>Figure R.1:</strong> Structural taxonomy of the LLM Reinforcement Learning and Reasoning Curriculum.</figcaption>
</figure>

---

<h3 id="foundations">Track 1: Foundations &amp; Policy Gradients</h3>

#### [Chapter 29: Language as an MDP &amp; Policy Gradients (REINFORCE &amp; Variance Reduction)](29-policy-gradients-and-reinforce/index.html)
- **The Metaphor**: The Blindfolded Archer and the Scorekeeper's Whispers. The archer fires arrows in total darkness; the coach whispers only the final score. Subtracting the average score ensures the archer only adjusts their stance for shots that outperformed the historical average.
- **The Math**:
  - Language generation as a discrete Markov Decision Process (<abbr title="Markov Decision Process">MDP</abbr>):
    - State $s_t = (x, y_{\lt t})$ (prompt plus generated prefix tokens).
    - Action $a_t = y_t \in \mathcal{V}$ (next token selected from vocabulary).
    - Transition $s_{t+1} = [s_t, a_t]$ (deterministic sequence concatenation).
    - Policy $\pi_{\boldsymbol{\theta}}(a_t \mid s_t)$ (softmax probability distribution over $\mathcal{V}$).
    - Trajectory Return $R(\tau) = \sum_{t=1}^T r(s_t, a_t)$.
  - The Policy Gradient Theorem via the Log-Derivative / Score-Function Trick:
    $$
    \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]
    $$
  - Baseline Subtraction Theorem: Proof that $\mathbb{E}\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = 0$, guaranteeing zero gradient bias while dramatically slashing sample variance $\operatorname{Var}(\hat{\mathbf{g}})$.
- **Formula Origin**: Ronald Williams (REINFORCE, 1992), Sutton et al. (Policy Gradient Theorem, 1999).

---

<h3 id="actor-critic">Track 2: The Actor-Critic Architecture &amp; PPO</h3>

#### [Chapter 30: The Actor-Critic Architecture &amp; PPO (Proximal Policy Optimization)](30-ppo-and-actor-critic/index.html)
- **The Metaphor**: The Stage Actor, the Director in the Front Row, and the Elastic Safety Tether. The actor delivers spontaneous lines; the director evaluates whether the performance exceeded expectations; the elastic tether prevents the actor from making sudden, catastrophic changes to their acting style.
- **The Math**:
  - The 4-Model System in classical LLM RLHF:
    - Actor Model $\pi_{\boldsymbol{\theta}}$ (trainable generative policy).
    - Critic Model $V_{\boldsymbol{\phi}}$ (trainable scalar state-value predictor).
    - Reference Model $\pi_{\text{ref}}$ (frozen base model anchoring distribution).
    - Reward Model $R_{\boldsymbol{\psi}}$ (frozen preference evaluator).
  - Generalized Advantage Estimation (<abbr title="Generalized Advantage Estimation">GAE</abbr>):
    $$
    \delta_t^V = r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1}) - V_{\boldsymbol{\phi}}(s_t), \quad \hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V
    $$
  - The PPO Clipped Surrogate Objective:
    $$
    \mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta}) \hat{A}_t, \; \operatorname{clip}(r_t(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
    $$
    where $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$.
  - Token-level Kullback-Leibler (<abbr title="Kullback-Leibler">KL</abbr>) divergence penalty: $r_t = R(s_t, a_t) - \beta D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}})$.
- **Formula Origin**: Schulman et al. (TRPO 2015, PPO 2017), Christiano et al. (2017), Ouyang et al. (InstructGPT, 2022).

---

<h3 id="grpo">Track 3: Critic-Free Group Optimization &amp; Reasoning</h3>

#### [Chapter 31: Group Relative Policy Optimization (GRPO &amp; DeepSeek-R1)](31-grpo-and-reasoning/index.html)
- **The Metaphor**: The Classroom Pop Quiz. Instead of hiring an exorbitant personal tutor to stand over each student and evaluate every single syllable (the Critic network), the teacher poses a challenging math problem to 4 students simultaneously. The students write full solutions independently, and grades are calculated strictly relative to the group's collective average!
- **The Math**:
  - Elimination of the Value Network ($V_{\boldsymbol{\phi}}$): Halving GPU VRAM requirements and eliminating critic estimation error.
  - Group sampling: For each prompt $q \sim \mathcal{P}(Q)$, sample a group of $G$ outputs $\{o_1, o_2, \dots, o_G\} \sim \pi_{\boldsymbol{\theta}_{\text{old}}}(q)$.
  - Group-Relative Advantage Normalization:
    $$
    A_i = \frac{R_i - \operatorname{mean}(\{R_1, \dots, R_G\})}{\operatorname{std}(\{R_1, \dots, R_G\}) + \epsilon}
    $$
  - The Full GRPO Objective Function:
    $$
    \mathcal{J}_{\text{GRPO}}(\boldsymbol{\theta}) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \frac{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} A_i, \, \operatorname{clip}\left(\dots, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}) \right]
    $$
  - Rule-Based Verifiable Rewards (<abbr title="Reinforcement Learning with Verifiable Rewards">RLVR</abbr>): Combining deterministic accuracy check (compiler / SymPy match) and syntactic format check (`<think>...</think><answer>...</answer>`).
  - Schulman's unbiased KL approximation:
    $$
    D_{\text{KL}} \approx \frac{\pi_{\text{ref}}}{\pi_{\boldsymbol{\theta}}} - \log \frac{\pi_{\text{ref}}}{\pi_{\boldsymbol{\theta}}} - 1
    $$
- **Formula Origin**: DeepSeek-Math (Shao et al., Feb 2024), DeepSeek-R1 (Jan 2025).

#### [Hands-on Lab 07: The GRPO Reasoning Engine in Pure Python](31b-lab-grpo-engine/index.html)
- **Architecture**: A self-contained, zero-dependency pure Python reinforcement learning engine implementing candidate group sampling, rule-based format and accuracy verifiers, group-relative advantage normalization, and clipped policy gradient updates.
- **Verification**: Simulates the emergence of extended reasoning traces and verified answer convergence on mathematical tasks.

---

<h3 id="prm">Track 4: Step-Level Verification &amp; Process Reward Models</h3>

#### [Chapter 32: Process Reward Models (PRM &amp; Step-Level Verification)](32-process-reward-models/index.html)
- **The Metaphor**: The Master Chef Inspecting the Soup at Every Step. Tasting the broth after caramelizing onions, after adding wine, and before adding salt &mdash; instead of only tasting the finished bowl when it is already ruined and having no clue which ingredient caused the failure.
- **The Math**:
  - Outcome-supervised Reward Models (<abbr title="Outcome Reward Model">ORM</abbr>) vs. Process-supervised Reward Models (<abbr title="Process Reward Model">PRM</abbr>).
  - The Credit Assignment Dilemma: In a 20-step reasoning chain, an error at Step 3 produces an outcome reward of $0$, penalizing Steps 1 and 2 (which were flawless) and providing zero gradient guidance on the precise point of failure.
  - Step-Level Mathematical Decomposition: Solution chain $y = (s_1, s_2, \dots, s_K)$, with individual step correctness probability:
    $$
    r_k = \sigma\left(f_{\boldsymbol{\psi}}(x, s_{1:k})\right) \in [0, 1]
    $$
  - Path Joint Validity Product: $P(\text{solution correct}) = \prod_{k=1}^K r_k$.
  - Automated Step Annotation via Monte Carlo Rollouts (Math-Shepherd):
    $$
    V^*(s_k) \approx \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left(\text{Rollout}_m(s_k) \text{ produces correct final answer}\right)
    $$
- **Formula Origin**: Uesato et al. (DeepMind, 2022), Lightman et al. (OpenAI, *Let's Verify Step by Step*, 2023), Wang et al. (Math-Shepherd, 2023).

---

<h3 id="search">Track 5: Test-Time Compute Scaling &amp; Search</h3>

#### [Chapter 33: Search, MCTS &amp; Test-Time Compute Scaling](33-mcts-and-test-time-compute/index.html)
- **The Metaphor**: The Chess Grandmaster Looking Ahead. Instead of immediately playing the first move that pops into mind, the grandmaster mentally explores 5 alternative moves, looks 3 steps into the future for each branch, identifies hidden traps, and chooses the safest, most lethal path.
- **The Math**:
  - Inference Compute Scaling Laws: Performance scales log-linearly with test-time compute expenditure $C_{\text{test}}$ independently of pre-training size:
    $$
    \text{Accuracy} \approx f\left(N_{\text{samples}}, \, T_{\text{thinking}}\right)
    $$
  - Best-of-$N$ Sampling and Majority Voting (Self-Consistency) mathematical bounds:
    $$
    P(\text{Majority Correct}) = \sum_{k=\lceil N/2 \rceil}^N \binom{N}{k} p^k (1-p)^{N-k}
    $$
  - Monte Carlo Tree Search (<abbr title="Monte Carlo Tree Search">MCTS</abbr>) on Language Reasoning Steps:
    1. **Selection**: Predictor Upper Confidence Bound applied to Trees (<abbr title="Predictor Upper Confidence Bound applied to Trees">PUCT</abbr>):
       $$
       \operatorname{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_{b} N(s, b)}}{1 + N(s, a)}
       $$
    2. **Expansion**: Sampling $K$ candidate reasoning steps from $\pi_{\boldsymbol{\theta}}(s_{t+1} \mid s_t)$.
    3. **Evaluation**: Value scoring via PRM $r(s_t, a_t)$ or simulated rollout.
    4. **Backpropagation**: Updating node statistics $N(s, a) \leftarrow N(s, a) + 1$ and $Q(s, a) \leftarrow Q(s, a) + \frac{V - Q(s, a)}{N(s, a)}$.
  - Autonomous Reasoning Phenomena: Thinking token expansion, backtracks (`"Wait, let me double check that..."`), and self-correction without human demonstrations (DeepSeek-R1 Zero).
- **Formula Origin**: Silver et al. (AlphaGo / AlphaZero, 2016-2018), Snell et al. (2024), OpenAI o1 (2024), DeepSeek-R1 (2025).

---

<fieldset>
<legend><strong>Reinforcement Learning &amp; Reasoning Mastery Summary</strong></legend>
<p>By completing this 5-chapter RL track, you master the exact mathematical machinery transforming Large Language Models from passive text predictors into autonomous reasoning agents. You understand how policy gradients guide sequence generation, why critic-free group optimization (GRPO) unlocked open-source frontier reasoning, how process reward models isolate reasoning errors, and how test-time tree search scales intelligence at inference time.</p>
</fieldset>
