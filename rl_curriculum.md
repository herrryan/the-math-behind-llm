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

<h2 id="primer">The Feynman Primer: Six Easy Pieces from the Number Line to Reinforcement Learning</h2>

<p>Richard Feynman famously taught that if you cannot explain a concept from first principles without hiding behind textbook jargon and abstract definitions, you do not truly understand it. In this primer, we throw away dry academic tuples like $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$ and build the entire mathematics of reinforcement learning from scratch &mdash; starting with a chalk line on the sidewalk and counting pebbles, and building an unbroken chain of physical logic all the way up to autonomous reasoning models like DeepSeek-R1.</p>

---

### Piece 1: Counting, The Number Line, and Wiggling Knobs

Imagine you are standing on a sidewalk with a piece of chalk:

1. **Counting Pebbles (The Natural Numbers)**:
   - You drop 1 pebble on the ground. Then another: 2 pebbles. Then another: 3 pebbles.
   - Counting is the primal root of all mathematics.

2. **The Number Line (Walking Left and Right)**:
   - Now draw a straight chalk line on the sidewalk. Make a mark where you are standing right now. That mark is **$0$**.
   - Take one normal stride forward. Make a mark: **$+1$**. Another stride forward: **$+2$**.
   - What if you take one stride backward? Make a mark behind you: **$-1$**. Another stride backward: **$-2$**.
   - *Addition* is just walking forward. *Subtraction* is just walking backward.
   - What if you take half a stride? You land on $0.5$. By slicing strides into finer pieces, every physical point on that continuous line has an exact address: a real number.

3. **A Machine with a Knob (Weights and Parameters)**:
   - Now imagine a wooden box with a volume knob on the front.
   - The position of the knob is just a number on our chalk line: let's call the knob's position $\theta$. Right now, the pointer is sitting at $\theta = 2.0$.
   - When you turn the knob, the box produces an electrical hum of loudness $y$. Suppose the loudness is connected to the knob by the simple rule: $y = 3\theta$. When $\theta = 2$, loudness is $y = 6$.

4. **What is a "Nudge"? (The Rate of Change, No Definitions Needed)**:
   - Suppose you give the knob a tiny physical nudge to the right by $\Delta \theta = 0.01$.
   - The loudness hum changes from $6.00$ to $6.03$, a shift of $\Delta y = 0.03$.
   - Now ask the most natural question in the world: *How sensitive is the loudness to a nudge on the knob?*
     $$
     \frac{\text{wobble in loudness}}{\text{wobble in knob}} = \frac{\Delta y}{\Delta \theta} = \frac{0.03}{0.01} = 3
     $$
   - That number &mdash; $3$ &mdash; is what mathematicians call a "derivative" or a "gradient" ($\nabla$).
   - Strip away the fancy Latin words: **a gradient is nothing more than a sensitivity number that tells you: if you nudge a knob to the right, does the output move up or down, and by how much?**
   - If the sensitivity is positive ($+3$), turning the knob to the right turns the volume up. If it is negative ($-3$), turning the knob to the right turns the volume down.

---

### Piece 2: The Jar of Marbles (Probability as Counting)

1. **A Machine That Rolls Dice**:
   - A language model does not just produce a single loudness hum. It chooses words.
   - How does a machine choose words? Imagine a glass jar filled with $100$ colored marbles:
     - $70$ marbles are blue, stamped with the word <kbd>"the"</kbd>.
     - $20$ marbles are green, stamped with the word <kbd>"cat"</kbd>.
     - $10$ marbles are red, stamped with the word <kbd>"apple"</kbd>.

2. **What is Probability? (Just Counting!)**:
   - You reach your hand in blindfolded and pull out one marble.
   - What is the chance of pulling out <kbd>"cat"</kbd>?
   - You don't need measure theory or axioms. You simply count: there are $20$ green marbles out of $100$ total marbles:
     $$
     P(\text{"cat"}) = \frac{20}{100} = 0.20 \quad (20\%)
     $$
   - A "probability" is just counting how many winning marbles you have divided by the total pile. It is a location on the number line strictly between $0$ (impossible) and $1$ (guaranteed).

3. **Connecting the Knob to the Jar**:
   - Inside the machine, our knob $\theta$ is connected to a mechanical lever inside the jar.
   - If you turn knob $\theta$ slightly to the right, the lever swaps $5$ red marbles for $5$ green marbles.
   - The chance of <kbd>"cat"</kbd> increases from $0.20$ to $0.25$.
   - How sensitive was the chance of <kbd>"cat"</kbd> to knob $\theta$?
     $$
     \frac{\Delta P}{\Delta \theta} = \frac{+0.05}{1} = +0.05
     $$
   - That is the gradient of the probability: it tells us which way to twist the knob to put more <kbd>"cat"</kbd> marbles into the jar.

---

### Piece 3: The Secret of the Percentage (Where the Logarithm Comes From)

In almost every reinforcement learning paper, you suddenly see the symbol $\nabla \log \pi$. Students panic: *Where did the natural logarithm come from? Why did researchers inject $\ln(x)$ into the formula?*

Here is the secret that textbooks never tell you:

1. **The Two Words Experiment**:
   - Suppose the jar contains two words:
     - Word A is an ultra-common word (<kbd>"is"</kbd>): its current probability is $P_A = 0.500$ ($500$ marbles out of $1{,}000$).
     - Word B is a rare, delicate mathematical word (<kbd>"hypotenuse"</kbd>): its current probability is $P_B = 0.001$ ($1$ marble out of $1{,}000$).
   - Now suppose you nudge a knob, and in both cases, you drop in exactly **$1$ extra marble** ($\Delta P = +0.001$):
     - For Word A: $P_A$ grows from $0.500 \to 0.501$. Its chance grew by a tiny **$0.2\%$**.
     - For Word B: $P_B$ grows from $0.001 \to 0.002$. Its chance **doubled** &mdash; a staggering **$100\%$ relative increase**!

2. **The Absolute Change Illusion**:
   - If you only look at the absolute change on the number line, both words gained the exact same $\Delta P = 0.001$.
   - But in the real world of language and discovery, taking a rare word from $1$-in-a-thousand to $2$-in-a-thousand is a monumental breakthrough, whereas nudging a common word by $0.2\%$ is imperceptible noise!
   - What measures this real-world impact? The **relative percentage change**:
     $$
     \text{Relative Nudge} = \frac{\text{change in probability}}{\text{current probability}} = \frac{\Delta P}{P}
     $$

3. **Why the Natural Logarithm Appears**:
   - Ask any calculus student: *What function has a slope equal to $\frac{1}{x}$?*
   - There is only one such function in all of mathematics: **the natural logarithm**:
     $$
     \frac{d}{dx} \ln(x) = \frac{1}{x}
     $$
   - By the chain rule, if you take the derivative of the logarithm of a probability:
     $$
     \nabla_\theta \ln P = \frac{1}{P} \cdot \nabla_\theta P = \frac{\nabla_\theta P}{P}
     $$
   - Look at that right-hand side: **$\frac{\nabla_\theta P}{P}$ is literally just the relative percentage nudge!**
   - **The Core Takeaway**: The logarithm $\nabla \log \pi$ was not invented to confuse students. It is simply the mathematical shorthand for **percentage growth**!

---

### Piece 4: The Discrete Wall & Policy Gradients (REINFORCE)

Now we can see the grand difference between ordinary deep learning and reinforcement learning:

1. **Supervised Learning (The Kindergarten Teacher)**:
   - When training a model on text, a human teacher provides the exact target marble: *"The next word must be 'cat'!"*
   - Because we know the target marble, we know exactly which knob to turn to add more 'cat' marbles. Gradients flow smoothly.

2. **The Teacher Leaves the Room (Reinforcement Learning)**:
   - Now ask the model to solve a novel math proof or write a 50-line program.
   - Nobody knows the exact sequence of words ahead of time. There is no human teacher.
   - The model reaches into its jar, draws a marble, places it on the table, and repeats this for $100$ steps:
     $$
     \text{"Let"} \to \text{"x"} \to \text{"="} \to \text{"5"} \dots
     $$
   - At the very end, a Python interpreter runs the code. A referee shouts:
     $$
     \text{"Correct! Score } R = +100 \text{ points!"}
     $$

3. **The Crisis of the Discrete Wall**:
   - When the machine pulled out the marble <kbd>"="</kbd>, that was a discrete, physical choice. You cannot take a derivative of a marble! A marble is a hard pebble; there is no such thing as a "fractional marble".
   - Furthermore, the Python interpreter is a black-box wall. You cannot backpropagate through an `if/else` statement in Python.
   - How can you update the internal knobs when you cannot push gradients through the choices?

4. **The Common-Sense Solution (Policy Gradient)**:
   - Think like a human coach:
     - Did the model draw a sequence of marbles that won $+100$ points?
     - Yes!
     - What do you want to happen tomorrow?
     - You want the machine to be **more likely to draw those exact same winning marbles**!
   - How much should you twist each knob?
     $$
     \text{Knob Adjustment } \Delta \theta = \text{Score } R \times (\text{Percentage nudge for that winning marble})
     $$
     $$
     \Delta \theta \propto R \times \frac{\nabla_\theta P}{P} = R \times \nabla_\theta \ln P
     $$
   - **That is the entire Policy Gradient Theorem (REINFORCE)!**
   - It is not an abstract piece of measure theory. It is the only sensible way to adjust dials when all you have is a scorecard: **increase the probability of every choice you made, scaled by the score you earned!**

---

### Piece 5: The Center of the Number Line (Subtracting the Average)

1. **The Positive Points Disaster**:
   - Suppose the referee gives scores from $0$ to $100$.
   - The model tries three different math attempts:
     - Attempt 1 gets $20$ points (bad mistakes).
     - Attempt 2 gets $50$ points (mediocre).
     - Attempt 3 gets $90$ points (brilliant).
   - Look at the scores: $+20$, $+50$, $+90$. All three numbers are positive!
   - If you use our naive rule, you will twist the knobs to increase the chance of Attempt 1, AND increase Attempt 2, AND increase Attempt 3!
   - But the jar can only hold $100\%$ of the marbles! If you try to increase the probability of every word, they violently fight for space. The knobs jerk back and forth, and learning becomes chaotic and unstable (high variance).

2. **Centering on the Number Line (The Baseline)**:
   - What is the natural center on the number line? **The Average!**
   - Add up the three scores and divide by $3$:
     $$
     \text{Average Score } b = \frac{20 + 50 + 90}{3} = \frac{160}{3} \approx 53.3 \text{ points}
     $$
   - Now, instead of rewarding by the raw score $R$, reward by **how much better or worse you were than average ($R - b$)**:
     - Attempt 1: $20 - 53.3 = \mathbf{-33.3}$ &mdash; **Negative! Twist knobs to suppress these mistakes!**
     - Attempt 2: $50 - 53.3 = \mathbf{-3.3}$ &mdash; **Slight penalty.**
     - Attempt 3: $90 - 53.3 = \mathbf{+36.7}$ &mdash; **Positive! Enthusiastically reward these brilliant steps!**

3. **The Zero-Sum Magic**:
   - Look what happens when you sum the three adjusted scores:
     $$
     (-33.3) + (-3.3) + (+36.7) \equiv 0
     $$
   - The nudges are strictly balanced on the number line! Exactly half the attempts are pushed down, and the best attempts are lifted up.
   - This simple act of subtracting the average is called **Baseline Subtraction**. It stabilizes reinforcement learning without introducing any mathematical bias whatsoever.

---

### Piece 6: The Modern Frontier (GRPO, PRMs, and Tree Search)

Now you possess the complete intuitive machinery to understand every modern reasoning breakthrough in AI:

1. **PPO vs. DeepSeek's GRPO (Firing the Expensive Private Tutor)**:
   - In classic PPO (Chapter 30), researchers trained a second, massive $70$-billion parameter neural network called the "Critic" just to guess what the average baseline $b$ should be at every word. It consumed half the GPU cluster and often hallucinated.
   - DeepSeek (Chapter 31) asked: *Why build a second giant neural network just to guess the average?*
   - Just take the prompt, roll the dice $4$ times to write $4$ answers on the chalkboard, compute the simple average of those $4$ scores, and subtract it!
   - That chalkboard average is your baseline. Zero extra neural network. Zero GPU memory waste. That is **Group Relative Policy Optimization (GRPO)**!

2. **Process Reward Models (Tasting the Soup at Every Step)**:
   - If an apprentice chef cooks a $20$-step French soup and accidentally drops in a teaspoon of sand at step $19$, the final bowl scores $0$ points.
   - Outcome models punish all $20$ steps equally.
   - A **Process Reward Model (PRM, Chapter 32)** stands by the pot and tastes the broth after *every single step*: Step 1 passed ($+1$), Step 2 passed ($+1$)... Step 19 failed ($0$). We keep the winning recipe up to step 18 and fix only step 19!

3. **Monte Carlo Tree Search (Counting Footsteps in a Maze)**:
   - Instead of picking words blindfolded, the model builds a tree of possibilities before speaking (Chapter 33).
   - At every fork in the road, it keeps track of two simple counters:
     - $N$: *How many times have I walked down this corridor?*
     - $Q$: *What was my average score down this corridor?*
   - If a corridor has a high average score $Q$, walk down it more (exploitation).
   - If a corridor has rarely been explored ($N \approx 0$), peek down it out of curiosity (exploration).
   - That balance is the **PUCT formula**. It allows models to spend "thinking compute" at test time to solve complex Olympiad math problems.

---

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
