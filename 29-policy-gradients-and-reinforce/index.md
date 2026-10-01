# Chapter 29: Language as an MDP & Policy Gradients (REINFORCE & Variance Reduction)

---

## Step 1: 3-Year-Old Intuition (The Chalk Line, Wiggling Knobs, and the Blindfolded Archer)

Let's start where all mathematics began: with counting and a piece of sidewalk chalk:

1. **The Number Line on the Sidewalk**:
   - Draw a straight line on the ground. Make a mark where you stand: **$0$**.
   - Walk strides forward: $+1, +2, \dots, +100$ (rewards and points). Walk strides backward: $-1, -2$ (penalties).
   - In reinforcement learning, our entire goal in life is very simple: **we want our machine's average score to walk further to the right on this chalk line.**

2. **The Box with Knobs and the Jar of Marbles**:
   - The language model is a wooden box with millions of volume knobs ($\boldsymbol{\theta}$).
   - Inside the box is a glass jar filled with $100{,}000$ marbles, each labeled with a word.
   - When you wiggle a knob, you change how many marbles of each word are packed into the jar. A "gradient" ($\nabla$) is simply asking: *If I twist this knob a tiny fraction to the right, how many more winning marbles fall into the jar?*

3. **The Kindergarten Teacher vs. The Blindfolded Archer**:
   - In **Supervised Learning** (Chapters 00 to 18), your teacher stands right behind you, holds your hands, and physically guides your fingers to point directly at the gold bullseye. For every single shot, you are shown the exact right move.
   - In **Reinforcement Learning**, your teacher leaves the room. You are handed a thick blindfold and spun around in a misty field. In your hands is a bow and a quiver of arrows.
   - You cannot see the target at all. You can only pull the string, point into the fog, and let go.
   - When an LLM generates text during RL, it is firing words into the fog &mdash; drawing one marble after another based on its current internal probabilities.

4. **The Coach's Whistle (The Black-Box Reward)**:
   - After your arrow lands with a *thud*, an invisible judge across the field shouts a single score through a megaphone:
     - *"Bullseye! 100 points!"*
     - or *"Missed the haystack entirely! 0 points!"*
   - Notice something vital: The judge does **not** tell you *how* to aim. The judge does not say *"raise your left elbow by two inches"*. They only announce how good the final result was.
   - You must figure out on your own which tiny muscle twitches were responsible for that high score.

5. **The Scorekeeper's Notebook (The Average Baseline)**:
   - If the judge shouts *"50 points!"*, is that great or terrible? You have no idea unless you know what you usually get!
   - If your historical average score is only 10 points, then 50 points is fantastic ($+40$ steps forward on the number line)! You want to remember the exact arm position that produced that shot.
   - But if your historical average is 90 points, then 50 points is a huge disappointment ($-40$ steps backward)! You want to steer away from whatever stance you just used.
   - By **subtracting your historical average score** from every shot, you center your scores around zero: bad shots become negative nudges backward, and good shots become positive nudges forward.

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│             THE POLICY GRADIENT REINFORCEMENT CYCLE                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Prompt [x] ──► [ Model Policy π_θ ] ──► Action tokens [y_1, ..., y_T]│
│                          ▲                               │             │
│                          │                               ▼             │
│                 Gradient Adjustment                Environment /       │
│                  Δθ ∝ ∇ log π_θ · (R - b)          Reward Function     │
│                          ▲                               │             │
│                          │                               ▼             │
│                  Surprise Advantage ◄──── Scalar Reward R(τ)           │
│                   A = R(τ) - Baseline b                                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 29.1:</strong> The reinforcement cycle: Language model acts as a policy, samples tokens, receives trajectory reward, and updates parameters proportionally to the surprise advantage.</figcaption>
</figure>

---

## Step 2: The Bridging Question

In Chapter 16, we learned Backpropagation: we computed the derivative of a continuous loss function with respect to every weight using the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{W}}
$$

In standard Supervised Fine-Tuning (<abbr title="Supervised Fine-Tuning">SFT</abbr>), every training sample comes with an exact golden answer $y^*$. The loss is standard Cross-Entropy:

$$
\mathcal{L}_{\text{SFT}}(\boldsymbol{\theta}) = -\log \pi_{\boldsymbol{\theta}}(y^* \mid x)
$$

However, when an LLM is tasked with solving complex problems (writing a 100-line Python script, proving a geometry theorem, or navigating an interactive dialogue), three mathematical barriers make supervised backpropagation impossible:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 29.1:</strong> The three mathematical barriers preventing standard backpropagation in autonomous generation.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">Barrier</th>
      <th align="left" width="38%">What Breaks in Calculus</th>
      <th align="left" width="40%">Concrete LLM Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Discrete Sampling</strong></td>
      <td>Sampling a token ($y_t \sim \operatorname{Categorical}(\mathbf{p}_t)$) is a discrete step. The mathematical derivative of a discrete choice $\frac{\partial \text{token}}{\partial \mathbf{W}}$ is <strong>undefined</strong> (or zero almost everywhere).</td>
      <td>Picking token index <kbd>4821</kbd> ("def") instead of <kbd>102</kbd> ("class") is a jump, not a smooth curve.</td>
    </tr>
    <tr>
      <td><strong>2. Black-Box Judges</strong></td>
      <td>The environment evaluating the output (a Python compiler, unit tests, a math verifier, or a human judge) is not a neural network. You cannot backpropagate gradients through a compiler's `if/else` checks!</td>
      <td>Running `pytest` returns Pass ($R=1$) or Fail ($R=0$). No gradient vector flows out of a terminal command.</td>
    </tr>
    <tr>
      <td><strong>3. Absence of Answer Keys</strong></td>
      <td>For an open-ended math proof or code design, there are millions of valid paths. Forcing the model to copy one fixed human demonstration prevents it from discovering simpler or better solutions.</td>
      <td>A proof can use induction, contradiction, or algebra. SFT penalizes valid methods if they differ from the human label!</td>
    </tr>
  </tbody>
</table>

<br>

The central bridging question is:
$$\text{How do we calculate the exact gradient of expected reward } \nabla_{\boldsymbol{\theta}} \mathbb{E}[R] \text{ when actions are discrete and the reward function is a non-differentiable black box?}$$

---

## Step 3: The Exact Math & Formula

### 1. From Marbles and Paper Strips to Language Generation (Demystifying the Markov Property)

In standard textbooks, authors often intimidate students by declaring: *"Let text generation be defined as a discrete-time Markov Decision Process tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$."*

Let's discard the academic jargon and look at what is physically happening on the desk:

1. **The Paper Strip (The State $s_t$)**:
   - Imagine a paper strip rolling out of a printer.
   - At the beginning, the prompt $x$ is printed on the strip: e.g. <samp>"What is 2 + 2?"</samp>.
   - Every time the model generates a word, it prints that word onto the end of the strip.
   - At step $t$, the state $s_t$ is simply **everything currently written on the paper strip**:
     $$
     s_t = (x, y_1, y_2, \dots, y_{t-1}) = (x, y_{\lt t})
     $$
   - You don't need an abstract state space $\mathcal{S}$ &mdash; the state is literally just the readable text on the paper tape!

2. **The Vocabulary Jar (The Action $a_t$)**:
   - Inside the machine is a glass jar containing $|\mathcal{V}|$ marbles (where $|\mathcal{V}| \approx 32{,}000$ to $128{,}000$ words).
   - Taking an "action" $a_t$ simply means **reaching into the jar and pulling out one word $y_t$** to print onto the paper strip.

3. **Demystifying the "Markov Property" (No History Amnesia)**:
   - In physics, a process is called *Markovian* if the next event depends **only on the current state, not on how you reached it**.
   - Students often ask: *"Doesn't the next word depend heavily on what was said 10 words ago? How can that be Markovian?"*
   - Look at your paper strip: **All 10 past words are already printed right there on the strip!**
   - Because the strip $s_t$ already contains the complete history, the machine does not need any hidden time machine or external memory. Everything it needs to know is printed on the tape in front of it right now. Conditioned on $s_t$, the next marble depends on nothing else.

4. **The Probability of an Entire Sentence (Multiplying Marbles)**:
   - The model has internal knobs $\boldsymbol{\theta}$. Given the paper strip $s_t$, the machine outputs a probability for each marble in the jar via softmax:
     $$
     \pi_{\boldsymbol{\theta}}(y_t \mid s_t) = \frac{\exp(z_{t, y_t})}{\sum_{v \in \mathcal{V}} \exp(z_{t, v})}
     $$
   - When the model writes a full sentence $\tau = (y_1, y_2, \dots, y_T)$, what is the chance of rolling that exact sequence of marbles?
   - From basic counting, you simply multiply the chances of each individual marble drawn:
     $$
     P(\tau; \boldsymbol{\theta}) = \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(y_t \mid s_t)
     $$
   - At the end of the sentence, an external evaluator (a Python runner, math checker, or human) awards a scalar score $R(\tau) \in \mathbb{R}$ on the number line.

---

### 2. The Expectation Gradient Paradox & The Score Function Trick

The optimization objective in reinforcement learning is to maximize the expected reward over all possible sampled trajectories:

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[R(\tau)] = \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

<fieldset>
<legend><strong>The Fundamental Calculus Paradox: Where Are the Parameters $\boldsymbol{\theta}$?</strong></legend>
<p>Look closely at where $\boldsymbol{\theta}$ appears in the equation:</p>
<ul>
  <li>In standard Supervised Learning: $\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{x \sim \mathcal{D}} [f_{\boldsymbol{\theta}}(x)]$. The dataset $\mathcal{D}$ is static (it does not change when the model learns). The parameters $\boldsymbol{\theta}$ sit <em>inside</em> the function $f_{\boldsymbol{\theta}}$. You can slide the derivative directly inside the expectation: $\nabla_{\boldsymbol{\theta}} \mathbb{E}[f_{\boldsymbol{\theta}}] = \mathbb{E}[\nabla_{\boldsymbol{\theta}} f_{\boldsymbol{\theta}}]$.</li>
  <li>In Reinforcement Learning: The reward $R(\tau)$ is an external judge (e.g., did the Python tests pass?). <strong>The reward function $R(\tau)$ does not contain $\boldsymbol{\theta}$ at all!</strong></li>
  <li>Instead, the parameters $\boldsymbol{\theta}$ live in the <em>subscript</em> &mdash; they govern the <strong>probability distribution $P(\tau; \boldsymbol{\theta})$</strong> that chooses which trajectories get generated!</li>
</ul>
</fieldset>

Taking the gradient with respect to $\boldsymbol{\theta}$:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau) = \sum_{\tau} \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

Now we face a massive computational barrier:
1. **The Trillion-Sentence Universe**: The sum $\sum_{\tau}$ spans every possible sentence the model could write. For a modest sequence length of $T = 500$ and vocabulary $|\mathcal{V}| = 100{,}000$, there are $100{,}000^{500} = 10^{2500}$ possible trajectories &mdash; vastly more than the total number of atoms in the observable universe ($10^{80}$)! No computer can sum over all $\tau$.
2. **We Need an Expectation**: In machine learning, whenever a sum over an impossible universe appears, we convert it into an expectation $\sum_u P(u) g(u) = \mathbb{E}_{u \sim P}[g(u)]$. An expectation can be approximated on a GPU using **Monte Carlo sampling** (simply sample $N$ trajectories and compute their average).
3. **The Obstacle**: The sum contains $\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})$, which is **not** a valid probability distribution! $\nabla P$ contains negative numbers and does not sum to 1. You cannot sample from $\nabla P$.

To solve this, we perform the foundational **Log-Derivative Trick** (also known as the **Score Function Estimator**) in three algebraic steps:

#### Step A: Multiply and Divide by $P(\tau; \boldsymbol{\theta})$
Since $P(\tau; \boldsymbol{\theta}) > 0$ for all valid trajectories, we multiply and divide the gradient by $P(\tau; \boldsymbol{\theta})$:

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \cdot \frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})}
$$

#### Step B: Apply the Derivative of the Natural Logarithm (The Relative Percentage Nudge)

Recall from elementary calculus that $\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}$. Applying this in reverse:

$$
\frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})} \equiv \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

<fieldset>
<legend><strong>Feynman's Secret: Why Does the Logarithm Appear? (Relative Percentage Growth)</strong></legend>
<p>Students often wonder: <em>Why did researchers inject a logarithm into policy gradients?</em></p>
<p>Think about two words in your vocabulary:</p>
<ul>
  <li><strong>Common Word A (<kbd>"the"</kbd>)</strong>: Current probability $P = 0.500$ ($500$ marbles out of $1{,}000$). If a knob adds $1$ marble ($\Delta P = +0.001$), its probability shifts to $0.501$. That is a tiny <strong>$+0.2\%$</strong> relative change &mdash; imperceptible noise.</li>
  <li><strong>Rare Reasoning Word B (<kbd>"hypotenuse"</kbd>)</strong>: Current probability $P = 0.001$ ($1$ marble out of $1{,}000$). If a knob adds $1$ marble ($\Delta P = +0.001$), its probability doubles to $0.002$. That is a massive <strong>$+100\%$</strong> surge &mdash; a genuine mathematical breakthrough!</li>
</ul>
<p>On the chalk number line, both words gained the exact same absolute step ($\Delta P = 0.001$). But in reality, discovering a rare reasoning token is vastly more meaningful than nudging a common article. What naturally measures this real-world impact?</p>
<p>The <strong>relative percentage change</strong>:</p>
$$
\frac{\text{change in probability}}{\text{current probability}} = \frac{\nabla_{\boldsymbol{\theta}} P}{P} \equiv \nabla_{\boldsymbol{\theta}} \log P
$$
<p><strong>The natural logarithm was not chosen out of mathematical vanity. It is simply the exact calculus expression for relative percentage growth!</strong></p>
</fieldset>

Therefore:

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

#### Step C: Re-establish the Expectation
Substituting this identity back into our gradient sum:

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{\tau} P(\tau; \boldsymbol{\theta}) \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right] \\
&= \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right]
\end{aligned}
$$

Because $P(\tau; \boldsymbol{\theta})$ is back on the outside, this is once again a genuine expectation! We can now approximate the true gradient on our computer by simply generating sample trajectories using our LLM and averaging them!

Now expand the trajectory log-probability:

$$
\log P(\tau; \boldsymbol{\theta}) = \log \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) = \sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

Taking the gradient distributes across the sum of tokens:

$$
\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

This yields the celebrated **Policy Gradient Theorem (REINFORCE)**:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]
$$

<fieldset>
<legend><strong>The Grand Unification: Policy Gradient Is Weighted Cross-Entropy!</strong></legend>
<p>Look at the mathematical structure of the policy gradient side-by-side with Supervised Fine-Tuning (SFT):</p>
<table border="1" cellpadding="6" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="30%">Paradigm</th>
      <th align="left" width="40%">Parameter Update Formula</th>
      <th align="left" width="30%">What It Does</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Supervised Fine-Tuning (SFT)</strong></td>
      <td>$\Delta \boldsymbol{\theta} \propto + \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_t^* \mid s_t)$</td>
      <td>Pulls parameters towards the human teacher's word $y^*$ with fixed weight $+1$.</td>
    </tr>
    <tr>
      <td><strong>Policy Gradient (RL)</strong></td>
      <td>$\Delta \boldsymbol{\theta} \propto + \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \cdot \mathbf{R(\tau)}$</td>
      <td>Pulls parameters towards the model's <em>own self-generated word</em> $a_t$, scaled by the reward volume knob $R(\tau)$!</td>
    </tr>
  </tbody>
</table>
<p>If $R(\tau) = +10$, the model performs standard cross-entropy on its own words with 10 times the normal strength! If $R(\tau) = 0$, nothing happens. If $R(\tau) < 0$, it pushes probability <em>away</em> from those words!</p>
<p><strong>Reinforcement learning does not require a magical new gradient engine; it is literally weighted supervised learning on the model's own self-generated attempts!</strong></p>
</fieldset>

---

### 3. The Baseline Subtraction Theorem (Slashing Variance)

While raw REINFORCE is mathematically correct, in practice it suffers from a catastrophic flaw: **extreme sample variance**.

<fieldset>
<legend><strong>The Catastrophe of All-Positive Rewards</strong></legend>
<p>Suppose an LLM is solving math problems, and the reward score ranges from 0 to 100 points. The model samples three different solution attempts:</p>
<ul>
  <li>Attempt 1 gets a mediocre score: $R = 90$</li>
  <li>Attempt 2 gets a good score: $R = 95$</li>
  <li>Attempt 3 gets a flawless score: $R = 100$</li>
</ul>
<p>Without a baseline, look at what REINFORCE does: It updates the weights with $+90$, $+95$, and $+100$! <strong>All three attempts get massive probability increases!</strong> Even Attempt 1 &mdash; which was the worst in the cohort &mdash; has its bad habits strongly reinforced. The model barely learns that Attempt 3 was superior.</p>
<p>Now introduce an average baseline $b = 95$ (the cohort mean):</p>
<ul>
  <li>Attempt 1 receives advantage: $A_1 = 90 - 95 = \mathbf{-5}$ (penalized!)</li>
  <li>Attempt 2 receives advantage: $A_2 = 95 - 95 = \mathbf{0}$ (neutral)</li>
  <li>Attempt 3 receives advantage: $A_3 = 100 - 95 = \mathbf{+5}$ (rewarded!)</li>
</ul>
<p>Suddenly, the learning signal is sharp, centered, and high-contrast! Good moves are reinforced, bad moves are suppressed, and gradient variance collapses.</p>
</fieldset>

To implement this, we subtract a **baseline** $b(s_t)$ that depends only on state $s_t$, but does not depend on the selected action $a_t$:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

#### Step-by-Step Proof of Zero Bias:
We must prove that subtracting $b(s_t)$ does not distort or bias the true gradient direction.
Consider the expectation of the baseline term over all possible actions at step $t$:

$$
\begin{aligned}
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] &= \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)} \right) b(s_t) && \text{[Expand expectation and log-derivative]} \\
&= b(s_t) \sum_{a_t \in \mathcal{V}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) && \text{[Cancel } \pi \text{ and factor out } b(s_t)\text{]} \\
&= b(s_t) \nabla_{\boldsymbol{\theta}} \left( \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right) && \text{[Gradient of a finite sum is sum of gradients]}
\end{aligned}
$$

Now observe the term inside the parenthesis: $\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$.
Because $\pi_{\boldsymbol{\theta}}$ is a valid probability distribution produced by softmax, the sum of probabilities over the entire vocabulary must equal 1:

$$
\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \equiv 1.0
$$

The derivative of a constant is strictly zero: $\nabla_{\boldsymbol{\theta}}(1.0) = \mathbf{0}$. Therefore:

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = b(s_t) \cdot \mathbf{0} \equiv \mathbf{0}
$$

Subtracting any baseline $b(s_t)$ has **strictly zero bias** on the expected gradient! The expectation remains 100% mathematically exact, while the variance drops by orders of magnitude!

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="1992">1992</time> &mdash; <strong>Ronald J. Williams (REINFORCE)</strong></dt>
  <dd>Williams introduced the <em>REINFORCE</em> algorithm (<cite>"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"</cite>), deriving the score function estimator for artificial neural networks and establishing baseline subtraction.</dd>

  <dt><time datetime="1999">1999</time> &mdash; <strong>Richard Sutton, David McAllester, Satinder Singh, & Yishay Mansour</strong></dt>
  <dd>Proved the general <em>Policy Gradient Theorem</em> for function approximators under arbitrary MDP state distributions, showing that we do not need to know the state distribution derivative $\nabla_{\boldsymbol{\theta}} d^{\pi}(s)$ to compute exact policy gradients.</dd>

  <dt><time datetime="2016">2016–2020</time> &mdash; <strong>The LLM RL Crisis</strong></dt>
  <dd>Early attempts to apply raw REINFORCE directly to language models (e.g. optimizing BLEU score in machine translation) suffered from unstable training, severe reward hacking, and mode collapse. The policy would quickly learn to generate repetitive punctuation or exploit edge-case reward loopholes, demanding dense actor-critic baselines (Chapter 30) and group normalization (Chapter 31).</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us trace a concrete policy gradient update by hand with tiny numbers.

### The Toy Setup
- Vocabulary: $\mathcal{V} = \{A, B, C\}$ ($|\mathcal{V}| = 3$).
- Single-token action $a \in \{A, B, C\}$.
- Model logits $\mathbf{z} = [z_A, z_B, z_C]^\top$.
- Currently, our model parameters produce logits:
  $$
  \mathbf{z} = [1.0, \; 0.0, \; -1.0]^\top
  $$
- Learning rate $\eta = 0.1$.

---

### Phase 1: Forward Pass & Softmax Probabilities

First, compute the exponentials:
- $e^{z_A} = e^{1.0} \approx 2.718$
- $e^{z_B} = e^{0.0} = 1.000$
- $e^{z_C} = e^{-1.0} \approx 0.368$
- Sum: $\sum e^{z_i} \approx 2.718 + 1.000 + 0.368 = 4.086$

Probabilities $\pi_{\boldsymbol{\theta}}(a)$:
$$
\pi(A) = \frac{2.718}{4.086} \approx 0.665 \quad (<meter min="0" max="1" value="0.665"></meter>)
$$
$$
\pi(B) = \frac{1.000}{4.086} \approx 0.245 \quad (<meter min="0" max="1" value="0.245"></meter>)
$$
$$
\pi(C) = \frac{0.368}{4.086} \approx 0.090 \quad (<meter min="0" max="1" value="0.090"></meter>)
$$

---

### Phase 2: Action Sampling & Environment Feedback

Suppose our stochastic sampling selects token <kbd>"B"</kbd> ($a = B$):
- The environment evaluates <kbd>"B"</kbd> and returns a reward:
  $$
  R = 2.0
  $$
- Our historical baseline average is:
  $$
  b = 0.5
  $$
- The **Advantage** (surprise) is:
  $$
  A = R - b = 2.0 - 0.5 = +1.5
  $$
  Because $A > 0$, the action <kbd>"B"</kbd> was much better than average! We want to increase its probability.

---

### Phase 3: Exact Logit Gradients

Recall the gradient of log-softmax with respect to logit $z_i$ when action $a$ was chosen:

$$
\frac{\partial \log \pi(a)}{\partial z_i} = \mathbb{I}(i = a) - \pi(i)
$$

For our chosen action $a = B$:
- For token $A$: $\frac{\partial \log \pi(B)}{\partial z_A} = 0 - \pi(A) = -0.665$
- For token $B$: $\frac{\partial \log \pi(B)}{\partial z_B} = 1 - \pi(B) = 1 - 0.245 = +0.755$
- For token $C$: $\frac{\partial \log \pi(B)}{\partial z_C} = 0 - \pi(C) = -0.090$

Check the sum of gradients: $-0.665 + 0.755 - 0.090 = 0.000$ (perfect zero-sum!).

---

### Phase 4: Policy Gradient Step

The policy gradient with baseline is:

$$
\mathbf{g} = \nabla_{\mathbf{z}} \log \pi(B) \cdot (R - b) = \begin{bmatrix} -0.665 \\ +0.755 \\ -0.090 \end{bmatrix} \times 1.5 = \begin{bmatrix} -0.9975 \\ +1.1325 \\ -0.1350 \end{bmatrix}
$$

Applying gradient ascent with step size $\eta = 0.1$:

$$
\mathbf{z}_{\text{new}} = \mathbf{z} + \eta \mathbf{g} = \begin{bmatrix} 1.0 \\ 0.0 \\ -1.0 \end{bmatrix} + 0.1 \begin{bmatrix} -0.9975 \\ +1.1325 \\ -0.1350 \end{bmatrix} = \begin{bmatrix} 0.90025 \\ 0.11325 \\ -1.01350 \end{bmatrix}
$$

Let us verify the new probabilities under $\mathbf{z}_{\text{new}}$:
- $e^{0.90025} \approx 2.460$
- $e^{0.11325} \approx 1.120$
- $e^{-1.01350} \approx 0.363$
- New sum $\approx 3.943$

New probabilities:
- $\pi_{\text{new}}(A) = \frac{2.460}{3.943} \approx 0.624$ (dropped from $0.665$)
- $\pi_{\text{new}}(B) = \frac{1.120}{3.943} \approx \mathbf{0.284}$ (<mark>increased from $0.245$!</mark>)
- $\pi_{\text{new}}(C) = \frac{0.363}{3.943} \approx 0.092$

Action <kbd>"B"</kbd> received an instant $+3.9\%$ probability boost because it yielded a positive advantage ($+1.5$)!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>The Policy Gradient Theorem bridges the discrete chasm of language generation by using the <strong>Log-Derivative Trick</strong>: we never differentiate through the sampled token or the reward function; instead, we scale the model's standard cross-entropy gradients by the scalar surprise of the reward.</p>
<p>Subtracting a state baseline leaves the expected gradient mathematically unchanged while drastically taming the catastrophic variance of sampling across vast token vocabularies.</p>
</fieldset>
