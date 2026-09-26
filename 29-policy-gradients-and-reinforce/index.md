# Chapter 29: Language as an MDP & Policy Gradients (REINFORCE & Variance Reduction)

---

## Step 1: 3-Year-Old Intuition (The Blindfolded Archer and the Whispering Coach)

Imagine you are standing in a misty field wearing a thick black blindfold. In your hands is a toy bow and arrow:

1. **The Blindfolded Archer (The Language Model)**:
   - You cannot see the target at all.
   - You can only pull the string, point the arrow into the fog, and release.
   - When you speak words, you are firing arrows into the fog &mdash; choosing one token after another based purely on gut feeling (your internal probabilities).

2. **The Coach's Whistle (The Reward)**:
   - After your arrow lands with a *thud*, an invisible judge across the field shouts a single score through a megaphone:
     - *"Bullseye! 100 points!"*
     - or *"Missed the haystack entirely! 0 points!"*
   - Notice something vital: The judge does **not** tell you *how* to aim. The judge does not say *"raise your left elbow by two inches"*. They only tell you how good the final result was.

3. **The Scorekeeper's Notebook (The Average Baseline)**:
   - If the judge shouts *"50 points!"*, is that great or terrible? You have no idea unless you know what you usually get!
   - If your historical average score is only 10 points, then 50 points is fantastic! You want to remember the exact arm position that produced that shot.
   - But if your historical average is 90 points, then 50 points is a huge disappointment! You want to steer away from whatever stance you just used.
   - By **subtracting the average score** from every shot, you only adjust your muscles when a shot was *better than your everyday expectation*.

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

However, when an LLM generates text during real-world tasks (writing Python code, solving a math proof, answering a user query), something fundamental breaks:
1. **Sampling is Non-Differentiable**: The model converts logits into tokens by sampling from a categorical distribution ($y_t \sim \operatorname{Categorical}(\mathbf{p}_t)$) or taking $\operatorname{argmax}$. You cannot compute the derivative of a discrete choice: $\frac{\partial \text{token}}{\partial \mathbf{W}}$ does not exist!
2. **The Environment is a Black Box**: A Python compiler or human reviewer evaluates the full output: either the unit tests pass ($R=1$) or they fail ($R=0$). We cannot backpropagate through a Python compiler or a human judge.

The bridging question is:
$$\text{How do we mathematically calculate the exact gradient of an expected reward } \nabla_{\boldsymbol{\theta}} \mathbb{E}[R] \text{ when the actions are discrete and the reward function is a non-differentiable black box?}$$

---

## Step 3: The Exact Math & Formula

### 1. Language Generation as a Markov Decision Process (MDP)

We formulate autoregressive sequence generation as a discrete-time, finite-horizon <abbr title="Markov Decision Process">MDP</abbr> defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$:

- **State Space ($\mathcal{S}$)**: At time step $t$, the state $s_t$ consists of the initial prompt $x$ and all generated tokens up to step $t-1$:
  $$
  s_t = (x, y_1, y_2, \dots, y_{t-1}) = (x, y_{\lt t}) \in \mathcal{S}
  $$
- **Action Space ($\mathcal{A}$)**: The action $a_t$ is the choice of the next token from the discrete vocabulary $\mathcal{V}$:
  $$
  a_t = y_t \in \mathcal{V} \quad (|\mathcal{V}| \approx 32{,}000 \text{ to } 128{,}000)
  $$
- **Transition Dynamics ($\mathcal{P}$)**: The state transition is deterministic string concatenation:
  $$
  s_{t+1} = [s_t, a_t] = (x, y_1, \dots, y_t)
  $$
  with probability $\mathcal{P}(s_{t+1} \mid s_t, a_t) = 1$.
- **Policy ($\pi_{\boldsymbol{\theta}}$)**: The language model parameterized by weights $\boldsymbol{\theta} \in \mathbb{R}^D$ outputs a categorical probability distribution over all tokens in $\mathcal{V}$ given state $s_t$:
  $$
  \pi_{\boldsymbol{\theta}}(a_t \mid s_t) = \operatorname{softmax}\left(\mathbf{z}_t\right)_{a_t} = \frac{\exp\left(z_{t, a_t}\right)}{\sum_{v \in \mathcal{V}} \exp\left(z_{t, v}\right)}
  $$
- **Trajectory ($\tau$) and Return ($R(\tau)$)**: A full generation episode produces a complete trajectory $\tau = (s_1, a_1, s_2, a_2, \dots, s_T, a_T)$. The scalar reward $R(\tau) \in \mathbb{R}$ is assigned upon terminal token generation (e.g., EOS token).

The total probability of generating trajectory $\tau$ under policy $\boldsymbol{\theta}$ is:

$$
P(\tau; \boldsymbol{\theta}) = P(s_1) \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t) = \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

---

### 2. The Expected Objective and the Score Function Trick

The optimization objective is to maximize the expected return over all possible sampled trajectories:

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[R(\tau)] = \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

Taking the gradient with respect to model parameters $\boldsymbol{\theta}$:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau) = \sum_{\tau} \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

Notice that we cannot approximate this sum via Monte Carlo sampling because it contains $\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})$, which is not a valid probability distribution!

To fix this, we apply the foundational **Log-Derivative Trick** (also known as the **Score Function Estimator**):
Recall from basic calculus that $\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}$, which implies:

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

Substituting this identity back into our gradient equation:

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{\tau} P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right]
\end{aligned}
$$

Now look at the term $\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})$:

$$
\log P(\tau; \boldsymbol{\theta}) = \log \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) = \sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

Taking the gradient eliminates the product and converts it into a clean sum over tokens:

$$
\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

This yields the celebrated **Policy Gradient Theorem (REINFORCE)**:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]
$$

<fieldset>
<legend><strong>Why This Formula Is Magic</strong></legend>
<p>Look carefully at what just happened: We took the derivative of an expectation over a non-differentiable reward $R(\tau)$, and transformed it into the standard gradient of the model's own log-probabilities $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}$ multiplied by a scalar number $R(\tau)$!</p>
<p>The term $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$ is identical to the gradient of the standard cross-entropy loss from Chapter 15. The only difference is that instead of pushing probability toward a fixed ground-truth token, we scale the push by the magnitude of the reward $R(\tau)$!</p>
</fieldset>

---

### 3. The Baseline Subtraction Theorem (Slashing Variance)

The raw REINFORCE estimator has a devastating defect: **extreme sample variance**.
If all rewards are positive (e.g. $R \in [10, 100]$), *every single sampled trajectory* gets pushed upward! Even bad outputs get their probabilities increased, just slightly less than good outputs.

To solve this, we subtract a **baseline** $b(s_t)$ that does not depend on the selected action $a_t$:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

#### Mathematical Proof of Zero Bias:
We must prove that subtracting $b(s_t)$ does not change the expected gradient.
Consider the expectation over actions at step $t$:

$$
\begin{aligned}
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] &= \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)} b(s_t) \\
&= b(s_t) \sum_{a_t \in \mathcal{V}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \\
&= b(s_t) \nabla_{\boldsymbol{\theta}} \left( \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right)
\end{aligned}
$$

Because probabilities over the entire vocabulary must sum to exactly 1 by definition of the softmax function:

$$
\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \equiv 1 \implies \nabla_{\boldsymbol{\theta}}(1) = 0
$$

Therefore:

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = b(s_t) \cdot \mathbf{0} = \mathbf{0}
$$

Subtracting any baseline $b(s_t)$ has **strictly zero bias** on the gradient direction, but it centers the reward signal, drastically reducing the variance $\operatorname{Var}(\hat{\mathbf{g}})$!

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
