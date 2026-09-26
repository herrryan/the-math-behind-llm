# Chapter 30: The Actor-Critic Architecture & PPO (Proximal Policy Optimization)

---

## Step 1: 3-Year-Old Intuition (The Stage Actor, the Director, and the Elastic Safety Tether)

Imagine a theatre rehearsal on Broadway with three people and a safety harness:

1. **The Stage Actor (The Actor Model $\pi_{\boldsymbol{\theta}}$)**:
   - The actor stands under the stage spotlight, improvising spoken lines in front of an empty auditorium.
   - The actor is full of spontaneous ideas &mdash; trying out different words, pauses, and gestures.

2. **The Director in the Front Row (The Critic Model $V_{\boldsymbol{\phi}}$)**:
   - Sitting in the third row is an experienced director holding a clipboard.
   - At every single sentence, the director writes down a predicted final rating:
     - *"At this moment in the scene, I predict this performance will earn an 8 out of 10."*
   - When the actor finishes the entire play and the audience votes:
     - If the audience gives a 9 out of 10, the performance was $+1$ point better than the director expected! That positive surprise is the **Advantage**.
     - If the audience gives a 5 out of 10, the performance was $-3$ points worse than expected. The actor gets an immediate critique.

3. **The Elastic Safety Tether (The PPO Clipping Mechanism)**:
   - What happens if the actor tells a hilarious joke and the audience laughs uncontrollably?
   - Without a safety tether, the actor might get so excited that the next day, they abandon the entire script and spend two hours doing crazy cartwheels and shouting knock-knock jokes! The play is ruined.
   - To stop this, we clip an **elastic safety tether** to the actor's waist.
   - The tether allows the actor to adjust their delivery by at most **$20\%$** in any single rehearsal. Even if an idea got a standing ovation, the actor is physically prevented from taking a massive leap into unpredictable territory.

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│               THE 4-MODEL LLM RLHF ACTOR-CRITIC CLUSTER                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Prompt [x] ────────────────────┬────────────────────┐                │
│                                  │                    │                │
│                                  ▼                    ▼                │
│                         [ Reference Model ]   [ Actor Policy ]         │
│                           (π_ref - Frozen)      (π_θ - Trains)         │
│                                  │                    │                │
│                                  │                    ▼                │
│                                  │           Generated Token y_t       │
│                                  │                    │                │
│                                  ▼                    ▼                │
│                           KL Divergence Penalty:                      │
│                           D_KL(π_θ || π_ref)                           │
│                                  │                                     │
│                                  ▼                                     │
│   [ Reward Model ] ──────► Total Reward r_t ──► [ Critic Model ]       │
│     (R_ψ - Frozen)          (Task + KL)           (V_φ - Trains)       │
│                                                       │                │
│                                                       ▼                │
│                             PPO Clipped Update ◄── Generalized         │
│                             min(r·A, clip(r)·A)    Advantage (GAE)     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 30.1:</strong> The 4-model RLHF training loop: Actor generates tokens, Reference prevents drift, Reward model evaluates quality, and Critic computes GAE advantages for the PPO clipped update.</figcaption>
</figure>

---

## Step 2: The Bridging Question

In Chapter 29, we derived the REINFORCE policy gradient:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

While mathematically elegant, running raw REINFORCE on Large Language Models encounters two catastrophic engineering bottlenecks:
1. **The Delayed Credit Bottleneck**: In a 1,000-token response, the model receives a single reward $R(\tau)$ at the very end. If token 20 was a stroke of genius but token 995 hallucinated a false fact, the final reward is $0$. REINFORCE punishes all 1,000 tokens equally! We need a fine-grained, token-level reward signal that evaluates partial progress.
2. **The Step-Size Cliff (Policy Collapse)**: In deep neural networks, a tiny step in weight space $\Delta \boldsymbol{\theta}$ can cause an enormous, non-linear shift in the probability distribution $\pi_{\boldsymbol{\theta}}$. If the policy shifts too far in a single gradient step, it starts generating complete gibberish. Once a language model falls off this performance cliff, all subsequent training samples become nonsense and the model never recovers.

The central bridging question is:
$$\text{How do we estimate token-level advantages in real time using a learned value function, while strictly bounding policy divergence to guarantee monotonically stable training?}$$

---

## Step 3: The Exact Math & Formula

### 1. The 4-Model System in LLM RLHF

Modern classical RLHF coordinates four separate language models simultaneously:

1. **Actor Policy $\pi_{\boldsymbol{\theta}}$**: The generative model whose parameters $\boldsymbol{\theta} \in \mathbb{R}^D$ are updated to maximize human preference.
2. **Critic (Value Model) $V_{\boldsymbol{\phi}}$**: A language model backbone with a scalar regression head replacing the unembedding layer. Parameters $\boldsymbol{\phi}$ are updated to predict expected discounted return: $V_{\boldsymbol{\phi}}(s_t) \approx \mathbb{E}\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t \right]$.
3. **Reference Policy $\pi_{\text{ref}}$**: A frozen copy of the original Supervised Fine-Tuned (<abbr title="Supervised Fine-Tuning">SFT</abbr>) model.
4. **Reward Model $R_{\boldsymbol{\psi}}$**: A frozen preference model trained on human comparison pairs $(y_w \succ y_l)$ via the Bradley-Terry objective (Chapter 19).

---

### 2. Token-Level Reward with KL Penalty

To prevent the Actor from exploiting loopholes in the Reward Model (known as <dfn id="def-reward-hacking">Reward Hacking</dfn>) and from losing its broad language capabilities, we penalize token divergence from the reference policy:

$$
r_t = \begin{cases}
-\beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} \right) & \text{for intermediate tokens } t < T \\
R_{\boldsymbol{\psi}}(x, y) - \beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_T \mid s_T)}{\pi_{\text{ref}}(a_T \mid s_T)} \right) & \text{for the terminal token } t = T
\end{cases}
$$

where $\beta > 0$ is the KL penalty coefficient. If the Actor increases the probability of token $a_t$ far beyond what $\pi_{\text{ref}}$ assigns, $\log(\pi_{\boldsymbol{\theta}} / \pi_{\text{ref}}) > 0$, subtracting points from the reward.

---

### 3. Generalized Advantage Estimation (GAE)

To evaluate whether token $a_t$ was better or worse than expected at state $s_t$, we first compute the **Temporal Difference (<abbr title="Temporal Difference">TD</abbr>) residual** $\delta_t^V$:

$$
\delta_t^V = r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1}) - V_{\boldsymbol{\phi}}(s_t)
$$

where $\gamma \in (0, 1]$ is the discount factor (typically $\gamma = 1.0$ for finite language generation).
To strike the optimal mathematical trade-off between bias (low $\lambda$) and variance (high $\lambda$), we use <dfn id="def-gae">Generalized Advantage Estimation</dfn> ($\text{GAE}(\gamma, \lambda)$):

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V
$$

In recursive form:

$$
\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}
$$

- When $\lambda = 0$: $\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ (lowest variance, highest bias from critic errors).
- When $\lambda = 1$: $\hat{A}_t = \sum_{k=t}^T \gamma^{k-t} r_k - V(s_t)$ (unbiased Monte Carlo return minus baseline, higher variance).
- In LLM post-training, researchers typically set $\gamma = 1.0$ and $\lambda = 0.95$.

---

### 4. The PPO Clipped Surrogate Objective

Let the probability ratio between the current policy $\pi_{\boldsymbol{\theta}}$ and the old policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$ that collected the rollout data be:

$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \quad \text{with } r_t(\boldsymbol{\theta}_{\text{old}}) = 1.0
$$

The PPO Clipped Surrogate Objective is:

$$
\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta})\hat{A}_t, \; \operatorname{clip}\left(r_t(\boldsymbol{\theta}), \, 1-\epsilon, \, 1+\epsilon\right)\hat{A}_t \right) \right]
$$

where $\epsilon$ is a clipping hyperparameter (typically $\epsilon = 0.2$, restricting the ratio to $[0.8, 1.2]$).

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 30.1:</strong> PPO clipping behavior across advantage and ratio regimes.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="20%">Advantage Regime</th>
      <th align="left" width="25%">Ratio $r_t(\boldsymbol{\theta})$</th>
      <th align="left" width="25%">Objective Value</th>
      <th align="left" width="30%">Gradient Behavior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Positive ($\hat{A}_t > 0$)</strong><br>Action was better than expected</td>
      <td>$r_t(\boldsymbol{\theta}) \le 1 + \epsilon$<br>(Within safety margin)</td>
      <td>$r_t(\boldsymbol{\theta}) \hat{A}_t$</td>
      <td><mark>Positive gradient</mark>: increases token probability further.</td>
    </tr>
    <tr>
      <td><strong>Positive ($\hat{A}_t > 0$)</strong><br>Action was better than expected</td>
      <td>$r_t(\boldsymbol{\theta}) > 1 + \epsilon$<br>(Policy already shifted enough)</td>
      <td>$(1 + \epsilon) \hat{A}_t$</td>
      <td><strong>Zero gradient</strong>: prevents excessive parameter updates.</td>
    </tr>
    <tr>
      <td><strong>Negative ($\hat{A}_t < 0$)</strong><br>Action was worse than expected</td>
      <td>$r_t(\boldsymbol{\theta}) \ge 1 - \epsilon$<br>(Within safety margin)</td>
      <td>$r_t(\boldsymbol{\theta}) \hat{A}_t$</td>
      <td><mark>Negative gradient</mark>: suppresses token probability.</td>
    </tr>
    <tr>
      <td><strong>Negative ($\hat{A}_t < 0$)</strong><br>Action was worse than expected</td>
      <td>$r_t(\boldsymbol{\theta}) < 1 - \epsilon$<br>(Token already heavily suppressed)</td>
      <td>$(1 - \epsilon) \hat{A}_t$</td>
      <td><strong>Zero gradient</strong>: prevents driving probability to zero too aggressively.</td>
    </tr>
  </tbody>
</table>

---

### 5. The Critic Value Loss

The Critic network $V_{\boldsymbol{\phi}}$ is trained concurrently via Mean Squared Error (<abbr title="Mean Squared Error">MSE</abbr>) against the empirical discounted returns:

$$
\mathcal{L}^V(\boldsymbol{\phi}) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left( V_{\boldsymbol{\phi}}(s_t) - V_t^{\text{targ}} \right)^2 \right]
$$

where $V_t^{\text{targ}} = \hat{A}_t + V_{\boldsymbol{\phi}_{\text{old}}}(s_t)$.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="2015">2015</time> &mdash; <strong>John Schulman et al. (TRPO: Trust Region Policy Optimization)</strong></dt>
  <dd>Introduced theoretical trust regions using the Fisher Information Matrix to bound the KL divergence between policy updates: $\mathbb{E}[D_{\text{KL}}(\pi_{\text{old}} \parallel \pi)] \le \delta$. While mathematically rigorous, TRPO required calculating the second-order Hessian inverse via conjugate gradient descent, making it computationally impossible for billion-parameter transformers.</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Schulman, Wolski, Dhariwal, Radford, & Klimov (PPO)</strong></dt>
  <dd>Replaced expensive second-order Hessian inversions with the first-order <em>clipped surrogate objective</em>. PPO achieved comparable or superior stability to TRPO using standard first-order Adam gradient ascent.</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>OpenAI InstructGPT (Ouyang et al.)</strong></dt>
  <dd>Applied PPO to large language models at scale (175B parameters), demonstrating that PPO RLHF aligned GPT-3 with human intent far more effectively than 100x larger unaligned models.</dd>

  <dt><time datetime="2023–2024">2023–2024</time> &mdash; <strong>The 4-Model Memory Crisis</strong></dt>
  <dd>Hosting four full 70B models in GPU memory (Actor, Critic, Ref, RM) requires over 560 GB of VRAM just to store weights in FP16, plus optimizer states for Actor and Critic. This immense complexity motivated the search for single-model alignment (DPO, Chapter 19) and critic-free reinforcement learning (GRPO, Chapter 31).</dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us step through a single PPO token update by hand with exact numbers.

### The Toy Setup
- Single token action $a_t$ generated at state $s_t$.
- Old policy probability: $\pi_{\text{old}}(a_t \mid s_t) = 0.40$.
- Reference model probability: $\pi_{\text{ref}}(a_t \mid s_t) = 0.50$.
- Terminal reward from Reward Model: $R_{\boldsymbol{\psi}} = 1.0$.
- KL penalty coefficient: $\beta = 0.1$.
- PPO clipping threshold: $\epsilon = 0.2$ (allowed ratio interval: $[0.80, 1.20]$).
- Critic baseline predictions:
  - $V(s_t) = 0.60$
  - Terminal state value: $V(s_{t+1}) = 0.0$ (episode ends).
  - Discount factor: $\gamma = 1.0$.

---

### Phase 1: Reward & Advantage Calculation

#### 1. KL Divergence Penalty
Compute the per-token KL log-ratio:
$$
\log \left( \frac{\pi_{\text{old}}}{\pi_{\text{ref}}} \right) = \log\left(\frac{0.40}{0.50}\right) = \log(0.80) \approx -0.2231
$$

The net reward incorporating the KL term is:
$$
r_t = R_{\boldsymbol{\psi}} - \beta \log\left(\frac{\pi_{\text{old}}}{\pi_{\text{ref}}}\right) = 1.0 - 0.1 \times (-0.2231) = 1.0 + 0.0223 = \mathbf{1.0223}
$$
*(Because the old policy was slightly more conservative than the reference model, it incurred zero penalty and gained a tiny bonus!)*

#### 2. Temporal Difference Residual & Advantage
Because this is the final token, $V(s_{t+1}) = 0$:
$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t) = 1.0223 + 0.0 - 0.60 = +\mathbf{0.4223}
$$

For a single-step episode, the GAE advantage is simply:
$$
\hat{A}_t = \delta_t^V = +\mathbf{0.4223}
$$
Because $\hat{A}_t > 0$, this token was significantly better than the critic's baseline ($0.60$).

---

### Phase 2: Evaluating the PPO Clipped Objective

Now suppose during an optimization epoch, our updated actor parameters adjust the probability of this token under two different scenarios:

#### Scenario A: Policy makes a moderate increase ($\pi_{\boldsymbol{\theta}} = 0.46$)
1. Importance ratio:
   $$
   r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{old}}} = \frac{0.46}{0.40} = \mathbf{1.15}
   $$
2. Is $r_t$ inside the clipping window $[1-\epsilon, 1+\epsilon] = [0.80, 1.20]$?
   - Yes! $1.15 \le 1.20$.
3. Compute unclipped and clipped terms:
   - Term 1: $r_t \hat{A}_t = 1.15 \times 0.4223 \approx \mathbf{0.4856}$
   - Term 2: $\operatorname{clip}(r_t, 0.8, 1.2) \hat{A}_t = 1.15 \times 0.4223 \approx \mathbf{0.4856}$
4. Objective:
   $$
   \mathcal{L}^{\text{CLIP}} = \min(0.4856, 0.4856) = \mathbf{0.4856}
   $$
   The gradient $\frac{\partial \mathcal{L}}{\partial \pi} = \frac{\hat{A}_t}{\pi_{\text{old}}} = \frac{0.4223}{0.40} = +1.056 > 0$. The model continues to receive positive feedback.

---

#### Scenario B: Policy makes an aggressive increase ($\pi_{\boldsymbol{\theta}} = 0.52$)
1. Importance ratio:
   $$
   r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{old}}} = \frac{0.52}{0.40} = \mathbf{1.30}
   $$
2. Is $r_t$ inside $[0.80, 1.20]$?
   - No! $1.30 > 1.20$. The ratio has breached the upper safety boundary!
3. Compute terms:
   - Term 1: $r_t \hat{A}_t = 1.30 \times 0.4223 \approx \mathbf{0.5490}$
   - Term 2: $\operatorname{clip}(1.30, 0.8, 1.2) \hat{A}_t = 1.20 \times 0.4223 \approx \mathbf{0.5068}$
4. Objective:
   $$
   \mathcal{L}^{\text{CLIP}} = \min(0.5490, 0.5068) = \mathbf{0.5068}
   $$
   <mark>The objective is strictly capped at $0.5068$!</mark>
   Because the clipped value is constant with respect to $\pi_{\boldsymbol{\theta}}$ beyond $1.20$, the gradient becomes **$0.0$**! The optimizer refuses to push the parameter any further, saving the model from policy collapse.

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>PPO stabilizes reinforcement learning in Large Language Models through a dual-safeguard design: the <strong>Critic network</strong> provides fine-grained, low-variance token advantages via Generalized Advantage Estimation (<abbr title="Generalized Advantage Estimation">GAE</abbr>), while the <strong>Clipped Surrogate Objective</strong> erects an unyielding mathematical wall that freezes parameter gradients the moment the updated policy drifts more than $\epsilon$ from its rehearsal distribution.</p>
</fieldset>
