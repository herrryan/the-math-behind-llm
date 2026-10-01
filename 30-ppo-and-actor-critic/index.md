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

1. **The Delayed Credit Bottleneck (The Typo at Token 995)**:
   - In a 1,000-token essay or coding solution, the model receives a single scalar reward $R(\tau)$ only after producing the final `<|endoftext|>` token.
   - Suppose tokens 1 through 994 were a masterclass in reasoning, but at token 995 the model missed a single closing bracket `)`. The Python interpreter crashes and outputs reward $R = 0$.
   - Raw REINFORCE treats every token in that trajectory identically: it punishes all 1,000 tokens with the same negative update!
   - We need an intermediate, step-by-step judge that can evaluate *partial progress* after every single word, rewarding good reasoning even if a subsequent step slips up.

2. **The Step-Size Cliff (Policy Collapse)**:
   - In deep neural networks, a tiny step in weight space $\Delta \boldsymbol{\theta}$ produces an exponential shift in output probabilities because of the softmax function ($\exp(z_i)$).
   - If an unconstrained gradient step increases the logit of a token slightly too much, its probability can surge from $0.01$ to $0.99$ overnight.
   - Once a language model takes a step off this cliff, it enters a catastrophic failure loop: it outputs repetitive gibberish (e.g., repeating the word *"the the the"* endlessly), receives $0$ reward, and its gradients vanish or explode. The model never recovers.

The central bridging question is:
$$\text{How do we estimate token-level advantages in real time using a learned value function, while strictly bounding policy divergence to guarantee monotonically stable training?}$$

---

## Step 3: The Exact Math & Formula

### 1. The 4-Model System in LLM RLHF

Modern classical RLHF coordinates four separate language models simultaneously:

1. **Actor Policy $\pi_{\boldsymbol{\theta}}$**: The generative language model whose weights $\boldsymbol{\theta} \in \mathbb{R}^D$ are updated to maximize human preference and problem accuracy.
2. **Critic (Value Model) $V_{\boldsymbol{\phi}}$**: A language model backbone that estimates expected future return from any intermediate point in text generation.
3. **Reference Policy $\pi_{\text{ref}}$**: A frozen copy of the original Supervised Fine-Tuned (<abbr title="Supervised Fine-Tuning">SFT</abbr>) model.
4. **Reward Model $R_{\boldsymbol{\psi}}$**: A frozen preference model trained on human comparison pairs $(y_w \succ y_l)$ via the Bradley-Terry objective (Chapter 19).

<fieldset>
<legend><strong>What Is the Critic Model Physically in Code?</strong></legend>
<p>Students often wonder: <em>Is the Critic a completely different kind of neural network?</em></p>
<p>No! The Critic is literally a clone of the Actor's Transformer architecture. The only physical difference is at the very final layer: Instead of an unembedding projection matrix $\mathbf{W}_{\text{unembed}} \in \mathbb{R}^{d \times |\mathcal{V}|}$ that outputs 100,000 logits for next-word prediction, the Critic has a single linear projection vector $\mathbf{w}_V \in \mathbb{R}^{d \times 1}$.</p>
<p>Given any prefix state $s_t = (x, y_1, \dots, y_{t-1})$, the Critic reads the final hidden state $\mathbf{h}_t \in \mathbb{R}^d$ and outputs <strong>a single scalar number</strong>:</p>
$$
V_{\boldsymbol{\phi}}(s_t) = \mathbf{w}_V^\top \mathbf{h}_t + b_V \in \mathbb{R}
$$
<p>This number answers: <em>"From this exact position in the sentence, how many total reward points do I expect this conversation to earn on average?"</em></p>
</fieldset>

---

### 2. Token-Level Reward with KL Divergence Leash

To prevent the Actor from exploiting loopholes in the Reward Model (known as <dfn id="def-reward-hacking">Reward Hacking</dfn>) and forgetting human grammar, we penalize token-by-token drift from the reference policy:

$$
r_t = \begin{cases}
-\beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} \right) & \text{for intermediate tokens } t < T \\
R_{\boldsymbol{\psi}}(x, y) - \beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_T \mid s_T)}{\pi_{\text{ref}}(a_T \mid s_T)} \right) & \text{for the terminal token } t = T
\end{cases}
$$

where $\beta > 0$ is the KL penalty coefficient (typically $\beta \approx 0.01$ to $0.1$).
- If the Actor keeps its probability close to the reference model ($\pi_{\boldsymbol{\theta}} \approx \pi_{\text{ref}}$), the ratio is $\approx 1$, and $\log(1) = 0$ (no penalty).
- If the Actor tries to game the system by assigning bizarrely high probability to an unnatural token ($\pi_{\boldsymbol{\theta}} \gg \pi_{\text{ref}}$), $\log(\pi_{\boldsymbol{\theta}} / \pi_{\text{ref}}) > 0$, subtracting points from the reward.
- This creates an **elastic safety leash**: The model is free to improve its reasoning, but is penalized if it forgets how humans write sentences.

---

### 3. The Temporal Difference (TD) Error Demystified

To evaluate whether a specific token $a_t$ was a stroke of genius or a blunder, we compute the **Temporal Difference (<abbr title="Temporal Difference">TD</abbr>) residual** $\delta_t^V$:

$$
\delta_t^V = \underbrace{r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1})}_{\text{New Reality + Updated Future Outlook}} - \underbrace{V_{\boldsymbol{\phi}}(s_t)}_{\text{Old Prior Expectation}}
$$

where $\gamma \in (0, 1]$ is the discount factor (typically $\gamma = 1.0$ for language generation).

<fieldset>
<legend><strong>A Concrete Numerical Story of the TD Error</strong></legend>
<p>Suppose the model is prompted with: <code>"Solve 2x + 6 = 14"</code>.</p>
<ol>
  <li><strong>State $s_0$ (Start)</strong>: The Critic inspects the prompt and predicts $V(s_0) = 0.50$ (it anticipates a 50% baseline chance of success).</li>
  <li><strong>Step 1 (Good Move)</strong>: The model chooses token <kbd>"Subtract 6: 2x = 8"</kbd>.
    <ul>
      <li>Immediate token reward $r_1 = 0$.</li>
      <li>New state $s_1$ is formed. The Critic inspects the progress and updates its forecast: $V(s_1) = 0.95$ (95% chance of success now!).</li>
      <li>Compute the TD surprise:
        $$
        \delta_1 = r_1 + V(s_1) - V(s_0) = 0 + 0.95 - 0.50 = \mathbf{+0.45}
        $$
      </li>
      <li><strong>Result</strong>: This single token caused future success probability to surge by $+0.45$. It gets strongly rewarded!</li>
    </ul>
  </li>
  <li><strong>Step 2 (Arithmetic Blunder)</strong>: The model chooses token <kbd>"Divide by 2: x = 5"</kbd>.
    <ul>
      <li>Immediate token reward $r_2 = 0$.</li>
      <li>New state $s_2$ is formed. The Critic immediately spots the math error and slashes its forecast: $V(s_2) = 0.05$ (only 5% chance now!).</li>
      <li>Compute the TD surprise:
        $$
        \delta_2 = r_2 + V(s_2) - V(s_1) = 0 + 0.05 - 0.95 = \mathbf{-0.90}
        $$
      </li>
      <li><strong>Result</strong>: This single token gets punished with a massive $-0.90$ penalty!</li>
    </ul>
  </li>
</ol>
<p><strong>Look at the power of TD error</strong>: Even though the final answer is wrong ($R=0$), the model does not punish Step 1! The TD error accurately rewards Step 1 ($+0.45$) and isolates the exact token where the blunder occurred (Step 2, $-0.90$). The delayed credit assignment problem is solved!</p>
</fieldset>

---

### 4. Generalized Advantage Estimation (GAE)

While the 1-step TD residual $\delta_t^V$ solves credit assignment, it relies heavily on the Critic's prediction $V(s_{t+1})$. If the Critic's neural network has prediction errors, the advantage will be **biased**.
Conversely, waiting until the end of the trajectory (Monte Carlo) is completely unbiased, but suffers from **high variance**.

To achieve the mathematically optimal trade-off, <dfn id="def-gae">Generalized Advantage Estimation (GAE-$\lambda$)</dfn> takes an exponentially-weighted sum of future TD residuals:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + (\gamma \lambda) \delta_{t+1}^V + (\gamma \lambda)^2 \delta_{t+2}^V + \dots
$$

In simple recursive form:

$$
\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}
$$

The hyperparameter $\lambda \in [0, 1]$ acts as a continuous tuning dial:
- **$\lambda = 0$ (Pure 1-Step TD)**: $\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$. Minimal variance, but high bias if the Critic makes mistakes.
- **$\lambda = 1$ (Pure Monte Carlo)**: $\hat{A}_t = \sum_{k=t}^T \gamma^{k-t} r_k - V(s_t)$. Zero bias from Critic estimates, but higher variance across long generations.
- **$\lambda = 0.95$ (The Goldilocks Standard in LLMs)**: Today's TD surprise counts $100\%$, tomorrow's counts $95\%$, the next day's counts $(0.95)^2 \approx 90\%$, smoothly blending immediate local credit with long-term trajectory outcomes.

---

### 5. The PPO Clipped Surrogate Objective (Boundary Fences on the Number Line)

Now that we have computed advantage $\hat{A}_t$, how do we update the Actor model without causing catastrophic policy collapse?

Let the probability ratio between the current candidate policy $\pi_{\boldsymbol{\theta}}$ and the old policy $\pi_{\boldsymbol{\theta}_{\text{old}}}$ that generated the training text be:

$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \quad \text{with } r_t(\boldsymbol{\theta}_{\text{old}}) = 1.0
$$

<fieldset>
<legend><strong>Visualizing the Ratio on the Chalk Number Line</strong></legend>
<p>Look at this ratio $r_t(\boldsymbol{\theta})$ on our sidewalk chalk line:</p>
<pre>
       Left Fence                                  Where You Stand                            Right Fence
       (1 - ε = 0.8)                                   (r = 1.0)                             (1 + ε = 1.2)
<───────────┼──────────────────────────────────────────────┼──────────────────────────────────────┼───────────>
           0.8                                            1.0                                    1.2
</pre>
<ul>
  <li>At the start of the day, your knobs are unchanged: $r = 1.0$ (you stand right in the middle).</li>
  <li>If you nudge a knob to make an action more likely, $r$ walks to the right ($r > 1.0$). If it reaches $1.10$, you gave that word a $10\%$ boost.</li>
  <li>If you make an action less likely, $r$ walks to the left ($r < 1.0$).</li>
</ul>
<p>In deep neural networks, an unconstrained gradient step can yank a knob so violently that $r$ leaps from $1.0$ all the way to $50.0$! Once a model takes that giant leap, its output degenerates into repetitive gibberish and it collapses. To stop this, we plant two sturdy <strong>boundary fences</strong> in the chalk at $1 - \epsilon = 0.8$ and $1 + \epsilon = 1.2$.</p>
</fieldset>

The PPO Clipped Surrogate Objective is:

$$
\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta})\hat{A}_t, \; \operatorname{clip}\left(r_t(\boldsymbol{\theta}), \, 1-\epsilon, \, 1+\epsilon\right)\hat{A}_t \right) \right]
$$

where $\epsilon$ is the clipping radius (typically $\epsilon = 0.2$, which confines the allowed ratio shift to $[0.8, 1.2]$).

<fieldset>
<legend><strong>Deconstructing the 4 Quadrants: Why Is There a Minimum ($\min$)?</strong></legend>
<p>Students often ask: <em>Why does the formula take the minimum between the unclipped and clipped terms?</em></p>
<p>The $\min$ operation creates a <strong>pessimistic lower bound</strong>. It ensures the optimizer never gets overly optimistic about a policy update:</p>
<ol>
  <li><strong>Case 1: Positive Advantage ($\hat{A}_t > 0$, Good Action)</strong>
    <ul>
      <li>We want to increase the probability of this action ($r_t > 1$).</li>
      <li>If $r_t$ grows past $1 + \epsilon$ (e.g. $r_t = 1.4$ while $1 + \epsilon = 1.2$):</li>
      <li>The unclipped term is $1.4 \hat{A}_t$, while the clipped term is $1.2 \hat{A}_t$.</li>
      <li>$\min(1.4 \hat{A}_t, 1.2 \hat{A}_t) = 1.2 \hat{A}_t$. The objective is capped! The gradient becomes <strong>zero</strong>, preventing the model from greedily over-committing to this token.</li>
    </ul>
  </li>
  <li><strong>Case 2: Negative Advantage ($\hat{A}_t < 0$, Bad Action)</strong>
    <ul>
      <li>We want to decrease the probability of this action ($r_t < 1$).</li>
      <li><em>Watch the sign flip!</em> Because $\hat{A}_t$ is negative, multiplying by a smaller number yields a <strong>larger</strong> value (e.g., $0.1 \times (-10) = -1.0$, whereas $0.8 \times (-10) = -8.0$).</li>
      <li>If the model aggressively suppresses the token so that $r_t$ drops below $1 - \epsilon$ (e.g. $r_t = 0.1$ while $1 - \epsilon = 0.8$):</li>
      <li>The unclipped term is $-1.0$, while the clipped term is $-8.0$.</li>
      <li>$\min(-1.0, -8.0) = -8.0$. The minimum selects the <strong>clipped, more pessimistic value</strong>! Because $-8.0$ is flat with respect to $\boldsymbol{\theta}$, its gradient is <strong>zero</strong>!</li>
      <li><strong>Result</strong>: Once a bad token has been suppressed sufficiently ($r_t < 0.8$), PPO stops kicking a dead horse! It leaves the token alone, preventing numerical underflow and preserving future exploration.</li>
    </ul>
  </li>
</ol>
</fieldset>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 30.1:</strong> Complete PPO clipping behavior across advantage and ratio regimes.</caption>
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
      <td><strong>Zero gradient</strong>: stops penalizing already-suppressed token.</td>
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
