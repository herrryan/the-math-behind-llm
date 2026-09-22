# Chapter 19: Teaching Good Manners (RLHF, Reward Modeling, and DPO)


## Step 1: 3-Year-Old Intuition (The Wild Parrot and the Golden Stickers) {: #step-1 }

!!! note "3-Year-Old Intuition: The Hyper-Intelligent Parrot and the Etiquette School"
    Imagine a wild, magical parrot that flew across the entire world, reading every single library, newspaper, chat forum, and bathroom wall on Earth:

    1. **The Wild Genius (Pre-trained Base Model)**:
       - The parrot knows everything: quantum physics, ancient Babylonian poetry, and how to bake sourdough bread.
       - But because it swallowed the whole internet raw, it has zero manners!
       - If you ask it: *"How do I make a chocolate cake?"*, it might give you a brilliant recipe, or it might reply with a toxic internet insult, or it might start reciting copyright legalese!
       - It is not trying to help you &mdash; it is just blindly continuing whatever pattern it saw on the internet.

    2. **Showing Polite Examples (Supervised Fine-Tuning &mdash; SFT)**:
       - First, we give the parrot a polite golden book written by patient schoolteachers.
       - Each page shows a polite question followed by a courteous, helpful answer:
         - <kbd>Question:</kbd> *"How do I bake a cake?"*
         - <kbd>Answer:</kbd> *"Here is a simple, delicious recipe: First, preheat your oven..."*
       - The parrot learns how a respectful assistant sounds. But what happens when someone asks a sneaky, complex, or dangerous question?

    3. **The Gold Sticker Referee (RLHF & Reward Modeling)**:
       - We ask human judges to rank pairs of parrot answers.
       - Answer A is helpful and kind $\to$ awarded a **Gold Sticker (<dfn id="def-reward">Reward, $+1$</dfn>)**!
       - Answer B is rude or dangerous $\to$ gets a gentle frown ($0$ stickers).
       - We train a robotic referee (<abbr title="Reward Model">RM</abbr>) to predict how many stickers a human judge would give.
       - Then, a training coach nudges the parrot's brain using Reinforcement Learning (<abbr title="Proximal Policy Optimization">PPO</abbr>) to maximize stickers.
       - An invisible leash (<abbr title="Kullback-Leibler">KL</abbr> penalty) keeps the parrot from drifting too far from its original language skills or cheating the referee!

    4. **The Mathematical Shortcut: Direct Preference Optimization (DPO)**:
       - Training a separate robotic referee, running four huge models in GPU memory, and stabilizing fickle reinforcement learning is notoriously unstable and expensive.
       - In 2023, Stanford researchers proved a breathtaking mathematical theorem:
         *"The parrot's own brain already contains the exact math of the referee's sticker book!"*
       - By directly comparing the probability ratio of the winning answer versus the losing answer, **DPO** skips the referee, skips reinforcement learning, and aligns the model with a single, elegant binary classification formula!

<figure>
<pre>
The Alignment Journey: Pre-training to DPO:

1. Pre-training:  Read the web  ──► Predict next token blindly
2. SFT:           Q&A Examples  ──► Learn assistant tone
3. RLHF (PPO):    Prompt ──► Policy ──► Reward Model ──► PPO Update
                                 ▲                          │
                                 └────── KL Leash ──────────┘
4. DPO (Modern):  Prompt + [ Winner / Loser ] ──► Direct Loss ──► Policy
                  (Zero Reward Model! Zero PPO! 100% Stable!)
</pre>
<figcaption><strong>Figure 19.1:</strong> Alignment progression from raw pre-training to multi-model RLHF and streamlined Direct Preference Optimization (DPO).</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: Why Can't We Just Use Cross-Entropy for Alignment?"
    In Chapter 15, we saw that Cross-Entropy Loss teaches the model to maximize the log-likelihood of target tokens:



    $$
    \mathcal{L}_{\text{SFT}}(\boldsymbol{\theta}) = -\sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(y_t \mid x, y_{\lt t})
    $$



    This works well for teaching the model basic formatting and conversational cadence (<abbr title="Supervised Fine-Tuning">SFT</abbr>). But it suffers from two fatal limitations:

    1. **Supervised Data Cannot Express "Never Do This"**: Cross-Entropy can only pull the model toward positive demonstrations. It cannot penalize bad behaviors without explicitly teaching the model how to generate bad behaviors!
    2. **Human Preference is Relative, Not Absolute**: If two human evaluators are asked to assign an absolute numerical grade to an essay (e.g. "Is this response an 8.4 or an 8.7?"), they disagree violently. But if you show them Response A and Response B side by side, they agree over 90% of the time on which one is better!

    *"How do we mathematically transform pairwise human preferences ($y_w \succ y_l$) into an optimization objective that directly boosts winning responses, suppresses losing responses, and mathematically guarantees the model doesn't drift into gibberish?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Bradley-Terry Preference Model (1952)

Given a prompt $x$ and two candidate responses: a preferred winning response $y_w$ and a dispreferred losing response $y_l$ (denoted $y_w \succ y_l \mid x$).

Under the **Bradley-Terry Luce model**, the probability that a human prefers $y_w$ over $y_l$ is governed by an underlying latent reward function $r(x, y) \in \mathbb{R}$:



$$
P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right) = \frac{1}{1 + \exp\left(-(r(x, y_w) - r(x, y_l))\right)}
$$



where $\sigma(u) = \frac{1}{1 + e^{-u}}$ is the standard sigmoid function.

---

### 2. The Classical RLHF Objective with KL Regularization

In classical RLHF (Christiano et al., 2017; Ouyang et al., 2022), we first train a separate Reward Model $r_\phi(x, y)$ via binary cross-entropy on human pairwise comparisons:



$$
\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]
$$



Once $r_\phi$ is frozen, the language model policy $\pi_{\boldsymbol{\theta}}$ is optimized via Reinforcement Learning (<abbr title="Proximal Policy Optimization">PPO</abbr>) to maximize expected reward, constrained by a Kullback-Leibler (<abbr title="Kullback-Leibler">KL</abbr>) divergence penalty relative to the initial reference policy $\pi_{\text{ref}}$:



$$
\max_{\pi_{\boldsymbol{\theta}}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\boldsymbol{\theta}}}\left[ r_\phi(x, y) \right] - \beta D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}}(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)
$$



where:
- $\beta > 0$ is the **KL regularization coefficient** (the "leash strength").
- $\pi_{\text{ref}}$ is the frozen SFT model.
- $D_{\text{KL}}(P \parallel Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)} \ge 0$ penalizes the policy if it drifts too far from natural English grammar.

---

### 3. The Closed-Form Optimal Policy

Remarkably, the constrained RL objective above has a known exact analytical solution!
By solving the calculus of variations problem subject to the constraint $\sum_y \pi(y \mid x) = 1$, the optimal policy $\pi^*$ is:



$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$



where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the **partition function** (an intractable normalization sum over all possible text responses).

---

### 4. The DPO Breakthrough: Inverting the Reward (Rafailov et al., 2023)

In standard RLHF, $Z(x)$ is impossible to compute because the space of all possible text outputs is infinite ($\sim 32000^{2048}$). This is why OpenAI had to use PPO.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn realized you can **algebraically invert** the optimal policy equation for the reward $r(x, y)$:



$$
\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{1}{Z(x)} \exp\left(\frac{1}{\beta} r(x, y)\right)
$$



Take the natural logarithm of both sides:



$$
\log \pi^*(y \mid x) - \log \pi_{\text{ref}}(y \mid x) = -\log Z(x) + \frac{1}{\beta} r(x, y)
$$



Rearrange to express the reward explicitly in terms of policy probabilities:



$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$



Now, substitute this formula into the Bradley-Terry preference difference $r(x, y_w) - r(x, y_l)$:



$$
\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{aligned}
$$



<mark>The intractable partition function $\beta \log Z(x)$ completely cancels out!</mark>

---

### 5. The Direct Preference Optimization (DPO) Loss Function

Plugging this cancellation directly into the negative log-likelihood of the Bradley-Terry model yields the **DPO Loss Function**:



$$
\mathcal{L}_{\text{DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma \left( \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$



#### The Gradient Dynamics of DPO:
Differentiating $\mathcal{L}_{\text{DPO}}$ with respect to the policy parameters $\boldsymbol{\theta}$ reveals its intuitive mechanics:



$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{DPO}} = -\beta \, \underbrace{\sigma\left(\hat{r}_{\boldsymbol{\theta}}(x, y_l) - \hat{r}_{\boldsymbol{\theta}}(x, y_w)\right)}_{\text{Surprise / Error Weight } (1 - \sigma)} \cdot \left[ \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_w \mid x)}_{\text{Increase Likelihood of } y_w} - \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_l \mid x)}_{\text{Decrease Likelihood of } y_l} \right]
$$



- If the model already strongly prefers the winner ($\hat{r}_w \gg \hat{r}_l$), $\sigma(\hat{r}_l - \hat{r}_w) \approx 0$, and the gradient vanishes (no unnecessary changes).
- If the model mistakenly prefers the loser ($\hat{r}_l > \hat{r}_w$), the weight approaches $1$, applying maximal force to boost $y_w$ and suppress $y_l$!

---

## Step 4: Where Did It Come From? (Bradley-Terry, InstructGPT, & DPO) {: #step-4 }

<dl>
  <dt><time datetime="1952">1952</time> &mdash; <strong>Ralph Bradley & Milton Terry</strong></dt>
  <dd>Formulated the mathematical Bradley-Terry model for analyzing pairwise comparisons in competitive tournaments and psychometric evaluations, proving that pairwise win probabilities can be modeled via differences in latent scores passed through a logistic sigmoid function.</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Paul Christiano et al.</strong> (<cite>"Deep Reinforcement Learning from Human Preferences"</cite>)</dt>
  <dd>Pioneered using human pairwise preferences to train reward models for deep reinforcement learning agents, demonstrating that agents could learn backflips and game policies without explicit programmatic reward functions.</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>Long Ouyang et al. / OpenAI</strong> (<cite>"Training language models to follow instructions with human feedback"</cite> &mdash; InstructGPT)</dt>
  <dd>Applied the SFT $\to$ RM $\to$ PPO pipeline to GPT-3, creating InstructGPT and ChatGPT. They proved that a 1.3-billion parameter aligned model was consistently preferred by humans over an unaligned 175-billion parameter base model.</dd>

  <dt><time datetime="2023">2023</time> &mdash; <strong>Rafael Rafailov et al. / Stanford University</strong> (<cite>"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"</cite>)</dt>
  <dd>Derived the closed-form reward equivalence, proved that the partition function vanishes under pairwise ratios, and eliminated the need for complex PPO reinforcement learning in LLM alignment.</dd>
</dl>

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

Let us compute the complete DPO loss and gradient scaling factor for a concrete pairwise alignment step with hand arithmetic.

### 1. Toy Setup
- Prompt: $x =$ <kbd>"Write a summary of quantum mechanics."</kbd>
- Winning response: $y_w$ (concise, clear, accurate)
- Losing response: $y_l$ (confusing, rude, rambles)
- Regularization coefficient: $\beta = 0.50$

### 2. Initial Model Probabilities
Suppose the probabilities assigned to these sequences by the frozen reference model $\pi_{\text{ref}}$ and the active policy $\pi_{\boldsymbol{\theta}}$ are:
- Reference policy:
  - $\pi_{\text{ref}}(y_w \mid x) = 0.20$
  - $\pi_{\text{ref}}(y_l \mid x) = 0.10$
- Current active policy:
  - $\pi_{\boldsymbol{\theta}}(y_w \mid x) = 0.30$
  - $\pi_{\boldsymbol{\theta}}(y_l \mid x) = 0.40$
  *(Notice that the active policy currently makes the mistake of favoring the bad response $y_l$ with probability $0.40$ over the good response $y_w$ at $0.30$!)*

---

### 3. Step-by-Step DPO Loss Calculation

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step A:</strong> Calculate probability ratios $\pi_{\boldsymbol{\theta}} / \pi_{\text{ref}}$ for both responses.</p>
<p><input type="checkbox" checked disabled> <strong>Step B:</strong> Compute log ratios and scale by $\beta$ to get implicit rewards.</p>
<p><input type="checkbox" checked disabled> <strong>Step C:</strong> Compute the reward difference $\Delta r = \hat{r}(y_w) - \hat{r}(y_l)$.</p>
<p><input type="checkbox" checked disabled> <strong>Step D:</strong> Evaluate the logistic sigmoid $\sigma(\Delta r)$.</p>
<p><input type="checkbox" checked disabled> <strong>Step E:</strong> Compute the final scalar loss $\mathcal{L}_{\text{DPO}} = -\log \sigma(\Delta r)$.</p>
<p><input type="checkbox" checked disabled> <strong>Step F:</strong> Compute the gradient error weight $\beta(1 - \sigma)$.</p>
</fieldset>

#### Step A: Probability Ratios
- For winning response $y_w$:


  $$
  \frac{\pi_{\boldsymbol{\theta}}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} = \frac{0.30}{0.20} = \mathbf{1.5000}
  $$


- For losing response $y_l$:


  $$
  \frac{\pi_{\boldsymbol{\theta}}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} = \frac{0.40}{0.10} = \mathbf{4.0000}
  $$



#### Step B: Natural Logarithms & Implicit Rewards
- Log ratio for $y_w$:


  $$
  \log(1.5000) \approx \mathbf{+0.4055}
  $$


- Implicit reward for $y_w$:


  $$
  \hat{r}(x, y_w) = \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_w)}{\pi_{\text{ref}}(y_w)} = 0.50 \times 0.4055 = \mathbf{+0.2028}
  $$


- Log ratio for $y_l$:


  $$
  \log(4.0000) \approx \mathbf{+1.3863}
  $$


- Implicit reward for $y_l$:


  $$
  \hat{r}(x, y_l) = \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_l)}{\pi_{\text{ref}}(y_l)} = 0.50 \times 1.3863 = \mathbf{+0.6931}
  $$



#### Step C: Reward Difference


$$
\Delta r = \hat{r}(x, y_w) - \hat{r}(x, y_l) = 0.2028 - 0.6931 = \mathbf{-0.4904}
$$



The reward difference is negative ($-0.4904$), confirming that the active policy currently ranks the bad response higher than the good one!

#### Step D: Evaluate Sigmoid Function


$$
\sigma(\Delta r) = \sigma(-0.4904) = \frac{1}{1 + e^{0.4904}} = \frac{1}{1 + 1.6330} = \frac{1}{2.6330} \approx \mathbf{0.3798}
$$



The model assesses only a $38.0\%$ probability that human judges will pick the winner!

#### Step E: Compute DPO Loss


$$
\mathcal{L}_{\text{DPO}} = -\log(0.3798) \approx \mathbf{0.9681}
$$



#### Step F: Compute Gradient Error Weight
The scale factor multiplying the gradient direction is:



$$
\text{Weight} = \beta \left( 1 - \sigma(\Delta r) \right) = 0.50 \times (1 - 0.3798) = 0.50 \times 0.6202 = \mathbf{0.3101}
$$



<mark>Because the model was wrong ($\sigma = 38\%$), it receives a substantial gradient push of magnitude $0.3101$, actively driving $\pi_{\boldsymbol{\theta}}(y_w)$ up and suppressing $\pi_{\boldsymbol{\theta}}(y_l)$ down!</mark>

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: The Punchline of RLHF and DPO"
    **Pre-training creates an encyclopedic brain, but alignment teaches it to be a helpful human partner.**

    By proving that a language model is secretly its own reward model, Direct Preference Optimization (DPO) replaces complex, brittle multi-agent reinforcement learning loops with a single, elegant binary classification loss &mdash; aligning billions of parameters directly with human values.
