# Hands-on Lab 07: The GRPO Reasoning Engine (Autonomous Reasoning in 200 Lines of Pure Python)

<fieldset id="evolution">
<legend><strong>The Python Brain Evolution Chain &bull; Reinforcement &amp; Reasoning Frontier (Stage 7 of 7)</strong></legend>
<p>In Lab 06, we built the Speculative Decoding &amp; INT4 Quantization Engine, conquering the memory bandwidth bottleneck of LLM inference. However, up to this point, all of our models were trained via pure imitation: static cross-entropy predicting human text.</p>
<p>In this crowning capstone lab of the Reinforcement Learning curriculum, we build <strong>The GRPO Reasoning Engine</strong> in ~200 lines of pure, standard-library Python. Zero external dependencies: no PyTorch, no HuggingFace, no NumPy. We implement the complete mathematical mechanics of frontier reasoning models (DeepSeek-Math, DeepSeek-R1): <strong>Critic-Free Group Advantage Normalization</strong>, <strong>Rule-Based Verifiable Rewards (RLVR)</strong>, <strong>PPO Ratio Clipping</strong>, and <strong>Schulman's Non-Negative KL Divergence Leash</strong>.</p>
<pre>
[The Python Brain Evolution Roadmap &bull; Complete 7-Stage Journey]
[Stage 1]  80 Lines Pure Python: Bengio 2003 MLP Language Model (Embeddings, Dense Layers, Manual Backprop)
       │
       ▼ (Amnesia flaw: 1-word context window)
[Stage 2] 140 Lines Pure Python: Attention Brain (Unlocking Q, K, V Projections &amp; Causal Attention)
       │
       ▼ (Instability flaw: vanishing gradients in deep networks)
[Stage 3] 220 Lines Pure Python: Modern Transformer Block (Pre-RMSNorm, Residual Highways, &amp; SwiGLU Gating)
       │
       ▼ (Sampling flaw: rigid greedy loops and O(T^2) redundant computation)
[Stage 4] 300 Lines Pure Python: Interactive LLM Engine (KV Cache &amp; Nucleus Sampling Suite)
       │
       ▼ (Multi-Tenant flaw: Static batching wastes 60%+ compute in bubbles; contiguous arrays fragment memory)
[Stage 5] 200 Lines Pure Python: Streaming KV &amp; Continuous Batching Engine (PagedAttention &amp; Iteration Scheduling)
       │
       ▼ (Bandwidth flaw: Serial single-token memory bound bottleneck O(T))
[Stage 6] 220 Lines Pure Python: Speculative Decoding &amp; INT4 Quantization Engine
       │
       ▼ (Imitation flaw: Supervised learning cannot discover new reasoning paths or self-correct errors)
[Stage 7 (Reasoning Frontier)] 200 Lines Pure Python: The GRPO Reasoning Engine
       │
       ▼ (Result: A self-contained RL engine driving autonomous mathematical reasoning and verification!)
</pre>
</fieldset>

---

## Step 1: 3-Year-Old Intuition (The Self-Correcting Study Group)

Imagine a study group of four friends working on a hard math riddle:

<figure>
<pre>
[Traditional RLHF with Critic: Heavy, Fragile &amp; Expensive]
Student writes one word ──► Critic guesses grade ──► Student writes next word
(Critic occupies 50% of GPU memory and frequently misjudges long math proofs!)

[GRPO: The Fast, Autonomous Study Group]
Step 1: All 4 students write their full solutions to the riddle simultaneously.
Step 2: Rule Verifier checks the final blackboard:
        Student 1: &lt;think&gt;3*4=12, 12+2=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;  ──► Reward: 1.5
        Student 2: &lt;think&gt;2+12=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;          ──► Reward: 1.5
        Student 3: &lt;think&gt;2+3=5, 5*4=20&lt;/think&gt;&lt;answer&gt;20&lt;/answer&gt;    ──► Reward: 0.5
        Student 4: 14 &lt;eos&gt;                                          ──► Reward: 1.0
Step 3: Group Mean = 1.125. Normalize scores against group standard deviation.
Step 4: Update weights: Students 1 &amp; 2 get positive advantage; Student 3 gets penalized!
        Result: 100% self-contained policy updates with ZERO Critic network!
</pre>
<figcaption><strong>Figure 31b.1:</strong> The GRPO study group cycle: Multi-candidate generation, rule-based verification, and group-normalized advantage updates without a Critic model.</figcaption>
</figure>

1. **No Private Tutor Required**:
   Instead of keeping a giant neural network just to guess interim values, the group's collective average acts as a perfectly fair, dynamic baseline.
2. **Deterministic Objective Rules**:
   A compiler or arithmetic checker never hallucinates. Either the tags are closed correctly and the arithmetic matches, or they do not.
3. **The Elastic Safety Tether (KL Leash)**:
   A mathematical leash keeps the model tethered to its base reference distribution, ensuring it doesn't forget general language capabilities while mastering the riddle.

---

## Step 2: The Bridging Question

How do we implement the complete mathematical pipeline of Group Relative Policy Optimization &mdash; categorical sampling, rule verification, group advantage standardization ($A_i = \frac{R_i - \mu}{\sigma}$), token-level importance ratios ($r_t = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{old}}}$), PPO clipping, and Schulman's non-negative KL penalty &mdash; in clean, zero-dependency Python?

---

## Step 3: The Exact Math & Formula

### 1. Group Advantage Normalization
For a group of $G$ sampled completions with rewards $R_1, \dots, R_G$:

$$
\mu = \frac{1}{G} \sum_{i=1}^G R_i, \quad \sigma = \sqrt{\frac{1}{G} \sum_{i=1}^G (R_i - \mu)^2 + \epsilon}
$$
$$
A_i = \frac{R_i - \mu}{\sigma}
$$

### 2. Token Importance Ratio & PPO Clipping
For each token $t$ in completion $i$:

$$
\rho_{i,t} = \frac{\pi_{\boldsymbol{\theta}}(a_{i,t} \mid s_{i,t})}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_{i,t} \mid s_{i,t})}
$$
$$
\mathcal{L}^{\text{CLIP}}_{i,t} = \min\left( \rho_{i,t} A_i, \; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right)
$$

### 3. Schulman Non-Negative KL Divergence Penalty
To anchor policy $\pi_{\boldsymbol{\theta}}$ to the reference model $\pi_{\text{ref}}$:

$$
u = \frac{\pi_{\text{ref}}(a_{i,t} \mid s_{i,t})}{\pi_{\boldsymbol{\theta}}(a_{i,t} \mid s_{i,t})}, \quad D_{\text{KL}} = u - \log(u) - 1 \ge 0
$$

The net policy gradient pushes parameter updates proportionally to $\frac{A_i}{|o_i|} - \beta D_{\text{KL}}$.

---

## Step 4: Where Did It Come From?

In early 2024, DeepSeek AI developed GRPO for **DeepSeek-Math**, proving that group-relative baselines eliminated the severe memory overhead of the Critic network. In January 2025, **DeepSeek-R1** showed that scaling GRPO with simple rule-based rewards (format + accuracy) unlocked emergent reasoning capabilities, competitive with closed-source frontier models.

---

## Step 5: The Complete Pure Python Engine

Below is the complete, self-contained Python script ([`labs/07_grpo_engine.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/07_grpo_engine.py)). You can execute it directly with `python3 labs/07_grpo_engine.py`:

```python
"""
Hands-on Lab 07: The GRPO Reasoning Engine in Pure Python
Stage 7 of the Python Brain Evolution Chain (RL & Reasoning Frontier)
"""

import math
import random

# 1. Vocabulary & Tokenizer
VOCAB = [
    "<pad>", "<eos>",
    "<think>", "</think>", "<answer>", "</answer>",
    "2", "3", "4", "5", "7", "8", "12", "14", "20",
    "+", "*", "=", "correct", "wrong"
]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: tok for idx, tok in enumerate(VOCAB)}
EOS_ID = TOKEN_TO_ID["<eos>"]
PAD_ID = TOKEN_TO_ID["<pad>"]

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def sample_token(probs):
    r = random.random()
    cum = 0.0
    for idx, p in enumerate(probs):
        cum += p
        if r <= cum:
            return idx
    return len(probs) - 1

# 2. Toy Policy Model
class ToyPolicy:
    def __init__(self, vocab_size, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.weights = [[(random.random() - 0.5) * 0.1 for _ in range(vocab_size)] for _ in range(vocab_size)]

    def forward(self, prev_token_id):
        return self.weights[prev_token_id]

    def get_probs(self, prev_token_id):
        return softmax(self.forward(prev_token_id))

    def clone(self):
        new_model = ToyPolicy(self.vocab_size)
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                new_model.weights[i][j] = self.weights[i][j]
        return new_model

# 3. Rule-Based Verifier (RLVR)
def verify_response(tokens, expected_answer="14"):
    text_tokens = [ID_TO_TOKEN[t] for t in tokens if t not in (PAD_ID, EOS_ID)]
    format_reward = 0.0
    has_ts = "<think>" in text_tokens
    has_te = "</think>" in text_tokens
    has_as = "<answer>" in text_tokens
    has_ae = "</answer>" in text_tokens

    if has_ts and has_te and has_as and has_ae:
        if text_tokens.index("<think>") < text_tokens.index("</think>") < text_tokens.index("<answer>") < text_tokens.index("</answer>"):
            format_reward = 0.5

    acc_reward = 0.0
    if expected_answer in text_tokens:
        if has_as and has_ae:
            ans_slice = text_tokens[text_tokens.index("<answer>"):text_tokens.index("</answer>")]
            if expected_answer in ans_slice:
                acc_reward = 1.0
        else:
            acc_reward = 0.5

    return format_reward + acc_reward, format_reward, acc_reward

# 4. Group Advantage Normalization
def compute_group_advantages(rewards):
    G = len(rewards)
    mean_r = sum(rewards) / G
    variance = sum((r - mean_r) ** 2 for r in rewards) / G
    std_r = math.sqrt(variance + 1e-4)
    advantages = [(r - mean_r) / std_r for r in rewards]
    return advantages, mean_r, std_r

# 5. GRPO Training Step with PPO Clipping & Schulman KL
def train_grpo_step(actor, ref_model, prompt_token_id, group_size=6, lr=0.05, clip_eps=0.2, beta=0.04):
    old_actor = actor.clone()
    rollouts, rewards = [], []
    
    for _ in range(group_size):
        actions = []
        tokens = [prompt_token_id]
        for _ in range(8):
            probs = old_actor.get_probs(tokens[-1])
            action = sample_token(probs)
            tokens.append(action)
            actions.append(action)
            if action == EOS_ID:
                break
        tot_r, fmt_r, acc_r = verify_response(actions, expected_answer="14")
        rollouts.append(actions)
        rewards.append(tot_r)
        
    advantages, mean_r, std_r = compute_group_advantages(rewards)
    
    for i in range(group_size):
        actions = rollouts[i]
        adv = advantages[i]
        seq_len = max(len(actions), 1)
        prev = prompt_token_id
        for action in actions:
            p_curr = actor.get_probs(prev)[action]
            p_old = old_actor.get_probs(prev)[action]
            p_ref = ref_model.get_probs(prev)[action]
            
            ratio = p_curr / (p_old + 1e-12)
            clipped_ratio = max(min(ratio, 1.0 + clip_eps), 1.0 - clip_eps)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            
            u = p_ref / (p_curr + 1e-12)
            kl = u - math.log(u + 1e-12) - 1.0
            
            grad_scale = (adv / (p_old + 1e-12) if (surr1 <= surr2 or (adv > 0 and ratio < 1 + clip_eps)) else 0.0)
            grad_scale -= beta * kl
            
            step_factor = lr * (grad_scale / seq_len)
            probs_curr = actor.get_probs(prev)
            for j in range(VOCAB_SIZE):
                grad_logit = (1.0 if j == action else 0.0) - probs_curr[j]
                actor.weights[prev][j] += step_factor * grad_logit
                
            prev = action
            
    return mean_r, std_r
```

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>By implementing GRPO in 200 lines of standard Python, you have demystified the crowning algorithmic engine behind DeepSeek-R1: frontier reasoning does not require giant Critic models or subjective human annotations. It requires only group-relative advantage normalization, unhackable rule verification, and clipped policy gradient bounds.</p>
</fieldset>
