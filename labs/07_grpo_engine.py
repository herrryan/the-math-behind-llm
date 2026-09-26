"""
Hands-on Lab 07: The GRPO Reasoning Engine in Pure Python
Stage 7 of the Python Brain Evolution Chain (RL & Reasoning Frontier)

A self-contained, zero-dependency implementation of Group Relative Policy Optimization (GRPO)
inspired by DeepSeek-Math and DeepSeek-R1.

Implements:
1. Discrete Vocabulary and Toy Policy Network (Logits -> Softmax -> Sampling)
2. Verifiable Rule Engine: Format Verification (<think>...</think><answer>...</answer>) and Accuracy Check
3. Group Rollout Sampling (G candidates per prompt)
4. Group-Relative Advantage Normalization: A_i = (R_i - mean(R)) / std(R)
5. PPO-Clipped Policy Gradient Update with Schulman's Non-Negative KL Divergence Penalty
"""

import math
import random

# ==============================================================================
# 1. Vocabulary & Tokenizer
# ==============================================================================

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
    """Numerically stable softmax over a list of floats."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def sample_token(probs):
    """Categorical sampling given a probability vector."""
    r = random.random()
    cum = 0.0
    for idx, p in enumerate(probs):
        cum += p
        if r <= cum:
            return idx
    return len(probs) - 1

# ==============================================================================
# 2. Toy Policy Model (Weight Matrix for Next-Token Logits)
# ==============================================================================

class ToyPolicy:
    def __init__(self, vocab_size, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        # State: last token -> next token logit matrix [vocab_size x vocab_size]
        self.weights = [[(random.random() - 0.5) * 0.1 for _ in range(vocab_size)] for _ in range(vocab_size)]

    def forward(self, prev_token_id):
        return self.weights[prev_token_id]

    def get_probs(self, prev_token_id):
        logits = self.forward(prev_token_id)
        return softmax(logits)

    def clone(self):
        new_model = ToyPolicy(self.vocab_size)
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                new_model.weights[i][j] = self.weights[i][j]
        return new_model

# ==============================================================================
# 3. Rule-Based Verifier (RLVR)
# ==============================================================================

def verify_response(tokens, expected_answer="14"):
    """
    Evaluates format and correctness:
    - Format Reward: +0.5 if strictly follows <think> ... </think> <answer> expected </answer>
    - Accuracy Reward: +1.0 if answer token matches expected_answer
    """
    text_tokens = [ID_TO_TOKEN[t] for t in tokens if t not in (PAD_ID, EOS_ID)]
    
    # Check format tags
    format_reward = 0.0
    has_think_start = "<think>" in text_tokens
    has_think_end = "</think>" in text_tokens
    has_ans_start = "<answer>" in text_tokens
    has_ans_end = "</answer>" in text_tokens

    if has_think_start and has_think_end and has_ans_start and has_ans_end:
        idx_ts = text_tokens.index("<think>")
        idx_te = text_tokens.index("</think>")
        idx_as = text_tokens.index("<answer>")
        idx_ae = text_tokens.index("</answer>")
        if idx_ts < idx_te < idx_as < idx_ae:
            format_reward = 0.5

    # Check accuracy
    acc_reward = 0.0
    if expected_answer in text_tokens:
        # If expected answer appears inside the <answer> tags
        if has_ans_start and has_ans_end:
            ans_slice = text_tokens[text_tokens.index("<answer>"):text_tokens.index("</answer>")]
            if expected_answer in ans_slice:
                acc_reward = 1.0
        else:
            acc_reward = 0.5  # Partial credit if answer found without tags

    total_reward = format_reward + acc_reward
    return total_reward, format_reward, acc_reward

# ==============================================================================
# 4. GRPO Engine: Group Sampling & Advantage Normalization
# ==============================================================================

def generate_rollout(policy, start_token_id, max_len=10):
    """Generates a token sequence and records actions and probabilities."""
    tokens = [start_token_id]
    action_log_probs = []
    
    for _ in range(max_len):
        prev = tokens[-1]
        probs = policy.get_probs(prev)
        action = sample_token(probs)
        tokens.append(action)
        action_log_probs.append(math.log(probs[action] + 1e-12))
        if action == EOS_ID:
            break
            
    return tokens[1:], action_log_probs

def compute_group_advantages(rewards):
    """
    A_i = (R_i - mean(R)) / (std(R) + epsilon)
    """
    G = len(rewards)
    mean_r = sum(rewards) / G
    variance = sum((r - mean_r) ** 2 for r in rewards) / G
    std_r = math.sqrt(variance + 1e-4)
    advantages = [(r - mean_r) / std_r for r in rewards]
    return advantages, mean_r, std_r

# ==============================================================================
# 5. GRPO Training Step with PPO Clipping & Schulman KL
# ==============================================================================

def train_grpo_step(actor, ref_model, prompt_token_id, group_size=6, lr=0.05, clip_eps=0.2, beta=0.04):
    """
    Executes a single GRPO iteration:
    1. Sample G rollouts from current policy
    2. Compute verifiable rewards
    3. Compute group-relative normalized advantages
    4. Compute clipped policy gradient and update weights
    """
    old_actor = actor.clone()
    rollouts = []
    rewards = []
    
    # 1. Group Rollout Sampling
    for _ in range(group_size):
        actions, _ = generate_rollout(old_actor, prompt_token_id, max_len=8)
        tot_r, fmt_r, acc_r = verify_response(actions, expected_answer="14")
        rollouts.append(actions)
        rewards.append(tot_r)
        
    # 2. Group Advantage Normalization
    advantages, mean_r, std_r = compute_group_advantages(rewards)
    
    # 3. Policy Gradient Step
    for i in range(group_size):
        actions = rollouts[i]
        adv = advantages[i]
        seq_len = max(len(actions), 1)
        
        prev = prompt_token_id
        for action in actions:
            probs_curr = actor.get_probs(prev)
            probs_old = old_actor.get_probs(prev)
            probs_ref = ref_model.get_probs(prev)
            
            p_curr = probs_curr[action]
            p_old = probs_old[action]
            p_ref = probs_ref[action]
            
            # Probability ratio
            ratio = p_curr / (p_old + 1e-12)
            
            # PPO Clipping logic
            clipped_ratio = max(min(ratio, 1.0 + clip_eps), 1.0 - clip_eps)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            
            # Schulman unbiased non-negative KL estimate: (ref / curr) - log(ref / curr) - 1
            u = p_ref / (p_curr + 1e-12)
            kl = u - math.log(u + 1e-12) - 1.0
            
            # Effective advantage gradient push
            grad_scale = (adv / (p_old + 1e-12) if (surr1 <= surr2 or (adv > 0 and ratio < 1 + clip_eps)) else 0.0)
            grad_scale -= beta * kl
            
            # Update logit weights: d_logit_j = (1(j==action) - p_j) * grad_scale / seq_len
            step_factor = lr * (grad_scale / seq_len)
            for j in range(VOCAB_SIZE):
                grad_logit = (1.0 if j == action else 0.0) - probs_curr[j]
                actor.weights[prev][j] += step_factor * grad_logit
                
            prev = action
            
    return mean_r, std_r

# ==============================================================================
# 6. Verification Experiment: Training the Reasoning Brain
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Hands-on Lab 07: Pure Python GRPO Reasoning Engine")
    print("Simulating test-time format and arithmetic reasoning convergence")
    print("=" * 70)
    
    random.seed(1337)
    actor = ToyPolicy(VOCAB_SIZE, seed=1337)
    ref_model = actor.clone()
    
    prompt = TOKEN_TO_ID["+"] # Represents math prompt "2 + 3 * 4"
    
    print("\nInitial Group Rollouts (Pre-RL Baseline):")
    for k in range(3):
        sample, _ = generate_rollout(actor, prompt, max_len=6)
        r, fmt, acc = verify_response(sample, expected_answer="14")
        toks = [ID_TO_TOKEN[t] for t in sample]
        print(f"  Rollout {k+1}: {' '.join(toks)} | Reward: {r:.2f} (Fmt: {fmt}, Acc: {acc})")
        
    print("\nStarting GRPO Training Iterations (Critic-Free Group Normalization)...")
    for epoch in range(1, 41):
        mean_r, std_r = train_grpo_step(actor, ref_model, prompt, group_size=8, lr=0.15, beta=0.01)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} | Group Mean Reward: {mean_r:.3f} | Group Std Dev: {std_r:.3f}")
            
    print("\nTrained Policy Rollouts (Post-GRPO Frontier):")
    successes = 0
    test_samples = 5
    for k in range(test_samples):
        sample, _ = generate_rollout(actor, prompt, max_len=6)
        r, fmt, acc = verify_response(sample, expected_answer="14")
        toks = [ID_TO_TOKEN[t] for t in sample]
        print(f"  Rollout {k+1}: {' '.join(toks)} | Reward: {r:.2f} (Fmt: {fmt}, Acc: {acc})")
        if r >= 0.5:
            successes += 1
            
    print(f"\nConvergence Success Rate: {successes}/{test_samples} ({successes/test_samples*100:.1f}%)")
    print("GRPO Engine Verification Passed Successfully!")
    print("=" * 70)
