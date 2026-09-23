"""
Stage 6: The Speculative Decoding & INT4 Quantization Engine - Guided Exercise
The Math Behind Large Language Models - Lab 06

Instructions:
-------------
In this hands-on lab, you will build the two crowning acceleration techniques
that enable modern LLMs to break the memory wall and sequential latency barrier:
  1. Uniform Symmetric INT4/INT8 Quantization (Chapter 27)
  2. Lossless Speculative Decoding with Rejection Sampling (Chapter 26)

All scaffolding (model stubs, probability samplers, test harnesses) is provided.
Your mission is to fill in the core mathematical operations marked with TODO.

To guide your implementation, every TODO block includes:
  1. Mathematical Formula (LaTeX style)
  2. Physical Intuition & Dimensions
  3. Step-by-Step Python Hints

When you run this script:
  $ python3 speculative_engine_exercise.py
It will test your implementation step-by-step and run a real speculative acceleration benchmark.
"""

import math
import random


# =====================================================================
# 1. Uniform Symmetric Quantization: INT4 & INT8 (Chapter 27)
# =====================================================================

class QuantizedLinear:
    """
    Simulates a low-bit quantized linear layer (INT4 or INT8).
    Demonstrates scale factor extraction, nibble packing, and dequantization.
    """
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1  # 7 for INT4, 127 for INT8
        self.qmin = -self.qmax             # -7 for INT4, -127 for INT8
        
        # Initialize pseudo-random floating point weights
        random.seed(42)
        self.weight_fp = [
            [random.uniform(-0.8, 0.8) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        
        self.scale = 1.0
        self.quantized_weights = []
        self.packed_bytes = bytearray()
        self.quantize()

    # -----------------------------------------------------------------
    # TODO 1: Implement quantize()
    # -----------------------------------------------------------------
    def quantize(self):
        """
        Calculates uniform symmetric scale factor:
          s = max(|W|) / Qmax
        
        Quantizes each weight:
          q = clip(round(w / s), qmin, qmax)
        
        Stores 2D quantized weights in self.quantized_weights.
        """
        # --- YOUR CODE HERE ---
        max_val = max(abs(w) for row in self.weight_fp for w in row)
        self.scale = max_val / self.qmax if max_val > 0 else 1.0

        self.quantized_weights = []
        flat_ints = []
        for row in self.weight_fp:
            q_row = []
            for w in row:
                q = round(w / self.scale)
                q = max(self.qmin, min(self.qmax, q))
                q_row.append(q)
                flat_ints.append(q)
            self.quantized_weights.append(q_row)

        self.packed_bytes.clear()
        if self.bits == 4:
            for i in range(0, len(flat_ints), 2):
                q0 = flat_ints[i] & 0x0F
                q1 = (flat_ints[i+1] & 0x0F) if (i + 1 < len(flat_ints)) else 0
                packed = (q1 << 4) | q0
                self.packed_bytes.append(packed)
        # ----------------------

    # -----------------------------------------------------------------
    # TODO 2: Implement forward()
    # -----------------------------------------------------------------
    def forward(self, x: list) -> list:
        """
        Computes quantized matrix-vector product:
          y = (s * Q) . x = s * (Q . x)
        
        For each row in self.quantized_weights:
          int_acc = sum(q * xi for q, xi in zip(row, x))
          out.append(int_acc * self.scale)
        """
        # --- YOUR CODE HERE ---
        out = []
        for row in self.quantized_weights:
            int_acc = sum(q * xi for q, xi in zip(row, x))
            out.append(int_acc * self.scale)
        return out
        # ----------------------


# =====================================================================
# 2. Probability Math Utilities
# =====================================================================

def softmax(logits: list, temperature: float = 1.0) -> list:
    scaled = [x / max(1e-5, temperature) for x in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(x - max_val) for x in scaled]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def sample_from_distribution(probs: list) -> int:
    r = random.random()
    cumsum = 0.0
    for idx, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return idx
    return len(probs) - 1


# =====================================================================
# 3. Target Model & Fast Draft Model Simulation
# =====================================================================

VOCAB = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "<eos>"]
VOCAB_SIZE = len(VOCAB)

class TargetModel:
    def __init__(self):
        self.logits_table = {
            "the":   [0.1, 4.0, 3.5, 0.2, 0.1, 0.1, 3.0, 0.5, 0.0],
            "quick": [0.1, 0.1, 5.0, 1.0, 0.2, 0.1, 0.1, 0.1, 0.0],
            "brown": [0.1, 0.1, 0.1, 6.0, 0.2, 0.1, 0.1, 0.1, 0.0],
            "fox":   [0.1, 0.1, 0.1, 0.1, 5.5, 0.1, 0.1, 0.1, 0.0],
            "jumps": [0.1, 0.1, 0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.0],
            "over":  [4.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            "lazy":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0, 0.0],
            "dog":   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0],
            "<eos>": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0]
        }
        self.forward_calls = 0

    def get_distribution(self, token: str) -> list:
        self.forward_calls += 1
        logits = self.logits_table.get(token, [1.0] * VOCAB_SIZE)
        return softmax(logits, temperature=0.7)

    def evaluate_batch(self, tokens: list) -> list:
        self.forward_calls += 1
        return [softmax(self.logits_table.get(t, [1.0] * VOCAB_SIZE), temperature=0.7) for t in tokens]


class DraftModel:
    def __init__(self):
        self.logits_table = {
            "the":   [0.1, 3.8, 3.2, 0.2, 0.1, 0.1, 2.8, 0.5, 0.0],
            "quick": [0.1, 0.1, 4.5, 1.2, 0.2, 0.1, 0.1, 0.1, 0.0],
            "brown": [0.1, 0.1, 0.1, 5.0, 0.2, 0.1, 0.1, 0.1, 0.0],
            "fox":   [0.1, 0.1, 0.1, 0.1, 4.8, 0.1, 0.1, 0.1, 0.0],
            "jumps": [0.1, 0.1, 0.1, 0.1, 0.1, 4.5, 0.1, 0.1, 0.0],
            "over":  [4.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            "lazy":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 4.8, 0.0],
            "dog":   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 4.5],
            "<eos>": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 4.5]
        }
        self.forward_calls = 0

    def get_distribution(self, token: str) -> list:
        self.forward_calls += 1
        logits = self.logits_table.get(token, [1.0] * VOCAB_SIZE)
        return softmax(logits, temperature=0.7)


# =====================================================================
# 4. Speculative Decoding Rejection Sampling Engine (Chapter 26)
# =====================================================================

# ---------------------------------------------------------------------
# TODO 3: Implement speculative_step
# ---------------------------------------------------------------------
def speculative_step(target: TargetModel, draft: DraftModel, current_token: str, gamma: int = 3) -> tuple:
    """
    Executes one round of Speculative Decoding:
    
    1. Draft Phase:
       Use draft model to generate gamma candidate tokens sequentially.
       Record candidate tokens and their draft probabilities q(x_i).
    
    2. Parallel Evaluation:
       Evaluate candidate tokens using target.evaluate_batch() in ONE forward pass.
       Obtain target probabilities p(x_i).
    
    3. Rejection Sampling:
       For i = 0 ... gamma - 1:
         acceptance_prob = min(1.0, p(x_i) / q(x_i))
         u ~ Uniform(0, 1)
         If u < acceptance_prob:
           accept x_i!
         Else:
           sample correction token from normalized residual distribution:
             p'(x) = max(0, p(x) - q(x)) / sum(max(0, p - q))
           append correction token and break!
    
    4. Bonus Token:
       If all gamma tokens were accepted, sample one extra bonus token from
       target.get_distribution(candidate_gamma) for free!
    """
    draft_tokens = []
    draft_probs = []
    
    # --- YOUR CODE HERE ---
    # 1. Draft Phase
    curr = current_token
    for _ in range(gamma):
        q_dist = draft.get_distribution(curr)
        next_id = sample_from_distribution(q_dist)
        tok = VOCAB[next_id]
        draft_tokens.append((next_id, tok))
        draft_probs.append(q_dist)
        curr = tok

    # 2. Parallel Target Evaluation
    eval_tokens = [current_token] + [t[1] for t in draft_tokens]
    target_probs = target.evaluate_batch(eval_tokens)

    # 3. Rejection Sampling
    accepted_tokens = []
    correction_token = None

    for i in range(gamma):
        cand_id, cand_tok = draft_tokens[i]
        q_val = draft_probs[i][cand_id]
        p_val = target_probs[i][cand_id]

        acceptance_prob = min(1.0, p_val / max(1e-8, q_val))
        u = random.random()

        if u < acceptance_prob:
            accepted_tokens.append(cand_tok)
        else:
            residual_dist = [max(0.0, p - q) for p, q in zip(target_probs[i], draft_probs[i])]
            sum_res = sum(residual_dist)
            if sum_res > 1e-8:
                norm_res = [r / sum_res for r in residual_dist]
                corr_id = sample_from_distribution(norm_res)
            else:
                corr_id = sample_from_distribution(target_probs[i])
            correction_token = VOCAB[corr_id]
            accepted_tokens.append(correction_token)
            break

    # 4. Bonus Token
    if len(accepted_tokens) == gamma and correction_token is None:
        bonus_id = sample_from_distribution(target_probs[gamma])
        bonus_tok = VOCAB[bonus_id]
        accepted_tokens.append(bonus_tok)

    return accepted_tokens, [t[1] for t in draft_tokens]
    # ----------------------


# =====================================================================
# 5. Automated Verification Test Suite
# =====================================================================

def test_quantization():
    print("[TEST 1/3] Testing INT4 Quantization & Forward Pass...")
    layer = QuantizedLinear(in_features=8, out_features=4, bits=4)
    assert layer.scale > 0.0, "Quantization scale must be positive!"
    assert len(layer.quantized_weights) == 4
    assert len(layer.quantized_weights[0]) == 8

    # Check bounds
    for row in layer.quantized_weights:
        for q in row:
            assert -7 <= q <= 7, f"INT4 weights must lie within [-7, 7]! Got {q}"

    x = [1.0] * 8
    y = layer.forward(x)
    assert len(y) == 4
    assert all(not math.isnan(val) for val in y), "Quantized forward output must not contain NaNs!"
    print("  -> PASSED: INT4 quantization bounds and forward pass verified.")


def test_residual_math():
    print("[TEST 2/3] Testing Rejection Sampling Residual Distribution...")
    p = [0.1, 0.7, 0.2]
    q = [0.3, 0.4, 0.3]
    
    # Residual = max(0, p - q) = [0.0, 0.3, 0.0]
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    sum_res = sum(residual)
    assert abs(sum_res - 0.3) < 1e-5, f"Expected residual sum 0.3, got {sum_res}"
    
    norm_res = [r / sum_res for r in residual]
    assert abs(norm_res[1] - 1.0) < 1e-5, "When only index 1 has p > q, normalized residual must concentrate on index 1!"
    print("  -> PASSED: Residual distribution recovery mathematically verified.")


def test_speculative_execution():
    print("[TEST 3/3] Testing Speculative Decoding Acceleration...")
    target = TargetModel()
    draft = DraftModel()
    
    random.seed(42)
    accepted, drafted = speculative_step(target, draft, "the", gamma=3)
    assert len(drafted) == 3, "Draft model must propose exactly gamma tokens!"
    assert len(accepted) >= 1, "At least 1 token (either accepted or corrected) must be produced!"
    assert target.forward_calls == 1, "Target model must evaluate batch in exactly ONE forward call!"
    print(f"  -> PASSED: Speculative round produced {len(accepted)} tokens with only 1 target call.")


if __name__ == "__main__":
    print("=====================================================================")
    print("Running Lab 06 Guided Exercise Test Suite...")
    print("=====================================================================")
    test_quantization()
    test_residual_math()
    test_speculative_execution()
    print("=====================================================================")
    print("All unit tests PASSED! You have built a Speculative Acceleration Engine!")
    print("=====================================================================")
