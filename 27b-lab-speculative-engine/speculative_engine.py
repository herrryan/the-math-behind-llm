"""
Stage 6: The Speculative Decoding & INT4 Quantization Engine
The Math Behind Large Language Models - Lab 06

A complete, self-contained inference acceleration engine in ~220 lines of pure Python.
Zero external libraries: No PyTorch, no HuggingFace, no NumPy.
Dependencies: Python standard library only (math, random).

Corresponds to:
- Chapter 26: Speculative Decoding & Rejection Sampling Mathematics
- Chapter 27: Quantization (INT4 / INT8 Uniform Symmetric Quantization)
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
        
        # Quantize upon initialization
        self.scale = 1.0
        self.quantized_weights = []
        self.packed_bytes = bytearray()
        self.quantize()

    def quantize(self):
        """Calculates scale factor s = max|W| / Qmax and quantizes weights."""
        max_val = max(abs(w) for row in self.weight_fp for w in row)
        self.scale = max_val / self.qmax if max_val > 0 else 1.0

        # Quantize to integer range [qmin, qmax]
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

        # For INT4: Pack two 4-bit nibbles into each 8-bit byte
        self.packed_bytes.clear()
        if self.bits == 4:
            for i in range(0, len(flat_ints), 2):
                q0 = flat_ints[i] & 0x0F
                q1 = (flat_ints[i+1] & 0x0F) if (i + 1 < len(flat_ints)) else 0
                packed = (q1 << 4) | q0
                self.packed_bytes.append(packed)

    def forward(self, x: list) -> list:
        """
        Matrix-vector multiplication using quantized weights.
        Leverages distributive law: sum(s * q_ij * x_j) = s * sum(q_ij * x_j).
        Accumulation happens in fast integers before a single float scale multiply!
        """
        out = []
        for row in self.quantized_weights:
            int_acc = sum(q * xi for q, xi in zip(row, x))
            out.append(int_acc * self.scale)
        return out

    def memory_footprint_bytes(self) -> dict:
        fp32_bytes = self.in_features * self.out_features * 4
        fp16_bytes = self.in_features * self.out_features * 2
        actual_bytes = len(self.packed_bytes) + 4 if self.bits == 4 else (self.in_features * self.out_features) + 4
        return {
            "fp32_bytes": fp32_bytes,
            "fp16_bytes": fp16_bytes,
            "quantized_bytes": actual_bytes,
            "compression_vs_fp16": fp16_bytes / actual_bytes
        }


# =====================================================================
# 2. Probability Math Utilities: Softmax & Residual Sampling (Chapter 26)
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
    """The authoritative high-capacity language model (e.g. 70B parameter LLM)."""
    def __init__(self):
        random.seed(101)
        # Authoritative transition table (current_token -> next_token logits)
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
        """Simulates parallel batch forward pass in a single GPU kernel."""
        self.forward_calls += 1  # Notice: Only 1 forward call for all tokens!
        return [softmax(self.logits_table.get(t, [1.0] * VOCAB_SIZE), temperature=0.7) for t in tokens]


class DraftModel:
    """The fast lightweight approximation model (e.g. 1B parameter assistant)."""
    def __init__(self):
        random.seed(202)
        # Approximate transition table (mostly matches target, but occasionally diverges)
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

def speculative_step(target: TargetModel, draft: DraftModel, current_token: str, gamma: int = 3) -> tuple:
    """
    Executes a single Speculative Decoding round:
      1. Draft Phase: Draft model generates gamma candidate tokens serially.
      2. Verify Phase: Target model evaluates all candidate tokens in ONE parallel forward pass.
      3. Rejection Sampling: Accept tokens according to acceptance probability alpha = min(1, p/q).
      4. Recovery Distribution: If rejected, sample correction from max(0, p - q).
    """
    draft_tokens = []
    draft_probs = []
    
    # 1. Draft Phase (Serial, but ultra-fast on tiny model)
    curr = current_token
    for _ in range(gamma):
        q_dist = draft.get_distribution(curr)
        next_id = sample_from_distribution(q_dist)
        tok = VOCAB[next_id]
        draft_tokens.append((next_id, tok))
        draft_probs.append(q_dist)
        curr = tok

    # 2. Parallel Target Model Verification
    # Notice: Target model processes [current_token + all draft_tokens] in ONE forward pass!
    # This simultaneously produces verification distributions for all draft tokens AND bonus token!
    eval_tokens = [current_token] + [t[1] for t in draft_tokens]
    target_probs = target.evaluate_batch(eval_tokens)

    # 3. Rejection Sampling Loop
    accepted_tokens = []
    correction_token = None

    for i in range(gamma):
        cand_id, cand_tok = draft_tokens[i]
        q_val = draft_probs[i][cand_id]
        p_val = target_probs[i][cand_id]

        acceptance_prob = min(1.0, p_val / max(1e-8, q_val))
        u = random.random()

        if u < acceptance_prob:
            # Token accepted!
            accepted_tokens.append(cand_tok)
        else:
            # Token rejected! Sample correction from residual distribution p'(x)
            residual_dist = [max(0.0, p - q) for p, q in zip(target_probs[i], draft_probs[i])]
            sum_res = sum(residual_dist)
            if sum_res > 1e-8:
                norm_res = [r / sum_res for r in residual_dist]
                corr_id = sample_from_distribution(norm_res)
            else:
                corr_id = sample_from_distribution(target_probs[i])
            
            correction_token = VOCAB[corr_id]
            accepted_tokens.append(correction_token)
            # Discard all remaining draft tokens
            break

    # If all gamma tokens were accepted, bonus token is retrieved directly from the final logit
    # of the very same forward pass at ZERO extra computational cost!
    if len(accepted_tokens) == gamma and correction_token is None:
        bonus_id = sample_from_distribution(target_probs[gamma])
        bonus_tok = VOCAB[bonus_id]
        accepted_tokens.append(bonus_tok)

    return accepted_tokens, [t[1] for t in draft_tokens]


# =====================================================================
# 5. Benchmark & Validation Routine
# =====================================================================

def run_simulation():
    print("=====================================================================")
    print("Lab 06: Speculative Decoding & INT4 Quantization Engine Simulation")
    print("=====================================================================")

    # 1. Demonstrate INT4 Quantization Memory Savings
    print("\n[PART 1: INT4 Uniform Symmetric Quantization Audit]")
    linear_layer = QuantizedLinear(in_features=64, out_features=64, bits=4)
    mem_stats = linear_layer.memory_footprint_bytes()
    print(f"  * Linear Layer: {linear_layer.in_features} x {linear_layer.out_features} weights")
    print(f"  * FP32 Footprint:        {mem_stats['fp32_bytes']} bytes")
    print(f"  * FP16 Footprint:        {mem_stats['fp16_bytes']} bytes")
    print(f"  * Packed INT4 Footprint: {mem_stats['quantized_bytes']} bytes")
    print(f"  * Effective Compression vs FP16: {mem_stats['compression_vs_fp16']:.2f}x (75% memory saved!)")

    # 2. Benchmark Autoregressive vs Speculative Decoding
    print("\n[PART 2: Speculative Decoding vs Standard Autoregressive Generation]")
    
    # Baseline Autoregressive Target Model Generation
    random.seed(42)
    target_base = TargetModel()
    curr = "the"
    base_generated = [curr]
    while len(base_generated) < 8 and curr != "<eos>":
        p_dist = target_base.get_distribution(curr)
        nxt_id = sample_from_distribution(p_dist)
        curr = VOCAB[nxt_id]
        base_generated.append(curr)

    print(f"Baseline Autoregressive Generated Sequence:")
    print(f"  {' '.join(base_generated)}")
    print(f"  Target Forward Calls: {target_base.forward_calls}")
    print(f"  Tokens per Forward Call: {len(base_generated) / target_base.forward_calls:.2f}")

    # Speculative Decoding Generation
    print("\nRunning Speculative Engine (gamma = 3):")
    random.seed(42)
    target_spec = TargetModel()
    draft_spec = DraftModel()
    
    spec_generated = ["the"]
    total_rounds = 0
    total_drafted = 0
    total_accepted_from_draft = 0

    while len(spec_generated) < 8 and spec_generated[-1] != "<eos>":
        total_rounds += 1
        curr = spec_generated[-1]
        accepted, drafted = speculative_step(target_spec, draft_spec, curr, gamma=3)
        total_drafted += len(drafted)
        
        # Count accepted tokens from draft (excluding bonus/correction)
        num_accepted = len(accepted)
        spec_generated.extend(accepted)
        print(f"  Round {total_rounds:02d}: Drafted {drafted} -> Produced {accepted} ({len(accepted)} tokens in 1 target call)")
        
        if "<eos>" in spec_generated:
            break

    # Trim to match sequence length
    spec_generated = spec_generated[:8]

    print("\n=====================================================================")
    print("Final Performance Audit & Acceleration Metrics")
    print("=====================================================================")
    print(f"Speculative Generated Sequence:")
    print(f"  {' '.join(spec_generated)}")
    print(f"  Total Tokens Generated:       {len(spec_generated)}")
    print(f"  Target Model Forward Passes:  {target_spec.forward_calls}")
    print(f"  Tokens per Forward Pass:      {len(spec_generated) / target_spec.forward_calls:.2f} tok/call")
    
    speedup = target_base.forward_calls / target_spec.forward_calls
    print(f"  Net Inference Acceleration:   {speedup:.2f}x Speedup over Autoregressive Decoding!")
    print("=====================================================================")


if __name__ == "__main__":
    run_simulation()
