# Chapter E21: Speculative Decoding: Draft-Verify Silicon Acceleration

## Step 1: Hardware Intuition
Imagine a slow professor and a fast student. The student rapidly drafts 5 potential sentences for an essay. The professor glances at all 5 sentences simultaneously in 1 second, approves the first 4, corrects the 5th, and moves on. The essay progresses $4\times$ faster than if the professor had penned every word from scratch.

## Step 2: Silicon Micro-Mechanics
- **Memory-Bound Bottleneck Exploitation:** Generating 1 token from a 70B model requires reading 140 GB of weights. Evaluating 5 tokens simultaneously takes virtually the same time because prefill evaluation is compute-bound.
- **The Speculative Loop (Leviathan et al.):**
  1. Draft model (e.g. 1B) autoregressively drafts $K$ candidate tokens.
  2. Target model (e.g. 70B) validates all $K$ candidates in a **single parallel forward pass**.
  3. Acceptance via modified rejection sampling guarantees exact mathematical distribution equivalence.

## Step 3: Cross-Component Coupling
- **Draft Model Selection:** Optimal draft model size is typically 5-10% of the target model's parameter count.
- **Speculative Trees (Medusa / EAGLE):** Using multiple decoding heads to branch candidate paths, boosting acceptance rates.

## Step 4: The Exact Performance Formula
$$\text{Expected Speedup} = \frac{1 + \alpha + \alpha^2 + \dots + \alpha^K}{1 + \frac{c_{\text{draft}}}{c_{\text{target}}} \cdot K}$$
where $\alpha$ is the average token acceptance rate.

## Step 5: Concrete Benchmark Walkthrough
Pairing LLaMA-3 70B with LLaMA-3 8B at $K = 5$ speculative tokens, assuming acceptance rate $\alpha = 0.75$:
- Expected accepted tokens per step: $1 + 0.75 + 0.56 + 0.42 + 0.31 = 3.04$ tokens.
- Draft cost ratio: $\approx 0.12$.
- Realized wall-clock speedup: $2.1\times$ faster generation with 100% mathematical output fidelity!

## Step 6: Core Systems Takeaway
> Speculative decoding trades cheap idle compute during the memory-bound decode phase to generate multiple tokens per HBM weight sweep, delivering pure wall-clock acceleration without changing output quality.
