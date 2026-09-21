# Chapter E26: Measuring True Performance: MFU, HFU, and Profiling

## Step 1: Hardware Intuition
A sports car speedometer might say "Top Speed: 300 km/h". But in city traffic with red lights, sharp turns, and speed bumps, your average speed is only 45 km/h. Model FLOPs Utilization (MFU) is the true speedometer of AI engineering: it measures what percentage of the GPU's theoretical top speed was converted into real token learning.

## Step 2: Silicon Micro-Mechanics
- **Model FLOPs Utilization (MFU):**
  $$\text{MFU} = \frac{\text{Tokens/Sec} \times 6P}{\text{Total Hardware Peak FLOPS}}$$
- **Hardware FLOPs Utilization (HFU):** Includes recomputation and backward overhead:
  $$\text{HFU} = \frac{\text{Actual Executed FLOPs/Sec}}{\text{Peak FLOPS}}$$
- **Profiling Tools:**
  - `nsys` (NVIDIA Nsight Systems): System-level timeline, CUDA kernel launches, NVLink collective stalls.
  - `ncu` (NVIDIA Nsight Compute): Kernel-level micro-architecture profiling, memory warp stalls, SM occupancy.

## Step 3: Cross-Component Coupling
- **Efficiency Thresholds:**
  - Naive implementations: 15-25% MFU.
  - Good implementations: 40-50% MFU.
  - World-class implementations (Megatron / LLaMA-3): 55-65% MFU.

## Step 4: The Exact Performance Formula
$$\text{FLOPs per Token (Training)} \approx 6P$$
$$\text{FLOPs per Token (Inference)} \approx 2P$$

## Step 5: Concrete Benchmark Walkthrough
Calculating MFU for a cluster of 512x H100 GPUs ($989\text{ TFLOPS}$ each) training a 70B model at 2,500 tokens/sec:
- Cluster peak: $512 \times 989 \times 10^{12} = 5.06 \times 10^{17} \text{ FLOPS}$.
- Useful FLOPs: $2500 \times (6 \times 70 \times 10^9) = 1.05 \times 10^{15} \text{ FLOPs/s}$.
- $\text{MFU} = \frac{1.05 \times 10^{15}}{5.06 \times 10^{17}} \approx 52.1\%$ (Solid, production-grade efficiency!).

## Step 6: Core Systems Takeaway
> Datasheet TFLOPS are marketing; Model FLOPs Utilization (MFU) is engineering truth. Profile before optimizing, and measure your systems code against the physical limits of the machine.
