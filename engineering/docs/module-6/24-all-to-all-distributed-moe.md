# Chapter E24: All-to-All Dispatch & Combine: Distributed MoE Bottlenecks

## Step 1: Hardware Intuition
Imagine 8 sorting hubs in 8 different cities. Each city receives mail meant for all 8 cities. Every hour, all 8 hubs must simultaneously exchange mail trucks in an all-to-all highway gridlock. If the highways are narrow, trucks sit in traffic jams and the sorting centers grind to a halt.

## Step 2: Silicon Micro-Mechanics
- **Expert Parallelism (EP):** Sharding $E$ experts across $N$ GPUs (e.g. 8 experts per GPU across 8 nodes).
- **The Two Communication Phases:**
  1. **All-to-All Dispatch:** Routes token activations from their origin GPU to the GPU hosting the assigned expert.
  2. **All-to-All Combine:** Returns processed expert outputs back to the original token sequence positions.
- **The Network Wall:** All-to-All transfers stress crossbar switches and inter-node network links.

## Step 3: Cross-Component Coupling
- **Overlap Innovations (DeepSeek DualPipe):** Bi-directional pipelining that overlaps All-to-All communication with FFN computation of adjacent microbatches.

## Step 4: The Exact Performance Formula
$$\text{All-to-All Volume per Layer} = 2 \times K \times B \times S \times d_{\text{model}} \times \text{sizeof(dtype)}$$

## Step 5: Concrete Benchmark Walkthrough
Profiling All-to-All dispatch for 256 tokens per GPU across 64 GPUs ($d = 7168$, FP8 = 1 byte):
- Communication volume per GPU: $2 \times 8 \times 256 \times 7168 \times 1 \text{ byte} \approx 29.4\text{ MB}$ per layer.
- Across 60 layers: $1.76\text{ GB}$ transferred per step.
- Over 400 Gbps InfiniBand: takes $35.2\text{ ms}$, accounting for up to 40% of step time without communication overlap.

## Step 6: Core Systems Takeaway
> Distributed MoE efficiency is governed by All-to-All network crossbar bandwidth. Overlapping dispatch communication with expert computation is essential to unlocking sparse scaling.
