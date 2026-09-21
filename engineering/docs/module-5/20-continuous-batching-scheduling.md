# Chapter E20: Continuous Batching & Dynamic Scheduling

## Step 1: Hardware Intuition
Imagine a ski chairlift.
- **Static Batching:** The chairlift waits until 4 people arrive, rides up the mountain, and refuses to let anyone new get on or off until all 4 skiers finish their entire afternoon of skiing.
- **Continuous Batching:** Skiers get on empty chairs as soon as one arrives; when a skier finishes, their seat is immediately filled by the next skier waiting in line.

## Step 2: Silicon Micro-Mechanics
- **Static Batching Flaw:** Padding shorter sequences with zeroes until the longest sequence in the batch completes wastes up to 70% of GPU compute on useless padding tokens.
- **Continuous / In-Flight Batching (Orca, vLLM):** Operates at the iteration step level. Requests are added and retired dynamically at every forward pass.
- **Ragged Tensors:** Packing non-uniform sequence lengths into a single 1D vector using offset pointers to eliminate all padding.

## Step 3: Cross-Component Coupling
- **Integration with PagedAttention:** Continuous batching requires dynamic page allocation to append KV blocks on a per-step basis without memory defragmentation stalls.

## Step 4: The Exact Performance Formula
$$\text{Throughput} = \frac{\sum_{i=1}^N \text{Tokens Generated}_i}{T_{\text{wall\_clock}}}$$
$$\text{Compute Efficiency Gain} = \frac{\text{Useful Tokens}}{\text{Useful Tokens} + \text{Padding Tokens}}$$

## Step 5: Concrete Benchmark Walkthrough
Benchmarking 64 requests with lengths uniformly distributed from 50 to 1000 tokens:
- Static batching: Padded to $1000$ tokens per request ($64,000$ tokens processed, of which $33,600$ are useless padding $\to 52.5\%$ waste).
- Continuous batching: Exactly $30,400$ tokens processed ($0\%$ padding waste, $2.1\times$ throughput speedup).

## Step 6: Core Systems Takeaway
> Never pad tokens in production. Iteration-level continuous batching eliminates tensor bubbles and doubles effective serving throughput.
