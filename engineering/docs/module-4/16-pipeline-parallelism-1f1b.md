# Chapter E16: Pipeline Parallelism (PP): 1F1B Scheduling & Bubble Analysis

## Step 1: Hardware Intuition
Imagine an automotive assembly line with 4 stations: Chassis, Engine, Body, and Paint. If Worker 1 builds the entire car before passing it to Worker 2, three workers stand idle at any moment. By breaking the car into 20 small micro-parts flowing continuously through the stations, all 4 workers stay busy simultaneously, with only tiny startup and shutdown delays.

## Step 2: Silicon Micro-Mechanics
- **Partitioning:** Splitting 80 layers across $p$ pipeline stages (e.g. 10 layers per GPU across 8 GPUs).
- **The Pipeline Bubble:** The idle time during pipeline ramp-up and ramp-down:
  $$F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$$
  where $m$ is the number of microbatches.
- **1F1B Scheduling:** Once warm, each stage alternates one forward pass with one backward pass, bounding activation memory to at most $p$ microbatches.

## Step 3: Cross-Component Coupling
- **Activation Stashing:** Stage 0 must retain activations for $p$ microbatches, increasing memory pressure on early pipeline stages.
- **Interleaved 1F1B:** Assigning multiple non-contiguous virtual stages per GPU to shrink the bubble factor by $2\times$.

## Step 4: The Exact Performance Formula
$$\text{Bubble Overhead Percentage} = \frac{p - 1}{m} \times 100\% \quad (\text{for } m \gg p)$$

## Step 5: Concrete Benchmark Walkthrough
Pipeline parallelism on 8 stages ($p = 8$) with 32 microbatches ($m = 32$):
- Bubble fraction: $\frac{8 - 1}{32 + 8 - 1} = \frac{7}{39} \approx 17.9\%$.
- With interleaved 1F1B (virtual stages $v = 2$): bubble fraction drops to $9.8\%$.

## Step 6: Core Systems Takeaway
> Pipeline Parallelism allows scaling across slow inter-node networks, but requires high microbatch counts to keep the idle bubble fraction below 10%.
