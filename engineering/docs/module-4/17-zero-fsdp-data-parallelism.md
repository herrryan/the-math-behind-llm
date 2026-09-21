# Chapter E17: Data Parallelism & Zero Redundancy: ZeRO-1, 2, 3 and FSDP

## Step 1: Hardware Intuition
Imagine a team of 64 researchers analyzing survey responses. Standard data parallelism requires every researcher to purchase and carry the entire 20-volume encyclopedia set in their backpack. ZeRO-3 assigns each researcher only 1 volume. When Researcher 5 needs to look up a word from Volume 2, they ask Researcher 2 over the intercom, read the definition, and hand the page back.

## Step 2: Silicon Micro-Mechanics
- **Memory Footprint of Training:**
  - Parameters: $2P$ bytes (FP16).
  - Gradients: $2P$ bytes (FP16).
  - Adam Optimizer: $4P$ (master weights) $+ 4P$ (momentum) $+ 4P$ (variance) $= 12P$ bytes.
  - Total: $16P$ bytes!
- **ZeRO Stages (Rajbhandari et al.):**
  - ZeRO-1: Shard optimizer states ($4\times$ memory reduction).
  - ZeRO-2: Shard optimizer states and gradients ($8\times$ memory reduction).
  - ZeRO-3 (FSDP): Shard optimizer, gradients, and model parameters. Fetches parameters on the fly via All-Gather and discards them immediately after forward/backward compute.

## Step 3: Cross-Component Coupling
- **Communication Trade-off:** ZeRO-3 increases total communication volume by 50% (adds 1 All-Gather per forward layer), requiring communication-compute overlap to hide latency.

## Step 4: The Exact Performance Formula
$$M_{\text{ZeRO-3}} = \frac{2P + 2P + 12P}{N_{\text{GPUs}}} + M_{\text{activations}} = \frac{16P}{N} + M_{\text{activations}}$$

## Step 5: Concrete Benchmark Walkthrough
Training a 70B parameter model across 64 GPUs:
- Standard DDP memory per GPU: $16 \times 70\text{ GB} = 1120\text{ GB}$ (impossible on any single GPU).
- ZeRO-3 memory per GPU: $\frac{1120\text{ GB}}{64} \approx 17.5\text{ GB}$ (comfortably fits on an 80GB A100/H100!).

## Step 6: Core Systems Takeaway
> ZeRO-3 / FSDP completely democratizes massive LLM training by sharding the 16P memory burden across all cluster GPUs, trading 50% extra communication for boundless memory scalability.
