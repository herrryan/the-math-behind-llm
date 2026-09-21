# Lab E3: The Paged KV Cache Engine

## Objective
Construct a zero-dependency Python and NumPy prototype of virtual memory page-table KV cache management (PagedAttention) for high-concurrency autoregressive decoding.

## Learning Milestones
1. Implement a physical memory block allocator with fixed block size (16 tokens).
2. Maintain per-sequence page tables mapping logical token indices to physical block IDs.
3. Handle dynamic token generation and copy-on-write memory sharing for branching prompts.
4. Measure memory fragmentation reduction compared to static buffer allocation.

## Key Code Components
- Physical block manager (`BlockAllocator`).
- Page table directory (`LogicalToPhysicalMap`).
- Dynamic token append and eviction simulation.
