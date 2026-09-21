# Chapter E22: Chunked Prefill & Prefix Caching

## Step 1: Hardware Intuition
Imagine an amusement park ride where a massive group of 500 tourists arrives. If the ride operator stops all regular visitors to load the entire tourist group, regular visitors wait for an hour. Instead, the operator lets 50 tourists board alongside 50 regular visitors every 5 minutes, keeping lines moving smoothly.

## Step 2: Silicon Micro-Mechanics
- **Chunked Prefill:** Slices large prompt prefills into uniform chunks (e.g. 512 tokens) and co-schedules them alongside decode requests.
- **Interference Mitigation:** Bounds Inter-Token Latency (ITL) spikes for existing streaming users.
- **Automatic Prefix Caching (APC):** Organizes KV pages in a Radix Tree. Common prompt prefixes (system instructions, multi-turn chat history) reuse cached KV blocks via pointer lookups.

## Step 3: Cross-Component Coupling
- **TTFT Elimination:** High prefix hit rates ($h > 80\%$) convert $O(S)$ prefill compute into an $O(1)$ pointer copy, dropping TTFT from seconds to milliseconds.

## Step 4: The Exact Performance Formula
$$\text{TTFT}_{\text{cached}} = (1 - h) \cdot \text{TTFT}_{\text{full}} + \epsilon_{\text{lookup}}$$
where $h$ is the prefix cache hit rate.

## Step 5: Concrete Benchmark Walkthrough
Benchmarking a multi-turn customer support agent with a 2,000-token system prompt:
- Without prefix caching: Every turn recomputes 2,000 tokens ($\\text{TTFT} \approx 120\text{ ms}$).
- With prefix caching: Reuses cached KV pages ($h = 100\%$ for prefix; $\text{TTFT} drops to $4\text{ ms}$).

## Step 6: Core Systems Takeaway
> Chunked prefill tames decode latency spikes, while prefix caching eliminates redundant prompt compute. Together, they form the backbone of predictable multi-tenant LLM serving.
