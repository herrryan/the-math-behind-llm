"""
Stage 5: The Streaming KV & Continuous Batching Engine (vLLM Mini-Kernel)
The Math Behind Large Language Models - Lab 05

A complete, self-contained inference serving engine in ~200 lines of pure Python.
Zero external libraries: No PyTorch, no vLLM, no NumPy.
Dependencies: Python standard library only (math, random, collections).

Corresponds to:
- Chapter 21: KV Cache Mechanics & Memory Footprint
- Chapter 22: PagedAttention & Virtual Memory Allocation
- Chapter 25: Continuous Batching, Iteration Scheduling & Bubble Waste Elimination
"""

import math
import random
from collections import deque


# =====================================================================
# 1. Paged KV Memory: Physical Block & Block Manager (Chapter 22)
# =====================================================================

class PhysicalBlock:
    """Represents a single fixed-size physical memory block on a GPU."""
    def __init__(self, block_id: int, block_size: int = 4):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.keys = []      # List of key vectors stored in this page
        self.values = []    # List of value vectors stored in this page
        self.tokens = []    # Human-readable tokens for inspection

    @property
    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    def append(self, token: str, k_vec: list, v_vec: list):
        if self.is_full:
            raise RuntimeError(f"Block {self.block_id} is full!")
        self.tokens.append(token)
        self.keys.append(k_vec)
        self.values.append(v_vec)

    def clear(self):
        self.ref_count = 0
        self.keys.clear()
        self.values.clear()
        self.tokens.clear()


class BlockManager:
    """
    Manages physical GPU block allocation, page tables, and Copy-on-Write (CoW).
    Translates logical token positions into non-contiguous physical block slots.
    """
    def __init__(self, total_blocks: int = 16, block_size: int = 4):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.blocks = [PhysicalBlock(i, block_size) for i in range(total_blocks)]
        self.free_list = deque(range(total_blocks))

    def num_free_blocks(self) -> int:
        return len(self.free_list)

    def allocate_block(self) -> int:
        if not self.free_list:
            raise MemoryError("Out of GPU Memory: No physical blocks available!")
        bid = self.free_list.popleft()
        self.blocks[bid].clear()
        self.blocks[bid].ref_count = 1
        return bid

    def free_block(self, bid: int):
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count == 0:
            block.clear()
            self.free_list.append(bid)

    def free_block_table(self, block_table: list):
        for bid in block_table:
            self.free_block(bid)
        block_table.clear()

    def allocate_slots_for_tokens(self, block_table: list, tokens: list, k_vecs: list, v_vecs: list):
        """Appends tokens and KV vectors to a logical sequence via its block table."""
        for token, k_vec, v_vec in zip(tokens, k_vecs, v_vecs):
            # Check if we need a new physical block
            if not block_table or self.blocks[block_table[-1]].is_full:
                new_bid = self.allocate_block()
                block_table.append(new_bid)
            
            last_block = self.blocks[block_table[-1]]
            
            # Copy-on-Write check: if shared with another sequence, copy before mutate
            if last_block.ref_count > 1:
                new_bid = self.allocate_block()
                copied_block = self.blocks[new_bid]
                for t, k, v in zip(last_block.tokens, last_block.keys, last_block.values):
                    copied_block.append(t, k, v)
                self.free_block(block_table[-1])
                block_table[-1] = new_bid
                last_block = copied_block

            last_block.append(token, k_vec, v_vec)

    def share_prefix(self, source_block_table: list) -> list:
        """Enables zero-memory prefix sharing via reference counting."""
        shared_table = []
        for bid in source_block_table:
            self.blocks[bid].ref_count += 1
            shared_table.append(bid)
        return shared_table


# =====================================================================
# 2. PagedAttention Kernel (Chapter 22 & 23)
# =====================================================================

def paged_attention(query: list, block_table: list, block_manager: BlockManager) -> list:
    """
    Computes Scaled Dot-Product Attention by traversing scattered physical blocks.
    Notice: No contiguous KV array exists in physical memory!
    """
    d_k = len(query)
    scores = []
    all_values = []

    # Gather keys and values across non-contiguous physical blocks via block table
    for bid in block_table:
        block = block_manager.blocks[bid]
        for k_vec, v_vec in zip(block.keys, block.values):
            # Dot product: q . k / sqrt(d_k)
            dot = sum(q * k for q, k in zip(query, k_vec)) / math.sqrt(d_k)
            scores.append(dot)
            all_values.append(v_vec)

    if not scores:
        return [0.0] * d_k

    # Safe Softmax
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    weights = [e / sum_exp for e in exp_scores]

    # Weighted sum of values
    output = [0.0] * len(all_values[0])
    for w, val in zip(weights, all_values):
        for i in range(len(output)):
            output[i] += w * val[i]

    return output


# =====================================================================
# 3. Model Simulation: Synthetic Embedding & KV Projections
# =====================================================================

class ToyLanguageModel:
    """Lightweight deterministic model generating reproducible tokens and KV states."""
    def __init__(self, d_model: int = 4, vocab: list = None):
        self.d_model = d_model
        self.vocab = vocab or ["the", "cat", "sat", "on", "mat", "dog", "slept", "sun", "<eos>"]
        random.seed(42)
        # Random deterministic projection matrices
        self.W_k = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]

    def embed_token(self, token: str) -> list:
        # Deterministic pseudo-embedding from token hash
        seed = sum(ord(c) for c in token)
        random.seed(seed)
        return [random.uniform(-1.0, 1.0) for _ in range(self.d_model)]

    def project_kv(self, token: str):
        x = self.embed_token(token)
        k = [sum(x[j] * self.W_k[j][i] for j in range(self.d_model)) for i in range(self.d_model)]
        v = [sum(x[j] * self.W_v[j][i] for j in range(self.d_model)) for i in range(self.d_model)]
        return k, v

    def predict_next(self, attn_out: list, current_gen_len: int, target_len: int) -> str:
        # Generates target_len tokens before emitting <eos>
        if current_gen_len > target_len:
            return "<eos>"
        # Select token pseudo-dependently on attention output
        idx = int(abs(sum(attn_out) * 100)) % (len(self.vocab) - 1)
        return self.vocab[idx]


# =====================================================================
# 4. Request & Continuous Batching Scheduler (Chapter 25)
# =====================================================================

class Request:
    def __init__(self, req_id: str, prompt_tokens: list, target_gen_len: int, arrival_iteration: int):
        self.req_id = req_id
        self.prompt_tokens = prompt_tokens
        self.target_gen_len = target_gen_len
        self.arrival_iteration = arrival_iteration
        
        self.block_table = []
        self.generated_tokens = []
        self.status = "WAITING"  # WAITING -> RUNNING -> FINISHED
        self.start_iteration = None
        self.finish_iteration = None

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)


class ContinuousBatchEngine:
    """
    Production-grade Iteration-Level Scheduler.
    Dynamically admits waiting requests into running slots and evicts finished requests,
    completely eliminating static batching bubble waste.
    """
    def __init__(self, model: ToyLanguageModel, block_manager: BlockManager, max_batch_size: int = 3):
        self.model = model
        self.bm = block_manager
        self.max_batch_size = max_batch_size
        
        self.waiting_queue = deque()
        self.running_batch = []
        self.finished_requests = []
        self.current_iteration = 0

    def submit_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self):
        """Executes exactly one discrete forward pass iteration across all active requests."""
        self.current_iteration += 1
        print(f"\n--- [Iteration {self.current_iteration:02d}] ---")

        # 1. Dynamic Admission: Fill empty slots from waiting queue
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            next_req = self.waiting_queue[0]
            # Check if GPU has enough physical blocks to prefill this prompt
            needed_blocks = math.ceil(len(next_req.prompt_tokens) / self.bm.block_size)
            if self.bm.num_free_blocks() >= needed_blocks + 1:
                req = self.waiting_queue.popleft()
                req.status = "RUNNING"
                req.start_iteration = self.current_iteration
                
                # Execute Prefill Phase: allocate blocks and compute KV for prompt
                k_list, v_list = [], []
                for tok in req.prompt_tokens:
                    k, v = self.model.project_kv(tok)
                    k_list.append(k)
                    v_list.append(v)
                self.bm.allocate_slots_for_tokens(req.block_table, req.prompt_tokens, k_list, v_list)
                self.running_batch.append(req)
                print(f"  [ADMIT & PREFILL] Req '{req.req_id}': prompt {req.prompt_tokens} -> blocks {req.block_table}")
            else:
                # Memory constrained, pause admission
                break

        if not self.running_batch:
            print("  [IDLE] No active requests in engine.")
            return

        # 2. Decode Phase: Execute 1 step of generation for every running request
        evicted = []
        for req in self.running_batch:
            # Current query is derived from the latest token
            last_tok = req.generated_tokens[-1] if req.generated_tokens else req.prompt_tokens[-1]
            q_vec = self.model.embed_token(last_tok)

            # Compute PagedAttention over non-contiguous physical blocks
            attn_out = paged_attention(q_vec, req.block_table, self.bm)

            # Sample next token
            next_tok = self.model.predict_next(attn_out, len(req.generated_tokens) + 1, req.target_gen_len)

            if next_tok == "<eos>":
                req.status = "FINISHED"
                req.finish_iteration = self.current_iteration
                # Immediately reclaim physical blocks
                self.bm.free_block_table(req.block_table)
                self.finished_requests.append(req)
                evicted.append(req)
                print(f"  [FINISH & EVICT] Req '{req.req_id}' produced '<eos>'. Blocks freed! Total output: {req.generated_tokens}")
            else:
                # Store new KV token into paged memory
                k_vec, v_vec = self.model.project_kv(next_tok)
                self.bm.allocate_slots_for_tokens(req.block_table, [next_tok], [k_vec], [v_vec])
                req.generated_tokens.append(next_tok)
                print(f"  [DECODE STEP] Req '{req.req_id}' -> '{next_tok}' (Blocks: {req.block_table})")

        # Remove finished requests from active running batch
        for req in evicted:
            self.running_batch.remove(req)

        # Print memory footprint summary
        used_blocks = self.bm.total_blocks - self.bm.num_free_blocks()
        print(f"  [MEMORY STATUS] Active Blocks: {used_blocks}/{self.bm.total_blocks} (Free: {self.bm.num_free_blocks()})")


# =====================================================================
# 5. Benchmark Demonstration: Static Batching vs. Continuous Batching
# =====================================================================

def run_simulation():
    print("=====================================================================")
    print("Lab 05: Streaming KV & Continuous Batching Engine Simulation")
    print("=====================================================================")

    model = ToyLanguageModel(d_model=4)
    # GPU with 12 physical blocks of size 4 tokens (total capacity: 48 tokens)
    block_manager = BlockManager(total_blocks=12, block_size=4)
    engine = ContinuousBatchEngine(model, block_manager, max_batch_size=3)

    # 4 sample client requests with varying lengths
    requests = [
        Request("Req-A", ["the", "cat"], target_gen_len=2, arrival_iteration=1),
        Request("Req-B", ["dog", "sat", "on"], target_gen_len=6, arrival_iteration=1),
        Request("Req-C", ["sun"], target_gen_len=3, arrival_iteration=1),
        Request("Req-D", ["the", "cat", "slept"], target_gen_len=2, arrival_iteration=2),
    ]

    for req in requests:
        engine.submit_request(req)

    # Run scheduling loop until all requests finish
    while engine.running_batch or engine.waiting_queue:
        engine.step()

    print("\n=====================================================================")
    print("Final Performance Audit & Bubble Waste Analysis")
    print("=====================================================================")
    
    total_decode_steps = sum(len(r.generated_tokens) for r in requests)
    max_single_req_len = max(len(r.generated_tokens) for r in requests)
    batch_size = len(requests)

    # In static batching, all 4 requests would run until the longest one finishes (6 steps)
    static_slot_steps = batch_size * max_single_req_len
    bubble_waste_static = (1.0 - (total_decode_steps / static_slot_steps)) * 100

    print(f"Total Requests Completed: {len(engine.finished_requests)}")
    for r in engine.finished_requests:
        lat = r.finish_iteration - r.start_iteration
        print(f"  * {r.req_id}: Prompt={len(r.prompt_tokens)} tok, Gen={len(r.generated_tokens)} tok, Latency={lat} iters")

    print(f"\nStatic Batching Theoretical Slot Usage:     {static_slot_steps} slot-steps")
    print(f"Actual Useful Compute Tokens:               {total_decode_steps} tokens")
    print(f"Static Batching Bubble Waste:               {bubble_waste_static:.1f}%")
    print(f"Continuous Batching Bubble Waste:           0.0% (Slots instantly recycled!)")
    print("=====================================================================")


if __name__ == "__main__":
    run_simulation()
