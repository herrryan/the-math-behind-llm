"""
Stage 5: The Streaming KV & Continuous Batching Engine - Guided Exercise
The Math Behind Large Language Models - Lab 05

Instructions:
-------------
In this hands-on lab, you will build the core data structures and scheduling algorithms
that power modern high-throughput LLM serving systems (such as vLLM, TensorRT-LLM, and TGI)
using 100% pure Python (no PyTorch, no vLLM, no NumPy).

All scaffolding (model simulation, request definitions, test harnesses) has been provided.
Your mission is to fill in the core memory management and scheduling logic marked with TODO.

To guide your implementation, every TODO block includes:
  1. High-Level Intuition & Requirements
  2. Data Structures & Block Table Mapping
  3. Step-by-Step Python Hints

When you run this script:
  $ python3 streaming_engine_exercise.py
It will test your implementation step-by-step and run a real continuous batching simulation.
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

    # -----------------------------------------------------------------
    # TODO 1: Implement allocate_block and free_block
    # -----------------------------------------------------------------
    def allocate_block(self) -> int:
        """
        Pops the next available physical block ID from self.free_list.
        Initializes the block's ref_count to 1 and clears any stale data.
        Raises MemoryError if no blocks are available in self.free_list.
        """
        # --- YOUR CODE HERE ---
        if not self.free_list:
            raise MemoryError("Out of GPU Memory: No physical blocks available!")
        bid = self.free_list.popleft()
        self.blocks[bid].clear()
        self.blocks[bid].ref_count = 1
        return bid
        # ----------------------

    def free_block(self, bid: int):
        """
        Decrements the block's ref_count by 1.
        If ref_count reaches 0, clears the block and returns bid to self.free_list.
        """
        # --- YOUR CODE HERE ---
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count == 0:
            block.clear()
            self.free_list.append(bid)
        # ----------------------

    def free_block_table(self, block_table: list):
        """Frees all physical blocks mapped to a sequence's logical block table."""
        for bid in block_table:
            self.free_block(bid)
        block_table.clear()

    # -----------------------------------------------------------------
    # TODO 2: Implement allocate_slots_for_tokens
    # -----------------------------------------------------------------
    def allocate_slots_for_tokens(self, block_table: list, tokens: list, k_vecs: list, v_vecs: list):
        """
        Appends tokens, keys, and values to a sequence via its block table.
        
        Logic:
        For each (token, k_vec, v_vec):
          1. Check if block_table is empty, or if the current last block is full (is_full == True).
             If so, call self.allocate_block() and append the new block_id to block_table.
          2. Append token, k_vec, v_vec to the last block in block_table.
        """
        # --- YOUR CODE HERE ---
        for token, k_vec, v_vec in zip(tokens, k_vecs, v_vecs):
            if not block_table or self.blocks[block_table[-1]].is_full:
                new_bid = self.allocate_block()
                block_table.append(new_bid)
            
            last_block = self.blocks[block_table[-1]]
            last_block.append(token, k_vec, v_vec)
        # ----------------------


# =====================================================================
# 2. PagedAttention Kernel (Chapter 22 & 23)
# =====================================================================

# ---------------------------------------------------------------------
# TODO 3: Implement paged_attention
# ---------------------------------------------------------------------
def paged_attention(query: list, block_table: list, block_manager: BlockManager) -> list:
    """
    Computes Scaled Dot-Product Attention over scattered physical blocks.
    
    Formula:
      Attention(Q, K, V) = Softmax(Q . K^T / sqrt(d_k)) . V
    
    Steps:
      1. Iterate through every block_id in block_table.
      2. For every k_vec and v_vec in that physical block:
         Compute dot product score: sum(q * k for q, k in zip(query, k_vec)) / sqrt(d_k)
      3. Compute numerically safe Softmax over all scores:
         max_s = max(scores)
         exp_s = [exp(s - max_s) for s in scores]
         weights = [e / sum(exp_s) for e in exp_s]
      4. Compute weighted sum of values:
         output[i] = sum(w * v[i] for w, v in zip(weights, all_values))
    """
    d_k = len(query)
    scores = []
    all_values = []

    # --- YOUR CODE HERE ---
    for bid in block_table:
        block = block_manager.blocks[bid]
        for k_vec, v_vec in zip(block.keys, block.values):
            dot = sum(q * k for q, k in zip(query, k_vec)) / math.sqrt(d_k)
            scores.append(dot)
            all_values.append(v_vec)

    if not scores:
        return [0.0] * d_k

    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    weights = [e / sum_exp for e in exp_scores]

    output = [0.0] * len(all_values[0])
    for w, val in zip(weights, all_values):
        for i in range(len(output)):
            output[i] += w * val[i]

    return output
    # ----------------------


# =====================================================================
# 3. Model Simulation: Synthetic Embedding & KV Projections
# =====================================================================

class ToyLanguageModel:
    """Lightweight deterministic model generating reproducible tokens and KV states."""
    def __init__(self, d_model: int = 4, vocab: list = None):
        self.d_model = d_model
        self.vocab = vocab or ["the", "cat", "sat", "on", "mat", "dog", "slept", "sun", "<eos>"]
        random.seed(42)
        self.W_k = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]

    def embed_token(self, token: str) -> list:
        seed = sum(ord(c) for c in token)
        random.seed(seed)
        return [random.uniform(-1.0, 1.0) for _ in range(self.d_model)]

    def project_kv(self, token: str):
        x = self.embed_token(token)
        k = [sum(x[j] * self.W_k[j][i] for j in range(self.d_model)) for i in range(self.d_model)]
        v = [sum(x[j] * self.W_v[j][i] for j in range(self.d_model)) for i in range(self.d_model)]
        return k, v

    def predict_next(self, attn_out: list, current_gen_len: int, target_len: int) -> str:
        if current_gen_len > target_len:
            return "<eos>"
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
        self.status = "WAITING"
        self.start_iteration = None
        self.finish_iteration = None


class ContinuousBatchEngine:
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

    # -----------------------------------------------------------------
    # TODO 4: Implement step() scheduling loop
    # -----------------------------------------------------------------
    def step(self):
        """
        Executes one discrete iteration step:
          1. Dynamic Admission: While len(running_batch) < max_batch_size and waiting_queue:
             - Check if free blocks >= needed prompt blocks.
             - If so: pop request, set status="RUNNING", prefill prompt KV into block_table, add to running_batch.
          2. Decode Step: For each request in running_batch:
             - Compute query from last token.
             - Run paged_attention.
             - Predict next token.
             - If next token is '<eos>': set status="FINISHED", free block_table, move to finished_requests.
             - Else: allocate 1 slot for next token, append to generated_tokens.
        """
        self.current_iteration += 1

        # --- YOUR CODE HERE ---
        # 1. Dynamic Admission
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            next_req = self.waiting_queue[0]
            needed_blocks = math.ceil(len(next_req.prompt_tokens) / self.bm.block_size)
            if self.bm.num_free_blocks() >= needed_blocks + 1:
                req = self.waiting_queue.popleft()
                req.status = "RUNNING"
                req.start_iteration = self.current_iteration
                
                k_list, v_list = [], []
                for tok in req.prompt_tokens:
                    k, v = self.model.project_kv(tok)
                    k_list.append(k)
                    v_list.append(v)
                self.bm.allocate_slots_for_tokens(req.block_table, req.prompt_tokens, k_list, v_list)
                self.running_batch.append(req)
            else:
                break

        if not self.running_batch:
            return

        # 2. Decode Phase
        evicted = []
        for req in self.running_batch:
            last_tok = req.generated_tokens[-1] if req.generated_tokens else req.prompt_tokens[-1]
            q_vec = self.model.embed_token(last_tok)
            attn_out = paged_attention(q_vec, req.block_table, self.bm)
            next_tok = self.model.predict_next(attn_out, len(req.generated_tokens) + 1, req.target_gen_len)

            if next_tok == "<eos>":
                req.status = "FINISHED"
                req.finish_iteration = self.current_iteration
                self.bm.free_block_table(req.block_table)
                self.finished_requests.append(req)
                evicted.append(req)
            else:
                k_vec, v_vec = self.model.project_kv(next_tok)
                self.bm.allocate_slots_for_tokens(req.block_table, [next_tok], [k_vec], [v_vec])
                req.generated_tokens.append(next_tok)

        for req in evicted:
            self.running_batch.remove(req)
        # ----------------------


# =====================================================================
# 5. Automated Verification Test Suite
# =====================================================================

def test_block_manager():
    print("[TEST 1/3] Testing BlockManager Allocation & Reclamation...")
    bm = BlockManager(total_blocks=4, block_size=2)
    assert bm.num_free_blocks() == 4, "Initial free blocks must match total!"

    b0 = bm.allocate_block()
    b1 = bm.allocate_block()
    assert bm.num_free_blocks() == 2, "Free blocks did not decrement correctly!"
    assert b0 == 0 and b1 == 1, "Block IDs must be allocated in order!"

    table = [b0, b1]
    bm.free_block_table(table)
    assert bm.num_free_blocks() == 4, "Free block table must return all blocks to pool!"
    assert len(table) == 0, "free_block_table should clear the input list!"
    print("  -> PASSED: BlockManager allocation and reclamation verified.")


def test_paged_attention():
    print("[TEST 2/3] Testing PagedAttention Kernel...")
    bm = BlockManager(total_blocks=4, block_size=2)
    table = []
    
    tokens = ["hello", "world", "!"]
    k_vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v_vecs = [[0.5, 0.5], [1.0, 2.0], [0.0, 3.0]]
    
    bm.allocate_slots_for_tokens(table, tokens, k_vecs, v_vecs)
    assert len(table) == 2, f"3 tokens in block_size 2 must span exactly 2 physical blocks! Got {len(table)}"
    
    q_vec = [1.0, 0.0]
    out = paged_attention(q_vec, table, bm)
    assert len(out) == 2, "Attention output dimension must match value dimension!"
    assert all(not math.isnan(x) for x in out), "Output must not contain NaNs!"
    
    bm.free_block_table(table)
    assert bm.num_free_blocks() == 4, "All blocks must be freed!"
    print("  -> PASSED: PagedAttention scatter-gather correctly computed.")


def test_continuous_batching():
    print("[TEST 3/3] Testing Continuous Batching Scheduler...")
    model = ToyLanguageModel(d_model=4)
    bm = BlockManager(total_blocks=8, block_size=4)
    engine = ContinuousBatchEngine(model, bm, max_batch_size=2)

    r1 = Request("Req1", ["the"], target_gen_len=1, arrival_iteration=1)
    r2 = Request("Req2", ["cat"], target_gen_len=3, arrival_iteration=1)
    r3 = Request("Req3", ["sat"], target_gen_len=1, arrival_iteration=2)

    engine.submit_request(r1)
    engine.submit_request(r2)
    engine.submit_request(r3)

    # Step 1: r1 and r2 admitted
    engine.step()
    assert len(engine.running_batch) == 2
    assert len(engine.waiting_queue) == 1

    # Step 2: r1 finishes and frees its slot
    engine.step()
    assert len(engine.finished_requests) == 1
    assert engine.finished_requests[0].req_id == "Req1"

    # Step 3: r3 is dynamically admitted into the vacant slot!
    engine.step()
    assert any(r.req_id == "Req3" for r in engine.running_batch), "Req3 must be dynamically admitted into vacant slot!"
    print("  -> PASSED: Continuous batching dynamic slot replacement verified.")


if __name__ == "__main__":
    print("=====================================================================")
    print("Running Lab 05 Guided Exercise Test Suite...")
    print("=====================================================================")
    test_block_manager()
    test_paged_attention()
    test_continuous_batching()
    print("=====================================================================")
    print("All unit tests PASSED! You have built a working Paged Continuous Engine!")
    print("=====================================================================")
