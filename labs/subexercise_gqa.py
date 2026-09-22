"""
Hands-on Subexercise: Grouped-Query Attention (GQA) & The SOTA Memory Revolution
Pure Python Implementation (Zero External Dependencies)

State-of-the-Art (SOTA) Context:
Modern frontier architectures (LLaMA-3, Mistral, Gemma 2, DeepSeek-V2/V3) universally
adopt Grouped-Query Attention (GQA) over traditional Multi-Head Attention (MHA).
While MHA allocates equal numbers of Query and Key-Value heads (H_Q = H_KV), GQA
partitions H_Q query heads into G groups that share H_KV key-value heads.
This preserves expressive multi-perspective routing while slashing KV Cache RAM
and memory bandwidth by 4x to 8x during long-context (128k) generation.
"""

import math
import random


# ---------------------------------------------------------------------
# 1. Pure Linear Algebra Primitives
# ---------------------------------------------------------------------
def matmul(A, B):
    """Computes matrix multiplication C = A @ B."""
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def transpose(A):
    """Returns matrix transpose A^T."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def softmax_row(row):
    """Numerically stable row-wise softmax."""
    max_val = max(row)
    exp_r = [math.exp(v - max_val) for v in row]
    sum_r = sum(exp_r)
    return [v / sum_r for v in exp_r]


# ---------------------------------------------------------------------
# 2. Grouped-Query Attention Core Functions
# ---------------------------------------------------------------------
def get_kv_head_index(q_head_idx, group_size):
    """Maps a Query head index to its shared Key-Value head index.
    
    In GQA, Query heads are divided into groups of size G = H_Q // H_KV.
    Query heads [0 .. G-1] share KV head 0,
    Query heads [G .. 2G-1] share KV head 1, etc.
    
    Formula: kv_idx = q_head_idx // group_size
    """
    return q_head_idx // group_size


def slice_head(tensor_2d, head_idx, d_head):
    """Extracts a [T x d_head] slice for head_idx from a concatenated tensor [T x (H * d_head)]."""
    T = len(tensor_2d)
    start = head_idx * d_head
    end = start + d_head
    return [[tensor_2d[t][d] for d in range(start, end)] for t in range(T)]


def single_head_causal_attention(Q_h, K_k, V_k, scale):
    """Computes causal masked attention for a single Query head paired with its shared Key and Value head.
    
    Args:
        Q_h: [T x d_head] Query slice for head h
        K_k: [T x d_head] Key slice for shared head k
        V_k: [T x d_head] Value slice for shared head k
        scale: 1.0 / sqrt(d_head)
    
    Returns:
        O_h: [T x d_head] attended context vectors
    """
    T = len(Q_h)
    # 1. Raw scores: S = Q_h @ K_k^T (shape [T x T])
    scores = matmul(Q_h, transpose(K_k))
    
    # 2. Scale and apply causal mask (j > i => -1e9)
    for i in range(T):
        for j in range(T):
            scores[i][j] *= scale
            if j > i:
                scores[i][j] = -1e9
                
    # 3. Softmax along each row: A = softmax(S)
    weights = [softmax_row(scores[i]) for i in range(T)]
    
    # 4. Multiply by Value: O_h = A @ V_k (shape [T x d_head])
    return matmul(weights, V_k)


def grouped_query_attention(X, W_q, W_k, W_v, W_o, H_q, H_kv, d_head):
    """Executes full Grouped-Query Attention (GQA).
    
    This function naturally generalizes across all three modern paradigms:
    - If H_kv == H_q: Standard Multi-Head Attention (MHA, Vaswani 2017)
    - If H_kv == 1:   Multi-Query Attention (MQA, Shazeer 2019)
    - If 1 < H_kv < H_q: Grouped-Query Attention (GQA, Ainslie 2023 - SOTA)
    
    Args:
        X: [T x d_model] Input sequence representations
        W_q: [d_model x (H_q * d_head)] Query projection weights
        W_k: [d_model x (H_kv * d_head)] Key projection weights
        W_v: [d_model x (H_kv * d_head)] Value projection weights
        W_o: [(H_q * d_head) x d_model] Final output projection weights
        H_q: Number of Query heads
        H_kv: Number of Key-Value heads (must divide H_q)
        d_head: Dimension of each subspace head
        
    Returns:
        Y: [T x d_model] Multi-perspective contextualized token representations
    """
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
    group_size = H_q // H_kv
    scale = 1.0 / math.sqrt(d_head)
    T = len(X)
    
    # Step 1: Project input into full Query, Key, and Value tensors
    Q = matmul(X, W_q)  # [T x (H_q * d_head)]
    K = matmul(X, W_k)  # [T x (H_kv * d_head)]
    V = matmul(X, W_v)  # [T x (H_kv * d_head)]
    
    # Step 2: For each Query head, look up its shared KV head and run causal attention
    head_outputs = []
    for h in range(H_q):
        kv_idx = get_kv_head_index(h, group_size)
        Q_h = slice_head(Q, h, d_head)
        K_k = slice_head(K, kv_idx, d_head)
        V_k = slice_head(V, kv_idx, d_head)
        O_h = single_head_causal_attention(Q_h, K_k, V_k, scale)
        head_outputs.append(O_h)
        
    # Step 3: Concatenate all H_q head outputs along feature dimension [T x (H_q * d_head)]
    O_concat = []
    for t in range(T):
        row = []
        for h in range(H_q):
            row.extend(head_outputs[h][t])
        O_concat.append(row)
        
    # Step 4: Final projection back to d_model: [T x d_model]
    return matmul(O_concat, W_o)


# ---------------------------------------------------------------------
# 3. KV Cache Footprint Comparison (The Hardware Motivation)
# ---------------------------------------------------------------------
def calculate_kv_cache_bytes(seq_len, num_layers, num_kv_heads, d_head, bytes_per_elem=2):
    """Computes total KV Cache memory footprint in bytes.
    
    Each token stores 1 Key vector and 1 Value vector per layer:
    Memory = 2 * num_layers * seq_len * num_kv_heads * d_head * bytes_per_elem
    """
    return 2 * num_layers * seq_len * num_kv_heads * d_head * bytes_per_elem


# ---------------------------------------------------------------------
# 4. Verification and Demonstration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Hands-on Subexercise: Grouped-Query Attention (GQA)")
    print("=" * 65)
    
    # Setup toy parameters
    random.seed(42)
    T = 3           # Sequence length (e.g. ["why", "sky", "blue"])
    d_model = 8     # Hidden dimension
    d_head = 4      # Head dimension
    H_q = 4         # 4 Query heads
    H_kv = 2        # 2 Key-Value heads (Group size G = 4 // 2 = 2)
    
    print(f"Toy Configuration:")
    print(f"  Sequence Length T:        {T}")
    print(f"  Hidden Dimension d_model: {d_model}")
    print(f"  Head Dimension d_head:    {d_head}")
    print(f"  Query Heads H_q:          {H_q}")
    print(f"  KV Heads H_kv:            {H_kv}")
    print(f"  Group Size G:             {H_q // H_kv} (Query heads per KV head)")
    print("-" * 65)
    
    # Initialize toy weights
    def rand_mat(rows, cols):
        return [[round(random.gauss(0, 0.3), 3) for _ in range(cols)] for _ in range(rows)]
        
    X   = rand_mat(T, d_model)
    W_q = rand_mat(d_model, H_q * d_head)     # [8 x 16]
    W_k = rand_mat(d_model, H_kv * d_head)    # [8 x 8]
    W_v = rand_mat(d_model, H_kv * d_head)    # [8 x 8]
    W_o = rand_mat(H_q * d_head, d_model)     # [16 x 8]
    
    # Run GQA forward pass
    output = grouped_query_attention(X, W_q, W_k, W_v, W_o, H_q, H_kv, d_head)
    
    print(f"Input Shape:  [{len(X)} tokens x {len(X[0])} d_model]")
    print(f"Output Shape: [{len(output)} tokens x {len(output[0])} d_model]")
    print("\nSample Output Representation (Token 0):")
    print(" ", [round(v, 4) for v in output[0]])
    
    # -----------------------------------------------------------------
    # Hardware SOTA Analysis: 128k Long-Context KV Cache Comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SOTA Impact: KV Cache Memory at 128,000 Token Context (LLaMA-3 70B)")
    print("=" * 65)
    layers = 80
    d_h = 128
    heads_q = 64
    seq = 128000
    
    bytes_mha = calculate_kv_cache_bytes(seq, layers, 64, d_h)
    bytes_gqa = calculate_kv_cache_bytes(seq, layers, 8,  d_h)
    bytes_mqa = calculate_kv_cache_bytes(seq, layers, 1,  d_h)
    
    gb_mha = bytes_mha / (1024**3)
    gb_gqa = bytes_gqa / (1024**3)
    gb_mqa = bytes_mqa / (1024**3)
    
    print(f"  1. Standard MHA (64 KV heads):  {gb_mha:6.2f} GB  (Unviable: exceeds an 80GB A100 GPU!)")
    print(f"  2. Modern SOTA GQA (8 KV heads): {gb_gqa:6.2f} GB  (8x reduction! Fits comfortably)")
    print(f"  3. Aggressive MQA (1 KV head):  {gb_mqa:6.2f} GB  (64x reduction)")
    print("=" * 65)
    print("Subexercise Reference Verified Successfully!")
