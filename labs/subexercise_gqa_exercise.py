"""
Hands-on Subexercise (Guided Exercise): Grouped-Query Attention (GQA)
Pure Python Implementation (Zero External Dependencies)

State-of-the-Art (SOTA) Context:
Modern frontier architectures (LLaMA-3, Mistral, Gemma 2, DeepSeek-V2/V3) universally
adopt Grouped-Query Attention (GQA) over traditional Multi-Head Attention (MHA).
While MHA allocates equal numbers of Query and Key-Value heads (H_Q = H_KV), GQA
partitions H_Q query heads into G groups that share H_KV key-value heads.
This preserves expressive multi-perspective routing while slashing KV Cache RAM
and memory bandwidth by 4x to 8x during long-context (128k) generation.

Instructions:
1. Complete the 4 TODO blocks marked below.
2. Run this file with `python3 labs/subexercise_gqa_exercise.py`.
3. The built-in automated test suite will verify each function step-by-step.
"""

import math
import random


# ---------------------------------------------------------------------
# Helper Functions (Provided)
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
# TODO 1: Group Index Mapping
# ---------------------------------------------------------------------
def get_kv_head_index(q_head_idx, group_size):
    """Maps a Query head index to its shared Key-Value head index.
    
    Mathematical Formulation:
      In GQA, H_Q query heads are partitioned into H_KV groups.
      Each group contains G = H_Q // H_KV query heads.
      Query head h belongs to group:
        kv_idx = floor(h / G) = h // group_size
        
      Example: If H_Q = 8 and H_KV = 2, group_size = 4:
        Query heads 0, 1, 2, 3 -> share KV head 0
        Query heads 4, 5, 6, 7 -> share KV head 1
        
    Args:
        q_head_idx (int): Index of query head in [0, H_Q - 1]
        group_size (int): Number of query heads per KV head (H_Q // H_KV)
        
    Returns:
        int: The shared KV head index in [0, H_KV - 1]
    """
    # -----------------------------------------------------------------
    # TODO 1: Implement get_kv_head_index
    # -----------------------------------------------------------------
    raise NotImplementedError("TODO 1: Implement get_kv_head_index(q_head_idx, group_size)")


# ---------------------------------------------------------------------
# TODO 2: Head Slicing
# ---------------------------------------------------------------------
def slice_head(tensor_2d, head_idx, d_head):
    """Extracts a [T x d_head] slice for head_idx from a combined 2D tensor [T x (num_heads * d_head)].
    
    Mathematical Formulation:
      The projection matrix projects all heads simultaneously into a single 
      contiguous row of width (num_heads * d_head).
      For head index h, the feature slice occupies column indices:
        start_col = h * d_head
        end_col   = (h + 1) * d_head
        
    Args:
        tensor_2d (list of lists): 2D matrix of shape [T x (num_heads * d_head)]
        head_idx (int): Which head to extract (0-indexed)
        d_head (int): Subspace dimension per head
        
    Returns:
        list of lists: 2D matrix of shape [T x d_head]
    """
    # -----------------------------------------------------------------
    # TODO 2: Implement slice_head
    # -----------------------------------------------------------------
    raise NotImplementedError("TODO 2: Implement slice_head(tensor_2d, head_idx, d_head)")


# ---------------------------------------------------------------------
# TODO 3: Single-Head Causal Attention
# ---------------------------------------------------------------------
def single_head_causal_attention(Q_h, K_k, V_k, scale):
    """Computes causal masked attention for a single Query head paired with its shared Key and Value head.
    
    Mathematical Formulation:
      1. Raw score matrix S = (Q_h @ K_k^T) * scale   (shape [T x T])
      2. Causal masking: For any j > i, set S[i][j] = -1e9
      3. Attention weights: A = softmax(S, dim=-1)   (row-wise softmax)
      4. Output context: O_h = A @ V_k               (shape [T x d_head])
      
    Args:
        Q_h: [T x d_head] Query slice for head h
        K_k: [T x d_head] Shared Key slice for group k
        V_k: [T x d_head] Shared Value slice for group k
        scale (float): Scaling factor 1.0 / sqrt(d_head)
        
    Returns:
        list of lists: [T x d_head] attended context vectors
    """
    # -----------------------------------------------------------------
    # TODO 3: Implement single_head_causal_attention
    # -----------------------------------------------------------------
    raise NotImplementedError("TODO 3: Implement single_head_causal_attention(Q_h, K_k, V_k, scale)")


# ---------------------------------------------------------------------
# TODO 4: Full Grouped-Query Attention Pipeline
# ---------------------------------------------------------------------
def grouped_query_attention(X, W_q, W_k, W_v, W_o, H_q, H_kv, d_head):
    """Executes full Grouped-Query Attention (GQA).
    
    Mathematical Pipeline:
      1. Linear projections:
           Q = X @ W_q   (shape [T x (H_q * d_head)])
           K = X @ W_k   (shape [T x (H_kv * d_head)])
           V = X @ W_v   (shape [T x (H_kv * d_head)])
      2. For each query head h in 0 .. H_q - 1:
           a. Determine shared KV index: kv_idx = get_kv_head_index(h, group_size)
           b. Slice Q_h = slice_head(Q, h, d_head)
           c. Slice K_k = slice_head(K, kv_idx, d_head)
           d. Slice V_k = slice_head(V, kv_idx, d_head)
           e. Compute O_h = single_head_causal_attention(Q_h, K_k, V_k, scale)
      3. Concatenate all H_q head outputs horizontally into O_concat of shape [T x (H_q * d_head)].
      4. Apply output projection: Y = O_concat @ W_o (shape [T x d_model]).
      
    Args:
        X: [T x d_model] Input sequence
        W_q: [d_model x (H_q * d_head)] Query weights
        W_k: [d_model x (H_kv * d_head)] Key weights
        W_v: [d_model x (H_kv * d_head)] Value weights
        W_o: [(H_q * d_head) x d_model] Output projection weights
        H_q (int): Number of Query heads
        H_kv (int): Number of Key-Value heads (must divide H_q)
        d_head (int): Dimension of each subspace head
        
    Returns:
        list of lists: [T x d_model] multi-perspective contextualized representations
    """
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
    group_size = H_q // H_kv
    scale = 1.0 / math.sqrt(d_head)
    T = len(X)
    
    # -----------------------------------------------------------------
    # TODO 4: Implement grouped_query_attention
    # -----------------------------------------------------------------
    raise NotImplementedError("TODO 4: Implement grouped_query_attention(...)")


# ---------------------------------------------------------------------
# Automated Unit Test Suite
# ---------------------------------------------------------------------
def run_unit_tests():
    print("=" * 65)
    print("Running Automated Unit Tests for Grouped-Query Attention (GQA)...")
    print("=" * 65)
    
    # Test 1: get_kv_head_index
    try:
        assert get_kv_head_index(0, 4) == 0
        assert get_kv_head_index(3, 4) == 0
        assert get_kv_head_index(4, 4) == 1
        assert get_kv_head_index(7, 4) == 1
        print("[PASS] Step 1: get_kv_head_index correctly maps query heads to shared KV groups.")
    except NotImplementedError:
        print("[TODO] Step 1: get_kv_head_index not implemented yet.")
        return False
    except AssertionError as e:
        print(f"[FAIL] Step 1: get_kv_head_index returned incorrect group index: {e}")
        return False
        
    # Test 2: slice_head
    try:
        sample_tensor = [
            [10, 11, 20, 21, 30, 31],
            [12, 13, 22, 23, 32, 33]
        ]
        # 3 heads of dimension 2
        h0 = slice_head(sample_tensor, 0, 2)
        h1 = slice_head(sample_tensor, 1, 2)
        h2 = slice_head(sample_tensor, 2, 2)
        assert h0 == [[10, 11], [12, 13]]
        assert h1 == [[20, 21], [22, 23]]
        assert h2 == [[30, 31], [32, 33]]
        print("[PASS] Step 2: slice_head correctly extracts contiguous head subspaces.")
    except NotImplementedError:
        print("[TODO] Step 2: slice_head not implemented yet.")
        return False
    except AssertionError as e:
        print(f"[FAIL] Step 2: slice_head returned incorrect tensor slices: {e}")
        return False

    # Test 3: single_head_causal_attention
    try:
        Q_test = [[1.0, 0.0], [0.0, 1.0]]
        K_test = [[1.0, 0.0], [0.0, 1.0]]
        V_test = [[2.0, 3.0], [4.0, 5.0]]
        scale_test = 1.0
        O_test = single_head_causal_attention(Q_test, K_test, V_test, scale_test)
        
        # Token 0 can only attend to Token 0 (causal mask): weight on token 0 is 1.0
        assert abs(O_test[0][0] - 2.0) < 1e-4
        assert abs(O_test[0][1] - 3.0) < 1e-4
        print("[PASS] Step 3: single_head_causal_attention correctly applies scaling, causal mask, and softmax.")
    except NotImplementedError:
        print("[TODO] Step 3: single_head_causal_attention not implemented yet.")
        return False
    except AssertionError as e:
        print(f"[FAIL] Step 3: single_head_causal_attention output mismatch: {e}")
        return False

    # Test 4: grouped_query_attention
    try:
        random.seed(42)
        T_test = 2
        d_model_test = 4
        d_head_test = 2
        H_q_test = 2
        H_kv_test = 1  # 2 query heads share 1 KV head (G = 2)
        
        X_test = [[1.0, 0.5, 0.0, 1.0], [0.2, 0.8, 1.0, 0.0]]
        W_q_test = [[0.1] * (H_q_test * d_head_test) for _ in range(d_model_test)]
        W_k_test = [[0.1] * (H_kv_test * d_head_test) for _ in range(d_model_test)]
        W_v_test = [[0.1] * (H_kv_test * d_head_test) for _ in range(d_model_test)]
        W_o_test = [[0.1] * d_model_test for _ in range(H_q_test * d_head_test)]
        
        Y_test = grouped_query_attention(
            X_test, W_q_test, W_k_test, W_v_test, W_o_test,
            H_q_test, H_kv_test, d_head_test
        )
        assert len(Y_test) == T_test
        assert len(Y_test[0]) == d_model_test
        print("[PASS] Step 4: grouped_query_attention pipeline produces correct output dimensionality.")
    except NotImplementedError:
        print("[TODO] Step 4: grouped_query_attention not implemented yet.")
        return False
    except AssertionError as e:
        print(f"[FAIL] Step 4: grouped_query_attention output error: {e}")
        return False

    print("=" * 65)
    print("All unit tests PASSED! GQA Architecture successfully validated.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    if run_unit_tests():
        print("\nDemonstrating 128k Long-Context Memory Footprint Savings (LLaMA-3 70B):")
        layers = 80
        d_h = 128
        seq = 128000
        # 2 (Key and Value) * layers * seq * heads * d_h * 2 bytes
        gb_mha = (2 * layers * seq * 64 * d_h * 2) / (1024**3)
        gb_gqa = (2 * layers * seq * 8  * d_h * 2) / (1024**3)
        gb_mqa = (2 * layers * seq * 1  * d_h * 2) / (1024**3)
        
        print(f"  - Multi-Head Attention (MHA, 64 KV heads):  {gb_mha:6.2f} GB (Unviable: >80GB VRAM)")
        print(f"  - Grouped-Query Attention (GQA, 8 KV heads): {gb_gqa:6.2f} GB (8x reduction! SOTA standard)")
        print(f"  - Multi-Query Attention (MQA, 1 KV head):    {gb_mqa:6.2f} GB (64x reduction)")
    else:
        print("\nPlease complete the TODO blocks above and rerun this script!\n")
