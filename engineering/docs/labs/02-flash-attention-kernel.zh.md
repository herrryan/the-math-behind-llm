# 实践实验 2：从零手写极简 FlashAttention 算子：在线 Softmax 与分块平铺

同学，你好！在第一模块第 E04 章中，我们推导了 FlashAttention 的核心数学武器——**在线 Softmax 动态更新公式**。

当时老师就向你承诺过：“别着急，后面老师一定会带你亲手把这个公式写成可运行的算子内核！”

今天，承诺兑现的时刻到了！在本次实验中，我们将使用当今业界最优雅、最强大的 GPU 算子编程语言——**OpenAI Triton**，从零实现一个极简但功能完备的 FlashAttention 前向内核！

---

## 实验目标与产出

1. **掌握分块平铺思维：** 学习如何在 Triton 中将长序列切分为 $B_r$ 与 $B_c$ 的微小瓦片（Tiles）；
2. **手写在线 Softmax：** 在片上 SRAM 循环中，亲手实现局部最大值维护、数值稳定缩放与输出向量修正；
3. **精度与性能对比：** 编写测试用例，验证手写内核与 PyTorch 原生标准实现的数学等价性（误差小于 $10^{-3}$）。

---

## 步骤 1：Triton 内核架构设计

我们把计算分为外层块和内层块：
- **外层块（沿 Query 序列切分）：** 每个 Triton 程序实例（Program Instance）负责一小块 Query 瓦片（行块尺寸 $B_r = 64$）；
- **内层循环（沿 Key/Value 序列平铺）：** 该实例在片上维持循环，每次加载一块 $B_c = 64$ 的 Key 和 Value，更新片上的中间累加器。

---

## 步骤 2：核心 Triton 内核代码编写

来，跟着老师看懂这段充满数学与工程美感的 Triton 内核：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # 获取当前线程块负责的批次、头数、以及 Query 块索引
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # 偏移基准指针
    q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh + start_m * BLOCK_M * stride_om

    # 生成局部网格索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # 加载当前负责的 Q 块至片上寄存器/SRAM (尺寸: [BLOCK_M, BLOCK_D])
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 初始化在线 Softmax 状态标量 (在寄存器中维护!)
    # m_i 维护行最大值，初值设为负无穷
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i 维护分母指数和，初值设为 0
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # acc 维护最终累加输出 [BLOCK_M, BLOCK_D]
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 缩放因子 1 / sqrt(d)
    scale = 1.0 / (BLOCK_D ** 0.5)
    q = (q * scale).to(tl.float16)

    # 沿 Key 和 Value 序列进行内层分块循环
    for start_n in range(0, N_CTX, BLOCK_N):
        curr_n = start_n + offs_n

        # 1. 加载当前的 K 块与 V 块至片上
        k_ptrs = K + k_offset + curr_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptrs = V + v_offset + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=curr_n[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=curr_n[:, None] < N_CTX, other=0.0)

        # 2. 计算当前分块注意力得分: S_ij = Q_i @ K_j^T (纯片上做功!)
        qk = tl.dot(q, k) # [BLOCK_M, BLOCK_N]

        # 3. 执行在线 Softmax 数学更新!
        # 计算当前块的局部最大值
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # 稳定化指数
        p = tl.exp(qk - m_ij[:, None])
        # 计算校正缩放因子 alpha = exp(m_old - m_new)
        alpha = tl.exp(m_i - m_ij)

        # 校正历史分母并累加当前新分母
        l_i = l_i * alpha + tl.sum(p, 1)

        # 校正历史累加输出 acc，并融入当前新块的贡献: acc = acc * alpha + P @ V
        p = p.to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # 更新行最大值状态
        m_i = m_ij

    # 4. 循环结束，全序列遍历完毕，乘以最终总分母的倒数: Out = acc / l_i
    acc = acc / l_i[:, None]

    # 将最终无缝拼接的输出写回全局显存 HBM (仅此一次回写!)
    out_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)
```

---

## 步骤 3：封装 Python 接口与正确性检验

```python
def minimal_flash_attention(q, k, v):
    # 输入形状: [Z, H, N_CTX, D]
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D
    )
    return out

# 精度对齐验证
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        Z, H, N_CTX, D = 2, 4, 1024, 64
        q = torch.randn(Z, H, N_CTX, D, dtype=torch.float16, device=device)
        k = torch.randn(Z, H, N_CTX, D, dtype=torch.float16, device=device)
        v = torch.randn(Z, H, N_CTX, D, dtype=torch.float16, device=device)

        # 运行我们手写的极简 FlashAttention
        out_custom = minimal_flash_attention(q, k, v)

        # 运行原生 PyTorch 标准注意力作为基准真值
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        p = torch.softmax(scores.float(), dim=-1).to(torch.float16)
        out_ref = torch.matmul(p, v)

        # 计算最大绝对误差
        diff = torch.max(torch.abs(out_custom - out_ref)).item()
        print(f"验证通过！手写内核与 PyTorch 基准最大绝对误差: {diff:.6f}")
        assert diff < 1e-2, "误差过大，请检查在线 Softmax 数学逻辑！"
```

---

## 老师点评与课后思考题

1. **观察神奇之处：** 请注意我们写的内核，从头到尾**完全没有在 HBM 中分配过任何 $[N_{\text{ctx}}, N_{\text{ctx}}]$ 大小的注意力得分矩阵**！所有中间计算完全在片上几十 KB 的极速空间里消化完毕！
2. **课后挑战：** 尝试在此基础上增加**因果掩码（Causal Mask）**逻辑，想一想：如何跳过完全位于右上角的无效计算块？
