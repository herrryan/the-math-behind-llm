# 实验 E2：极简 FlashAttention 算子实现（Triton / C++）

## 实验目标与挑战

在第 E03 和 E04 章中，我们深入剖析了 FlashAttention 的数学原理与硬件设计。没有亲自实现过在线 Softmax（Online Softmax）分块平铺的工程师，永远无法真正体会片上共享内存（SRAM）调度的精妙。

**本实验的核心任务：** 使用现代 GPU 算子编程语言 **OpenAI Triton**（或原生 CUDA/C++），从零实现一个极简但功能完全正确的 **FlashAttention 前向传播算子**。
1. 将输入的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 张量在序列维度切分成适配 SRAM 大小的局部 Tile（例如 $64 \times 64$ 或 $128 \times 64$）；
2. 在 Triton 核函数中，维护两个正在迭代更新的 SRAM 累加向量：行最大值向量 $m_i$ 与归一化分母向量 $l_i$；
3. 实现标准的在线 Softmax 缩放校正逻辑，将局部乘加结果累加至最终输出 $\mathbf{O}_i$；
4. 验证算子输出与 PyTorch 原生 `F.scaled_dot_product_attention` 的数值绝对对齐（误差控制在 $10^{-3}$ 以内）；
5. 与未熔合的朴素 PyTorch 注意力对比在长序列（$S = 2048, 4096, 8192$）下的显存占用峰值与执行速度。

---

## 动手实践代码框架

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
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 识别当前线程块负责的批次、注意力头与序列行分块索引
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 初始化在线 Softmax 状态变量 (位于片上寄存器)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 核心任务: 编写内层循环，流式分块加载 K 和 V，就地完成在线累加更新!
```

---

## 思考与检验题
1. 为什么在线 Softmax 必须在 FP32 精度下维护累加状态 $m_i$ 和 $l_i$，即使输入和输出都是 FP16/BF16？
2. 尝试在核函数中加入因果掩码（Causal Mask），观察为什么可以直接整块跳过右上角无用的 Tile，从而使运行时间瞬间减半？
