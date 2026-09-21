# 第 E08 章：前馈神经网络与 SwiGLU 物理实现：三矩阵投影结构、激活值显存激增与双矩阵熔合优化

## 步骤 1：物理直觉

想象厨房的一道精细调味工序：
- **传统做法（ReLU / GELU）：** 原料推入搅拌机（矩阵 1），机器转一圈完成调味，接着直接推入下一道热压成型机（矩阵 2）；
- **现代 SwiGLU 做法：** 追求极致绝妙的口感层次，厨房增设了三台并行工作台：
  - 机器 A 切出蔬菜丝（门控投影 $W_{\text{gate}}$）；
  - 机器 B 调配秘制浓酱（上升投影 $W_{\text{up}}$）；
  - 机器 C 是一个高速空气阀门，将蔬菜丝与浓酱在空中猛烈碰撞混合（Swish 门控相乘）；
  - 最后，把混合物送入收拢成型机（下降投影 $W_{\text{down}}$）。

风味（模型表达力）得到了惊人的提升，但厨师的操作台（片上显存）必须在同一瞬间同时摊开两倍以上的半成品配料，稍不留神就会把灶台塞满！

---

## 步骤 2：芯片微观底层执行机制

在现代主流大模型（LLaMA, Mistral, Qwen）中，传统 FFN 已被 **SwiGLU（Swish-Gated Linear Unit）** 完全取代：

$$
\text{SwiGLU}(\mathbf{x}) = \left( \text{Swish}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}}) \right) \mathbf{W}_{\text{down}}
$$

```
SwiGLU 三矩阵结构与计算流:
输入 X [B*S, d_model]
        │
        ├── 投影 1: X @ W_gate  ──► 中间激活 A [B*S, d_ffn]
        │                                  │ (经过 SiLU 激活)
        │                                  ▼
        ├── 投影 2: X @ W_up    ──► 中间激活 B [B*S, d_ffn] ──► 元素级相乘 (A * B)
        │                                                           │
        │                                                           ▼
        └── 投影 3: (A * B) @ W_down ────────────────────────► 最终输出 Y [B*S, d_model]
```

### 显存与计算的双重冲击：
1. **参数量与计算量大户：** 中间维度通常取 $d_{\text{ffn}} \approx \frac{8}{3} d_{\text{model}}$。全模型的参数与浮点运算量中，**近 $66\%$ 集中在 FFN 层**，注意力和规范化仅占 $34\%$！
2. **激活值显存激增：** 在反向传播训练中，为了对 $W_{\text{gate}}$ 和 $W_{\text{up}}$ 求偏导，系统必须在 HBM 中同时保存 $A$ 和 $B$ 两个巨型张量，激活值显存比传统 GELU 陡增 $50\%$！

---

## 步骤 3：跨组件相互耦合机制

1. **双矩阵水平拼接熔合（GEMM Concatenation Fusion）：**  
   若分别启动两个独立的 CUDA 核函数计算 $\mathbf{x}\mathbf{W}_{\text{gate}}$ 与 $\mathbf{x}\mathbf{W}_{\text{up}}$，将产生两次独立的内核启动开销与输入读取。工业级优化将两矩阵合并为一个大矩阵 $\mathbf{W}_{\text{fused}} = [\mathbf{W}_{\text{gate}} \mid \mathbf{W}_{\text{up}}] \in \mathbb{R}^{d \times 2d_{\text{ffn}}}$，**一次 GEMM 调用直接输出拼接张量**，Tensor Core 效率大幅跃升！
2. **选择性重计算（Selective Recomputation）：**  
   在显存紧张的训练中，反向传播不保存元素级相乘的中间张量，而是在求导时于片上 SRAM 中瞬时重算，用微小的算力代价置换出数十吉字节的物理显存。

---

## 步骤 4：精确性能数学公式

单层 SwiGLU 的理论浮点运算量为：

$$\text{FLOPs}_{\text{SwiGLU}} = 2 \times (B \cdot S) \times d_{\text{model}} \times d_{\text{ffn}} \times 3 \text{ FLOPs}$$

代入标准比例 $d_{\text{ffn}} = \frac{8}{3} d_{\text{model}}$：

$$\text{FLOPs}_{\text{SwiGLU}} = 6 \times (B \cdot S) \times d_{\text{model}} \times \left(\frac{8}{3} d_{\text{model}}\right) = 16 (B \cdot S) d_{\text{model}}^2$$

而在前向传播中，未优化前需要缓存的中间激活值显存为：

$$\text{Memory}_{\text{act\_ffn}} = 2 \times (B \cdot S) \times d_{\text{ffn}} \times \text{sizeof(dtype)} \quad (\text{同时保存 Gate 与 Up 输出})$$

---

## 步骤 5：具体微基准数字推导

以 LLaMA-3 70B（$d_{\text{model}} = 8192, d_{\text{ffn}} = 28672$），批大小 $B=1$、序列长度 $S=4096$ 在单层 FFN 上的物理消耗：

1. **单层参数量与显存（FP16）：**
   $$P_{\text{layer}} = 3 \times (8192 \times 28672) \approx 7.046 \times 10^8 \text{ 参数} \approx 1.41 \text{ GB}$$
   全模型 80 层 FFN 参数显存累计高达 **$112.8\text{ GB}$**（占全模型 140GB 权重的整整 $80.5\%$）！
2. **单层单步前向浮点运算量：**
   $$F = 6 \times 4096 \times 8192 \times 28672 \approx 5.77 \times 10^{12} \text{ FLOPs} = 5.77 \text{ TFLOPs}$$
3. **单层前向激活值显存开销：**
   $$\text{Act} = 2 \times 4096 \times 28672 \times 2 \text{ 字节} \approx 469.7 \text{ MB}$$
   全模型 80 层累计激活值达 **$37.6\text{ GB}$**！
- **结论：** 通过将 Gate 与 Up 投影熔合成单次 GEMM，并在反向传播中开启选择性重计算，单层前向时延缩短了 **$22\%$**，同时释放出近 **$30\text{ GB}$** 的宝贵训练显存！

---

## 步骤 6：核心系统工程铁律

> 前馈神经网络是大模型参数与计算的绝对重镇。将 Gate 和 Up 投影物理拼接为单一大矩阵以拉长 GEMM 维度，并使用片上核函数即时消化非线性激活，是榨干现代 GPU 计算密度的第一必修课。
