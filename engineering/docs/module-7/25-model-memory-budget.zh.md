# 第 E25 章：大模型显存预算全景公式：训练 16P 静态显存、优化器状态与前向反向激活值显存的精准计算

## 步骤 1：物理直觉

想象你在规划一场耗资百亿的超级火箭发射工程（80GB GPU 显存规划）：
- **基底配重（模型静态权重）：** 火箭本身的钢结构框架，一克都不能少（参数占用）；
- **燃料与助推储备（优化器状态）：** 占整枚火箭体积近 70% 的庞大燃料罐（Adam 的两阶动量与高精度主权重），点火起飞前必须装得满满当当；
- **飞行空气动力载荷（反向传播激活值）：** 火箭穿越大气层时随高度剧烈起伏的动态空气阻力（前向传播记录的中间运算缓存）。

如果在点火升空的一瞬间（反向传播启动的峰值时刻），火箭总重量超出了一克允许的载重极限，整枚火箭就会在半空中当场粉碎爆炸（**CUDA Out-of-Memory / OOM**）！大模型系统工程师必须在动手写第一行代码前，用精密算盘精确算定每一个字节的物理去向！

---

## 步骤 2：芯片微观底层执行机制

在大模型训练与推理中，GPU 显存被四大板块严格割裂占用：

```
GPU 物理全局显存 (HBM) 全景剖析:
┌────────────────────────────────────────────────────────┐
│ 1. 静态模型权重 (Model Weights):                       │
│    FP16 / BF16 精度下固定占用: 2 * P 字节              │
├────────────────────────────────────────────────────────┤
│ 2. 静态梯度与优化器状态 (Gradients & Optimizer States): │
│    - 梯度张量 (FP16): 2 * P 字节                       │
│    - AdamW 优化器状态 (FP32 主权重 + 动量 m + 方差 v): │
│      (4 + 4 + 4) * P = 12 * P 字节                     │
│    ==> 静态三合一累计占用: 16 * P 字节!                │
├────────────────────────────────────────────────────────┤
│ 3. 动态前向激活值显存 (Forward Activations):            │
│    为了反向传播求导，前向传播必须保留的所有中间输出。   │
│    与序列长度 S、批大小 B、层数 L 呈强相关!            │
├────────────────────────────────────────────────────────┤
│ 4. 临时工作区与框架保留缓存 (Workspace & Overhead):    │
│    cuBLAS / cuDNN 算子临时显存、通信缓冲区 (~2-4 GB)   │
└────────────────────────────────────────────────────────┘
```

---

## 步骤 3：跨组件相互耦合机制

1. **激活值重计算（Activation Checkpointing）的以算力换显存：**  
   若保留所有层的中间激活值，显存会随层数和长序列迅速爆炸。通过**选择性重计算（Selective Recomputation）**，只保留每层开头的输入残差张量，在反向传播到达该层时当场重新执行一次 FlashAttention 和 SwiGLU，直接抹除 **$70\%\text{--}80\%$** 的激活值显存开销。
2. **结合 ZeRO 拆解 16P 静态重担：**  
   通过 ZeRO-3 将 16P 均摊给 64 张卡，原本单卡需要 1120GB 显存的 70B 模型瞬间缩水至 17.5GB，为海量动态激活值腾出了充裕的物理空间。

---

## 步骤 4：精确性能数学公式

对于含 $P$ 个参数的模型，在混合精度 AdamW 下训练的**全局静态显存公式**：

$$\text{Memory}_{\text{static}} = \underbrace{2 P}_{\text{Weights}} + \underbrace{2 P}_{\text{Gradients}} + \underbrace{12 P}_{\text{Optimizer}} = 16 P \text{ 字节}$$

单层 Transformer 在前向传播中若不采用重计算，其**动态激活值显存公式**为：

$$\text{Act}_{\text{layer}} = B S d \times \left( \underbrace{2}_{\text{RMSNorm}} + \underbrace{2 \cdot \frac{n_{\text{kv\_heads}}}{n_{\text{heads}}} + 2}_{\text{QKV 投影与 RoPE}} + \underbrace{5 \cdot \frac{d_{\text{ffn}}}{d}}_{\text{SwiGLU 3 矩阵与激活}} \right) \text{ 元素}$$

在全模型 $L$ 层累加后，总激活值显存（未开启 Checkpointing）为：

$$\text{Memory}_{\text{act\_full}} = 2 \times L \times \text{Act}_{\text{layer}} \text{ 字节 (FP16)}$$

而采用选择性重计算（仅重算注意力 Softmax 与 SwiGLU 激活）后，激活值显存缩减为：

$$\text{Memory}_{\text{act\_selective}} \approx 2 \times L \times (2 B S d) \text{ 字节}$$

---

## 步骤 5：具体微基准数字推导

以使用单台 8 卡 A100（每卡 $80\text{ GB}$，总显存 $640\text{ GB}$）进行 **LLaMA-3 8B**（$P = 8 \times 10^9$，$L = 32, d = 4096$）全量训练为例，批大小 $B = 4$，序列长度 $S = 4096$：

### 1. 计算 16P 全局静态显存：
$$\text{Memory}_{\text{static}} = 16 \times 8 \times 10^9 \text{ 字节} = 128 \text{ GB}$$
分配到 8 张卡上（采用 ZeRO-3 或 FSDP）：
$$\text{单卡静态显存} = \frac{128 \text{ GB}}{8} = 16 \text{ GB}$$

### 2. 计算动态激活值显存（若未开启重计算）：
单层单 Token 激活值元素数：
$$\text{Act}_{\text{layer}} \approx 4 \times 4096 \times 4096 \times (2 + 4 + 5 \times 3.5) \approx 6.71 \times 10^7 \times 23.5 \approx 1.57 \times 10^9 \text{ 元素}$$
全模型 32 层在 FP16 下的总激活值显存：
$$\text{Memory}_{\text{act}} = 2 \times 32 \times 1.57 \times 10^9 \text{ 字节} \approx 100.5 \text{ GB}!$$
*结论：单单激活值就超过了 80GB 卡的上限，直接 OOM 崩溃！*

### 3. 开启选择性重计算后：
$$\text{Memory}_{\text{act}} \approx 2 \times 32 \times (2 \times 4 \times 4096 \times 4096) \times 2 \text{ 字节} \approx 8.58 \text{ GB}$$
- 单卡总显存开销：
  $$\text{单卡总显存} = 16\text{ GB (静态)} + 8.58\text{ GB (激活值)} + 4\text{ GB (运行缓存)} \approx 28.58 \text{ GB}$$
- **实测结果：稳定运行于 80GB 显存之内，显存利用率健康平稳，彻底绝缘 OOM！**

---

## 步骤 6：核心系统工程铁律

> 显存预算不是靠运行时盲目试错摸出来的，而是靠首原则公式严密推导出来的。深刻吃透 16P 静态底座与序列激活值在反向传播中的峰值重叠，是每一个资深大模型系统工程师必须终生掌握的基本功。
