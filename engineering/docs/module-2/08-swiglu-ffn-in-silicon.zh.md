# 第 E08 章：SwiGLU 与前馈网络在芯片上的执行：三矩阵投影与反向传播激活值治理

同学，你好！在前面我们探讨了注意力机制。很多同学以为注意力是大模型算力开销的最大头，但老师今天告诉你一个颠覆常识的事实：

**在整个 Transformer 模型中，真正吃掉全网 60% 到 66% 显存参数和前向浮点计算量的，不是注意力机制，而是紧跟其后的前馈网络（Feed-Forward Network，FFN）！**

而在现代前沿开源大模型（LLaMA 系列、DeepSeek、Qwen、Mistral）中，大家普遍摒弃了传统的 ReLU/GELU 激活，全员拥抱了一个名字有点拗口的结构——**SwiGLU（Swish Gated Linear Unit）**。

你可能会好奇：“老师，为什么大家都换成了 SwiGLU？多出来的那张投影矩阵，在显卡底层会带来怎样沉重的显存代价呢？”

今天，老师就带你深入前馈网络的芯片执行内幕！

---

## 步骤 1：物理直觉（进食双通道与门控水龙头）

为了让你彻底看清传统 FFN 与 SwiGLU 的区别，老师打个厨房水龙头的比方：

- **传统 FFN（经典单通道）：**  
  食材送进厨房后，只有一条传输带：先经过一个大功率挤压机（膨胀线性层），然后通过一个筛子（ReLU/GELU 激活函数过滤），最后再压平装盘（收缩线性层）。这个流程简单粗暴，但表达能力比较单一。
- **SwiGLU（双通道门控调节）：**  
  食材一进门，立刻被兵分两路送进**两条并行的流水线**：
  - **一路叫做 Up 投影（主食材管道）：** 负责把食材特征升维放大；
  - **另一路叫做 Gate 投影（水龙头门控管道）：** 它计算出一组平滑的 0 到 1 之间的旋转阀门开度（Swish 激活）；
  - **在十字路口两两相乘：** 门控管道的水龙头旋转，精准决定主食材管道的每一滴调料流出多少；
  - **最后汇流进 Down 投影：** 重新压回原维度输出。

你看，SwiGLU 的语言理解能力极其细腻强大，但代价也显而易见：**原本只要 2 个大矩阵，现在变成了 3 个大矩阵！**

原本只需维护一份半成品，现在在芯片里必须同时维护两份巨大的中间张量，并在内存里把它们点对点相乘！

---

## 步骤 2：芯片微观底层执行机制

我们把 SwiGLU 的数学公式写在黑板上：

$$\text{FFN}_{\text{SwiGLU}}(\mathbf{x}) = \left(\text{Swish}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}$$

其中：
- $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$
- $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$
- 现代通常设置中间隐藏层维度 $d_{\text{ffn}} \approx \frac{8}{3} d$（例如当 $d=4096$ 时，$d_{\text{ffn}} = 11008$ 或 $14336$）。

```
【朴素实现 vs 算子水平拼接熔合】

朴素执行路线 (严重浪费 HBM 带宽):
  x ──► GEMM 1 ──► 写入中间张量 A 到 HBM ──►┐
  x ──► GEMM 2 ──► 写入中间张量 B 到 HBM ──►┴─► 从 HBM 读出 A, B ──► 执行 Swish 并逐元素乘 ──► 写入 C 到 HBM ──► GEMM 3 ──► 输出

算子水平拼接熔合优化 (Horizontal Fusion):
把 W_gate 与 W_up 在内存中横向拼接为一个宽矩阵:
  W_fused = [W_gate | W_up] 尺寸为 [d x (2 * d_ffn)]

执行微观数据流:
  x ──► 单次超大 GEMM 冲压 ──► 在片上 SRAM/寄存器中直接拆分并执行 Swish(A) * B ──► 仅将最终相乘结果写出 ──► GEMM 3
```

### 为什么水平熔合如此重要？
1. **打满 Tensor Core 冲压机：** 单次大矩阵乘法可以让 GPU 的网格调度器分配更多的线程块（Thread Blocks），避免两次独立小矩阵带来的内核启动气泡。
2. **消灭两份巨量中间特征的显存往返：** 逐元素逐位相乘（$\odot$）直接在片上完成，彻底省去了将 $\text{Swish}(\mathbf{x}\mathbf{W}_{\text{gate}})$ 与 $\mathbf{x}\mathbf{W}_{\text{up}}$ 写入全局显存再重新读出的开销！

---

## 步骤 3：跨组件相互耦合机制

1. **占据全模型 66% 参数量的大头：**  
   每个 Transformer 层包含 1 个注意力模块和 1 个 FFN 模块。注意力层包含 $Q, K, V, O$ 四个 $d \times d$ 矩阵（共 $4d^2$ 参数）。而 SwiGLU 拥有 3 个 $d \times \frac{8}{3}d$ 矩阵（共 $3 \times \frac{8}{3}d^2 = 8d^2$ 参数）！**FFN 的参数量整整是注意力层的 2 倍！**
2. **反向传播显存激活值的巨大包袱：**  
   在模型训练反向传播求导时，为了求出输入 $\mathbf{x}$ 的梯度，必须保存前向传播时的中间激活值。如果不做激活值重计算（Activation Recomputation），前馈层将成为训练时爆显存的头号元凶。

---

## 步骤 4：精确性能数学公式

设序列长为 $S$，批大小为 $B$，隐藏维度为 $d$，中间膨胀维度为 $d_{\text{ffn}}$：

### 1. 前向浮点运算量（FLOPs）
- Gate 投影：$2 B S d d_{\text{ffn}}$
- Up 投影：$2 B S d d_{\text{ffn}}$
- Down 投影：$2 B S d_{\text{ffn}} d$
- 激活函数与相乘：约 $4 B S d_{\text{ffn}}$（相比矩阵乘法可忽略）

$$\text{FLOPs}_{\text{SwiGLU}} \approx 6 B S d d_{\text{ffn}}$$

代入常用的 $d_{\text{ffn}} = \frac{8}{3} d$：

$$\text{FLOPs}_{\text{SwiGLU}} \approx 6 B S d \left(\frac{8}{3} d\right) = 16 B S d^2$$

### 2. 对比注意力机制线性投影
注意力层 4 个矩阵投影的计算量为：$4 \times 2 B S d^2 = 8 B S d^2$。  
**SwiGLU 线性部分的算力开销恰好是注意力线性投影的整整 2 倍！**

---

## 步骤 5：具体微基准数字推导

我们以 **LLaMA-3 8B** 单层的前向计算为例，算一算具体的硬件账本：
- $d = 4096$
- $d_{\text{ffn}} = 14336$
- 批大小 $B = 1$，上下文预填长度 $S = 2048$
- 精度：FP16（每元素 2 字节）

### 1. 计算三个权重矩阵的显存静态体积
- $\mathbf{W}_{\text{gate}}$ 大小：$4096 \times 14336 \times 2 \approx 117.4 \text{ MB}$
- $\mathbf{W}_{\text{up}}$ 大小：$4096 \times 14336 \times 2 \approx 117.4 \text{ MB}$
- $\mathbf{W}_{\text{down}}$ 大小：$14336 \times 4096 \times 2 \approx 117.4 \text{ MB}$
- **单层 SwiGLU 权重总大小：** $352.2 \text{ MB}$（全模型 32 层仅 FFN 权重就占去 $11.27\text{ GB}$！）。

### 2. 计算前向计算总做功
$$\text{FLOPs} = 6 \times 1 \times 2048 \times 4096 \times 14336 \approx 7.219 \times 10^{11} \text{ FLOPs} \approx 722 \text{ GFLOPs}$$

### 3. 未熔合 vs 熔合的中间访存开销对比
- **未熔合状态：**  
  计算 Gate 得到中间张量（$2048 \times 14336 \times 2 \approx 58.7 \text{ MB}$），写回 HBM；  
  计算 Up 得到中间张量（$58.7 \text{ MB}$），写回 HBM；  
  启动激活核函数，重新从 HBM 读出这两份数据（$117.4 \text{ MB}$），在片上算完相乘，再将结果写回 HBM（$58.7 \text{ MB}$）；  
  总计带来额外的无用 HBM 读写：$58.7 \times 4 = 234.8 \text{ MB}$！
- **水平熔合状态：**  
  中间这 $234.8 \text{ MB}$ 的往返搬运在芯片片上被**彻底抹除，直接降为 0！**

---

## 步骤 6：核心系统工程铁律

> **把 Gate 与 Up 投影捆在一起冲压，在片上完成门控开合。**  
> FFN 占据了全模型绝大多数的计算和参数量。必须通过权重水平拼接与算子熔合消灭中间激活值的显存回写，以最庞大的矩阵乘法喂饱底层张量核心。
