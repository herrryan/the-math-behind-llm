# 第 E10 章：旋转位置编码（RoPE）片上即时计算：避免显存查表开销与 FlashAttention 融合实践

## 步骤 1：物理直觉

想象一位百步穿杨的神箭手，必须根据目标距离调整每次射箭的瞄准角度：
- **查表法（静态预计算 Table Lookup）：** 弓箭手怀里揣着一本厚重的大英百科全书式的仰角表。每次搭箭前，他必须把弓放下，双手在口袋里掏出大书翻查半天（从 HBM 加载长序列的旋转矩阵表），翻书的时间比射箭本身慢了 10 倍！
- **即时心算法（On-the-fly In-register Computation）：** 弓箭手脑海中烂熟简单的几何公式。眼睛一瞥距离，脑中闪电一算（利用芯片内部超高速的三角函数计算单元），手中的箭当场顺势调整角度脱弦而出，**根本不需要任何大书占用口袋空间**！

---

## 步骤 2：芯片微观底层执行机制

旋转位置编码（Su et al., RoFormer）通过正交二维旋转矩阵给每个 Query 和 Key 注入绝对与相对位置信息：

$$
\mathbf{R}_{\Theta, m}^{(i)} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

```
朴素实现缺陷 (Naive Precomputed Table):
在 HBM 中预先分配一张巨大的 [S_max, d_head] 尺寸的 cos/sin 查找表:
每次前向计算: 从 HBM 读取 Q, K ──► 从 HBM 额外读取 cos, sin 查找表 ──► 执行旋转 ──► 再次写入 HBM
==> 显存带宽消耗翻倍! 且当序列扩展至 128K 时，预计算表占据巨量静态显存。

片上即时融合实现 (Fused On-the-fly RoPE):
在 Q, K 刚由 GEMM 计算出并驻留在片上寄存器时:
直接调用 GPU 硬件级超越函数单元 (MUFU 单元中的 __sinf, __cosf)
现场即时计算当前位置 m 对应的角度并就地完成二维旋转!
==> 额外 HBM 显存读写量: 严格等于 0!
```

---

## 步骤 3：跨组件相互耦合机制

1. **与 FlashAttention 的前置融合（Prologue Fusion）：**  
   最先进的注意力内核（如 FlashAttention-2 / vLLM 算子）将 RoPE 计算直接熔合在加载 Key/Value Tile 进入 SRAM 的瞬间完成，彻底消除了 RoPE 算子与后续注意力之间的显存物化屏障。
2. **长文本外推（YaRN / Frequency Scaling）零显存代价：**  
   当需要将上下文长度从 8K 动态外推到 128K 时，如果采用片上即时计算，只需要动态修改寄存器中的频率标量基底 $\text{base}$，模型不需要重新分配或扩容任何显存缓冲区。

---

## 步骤 4：精确性能数学公式

单步前向传播中，全量 Q 和 K 张量应用 RoPE 的二维平面旋转代数式：

$$\begin{aligned}
x_{2i}' &= x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i) \\
x_{2i+1}' &= x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
\end{aligned}$$

未熔合查表实现下的 HBM 访存总开销：

$$\text{Bytes}_{\text{unfused}} = 2 \times \underbrace{(B \cdot H \cdot S \cdot d \cdot 2)}_{\mathbf{Q}, \mathbf{K} \text{ 读写}} + \underbrace{(2 \cdot S \cdot d \cdot 2)}_{\cos, \sin \text{ 查表读取}} \text{ 字节}$$

而在片上即时融合实现下：

$$\text{Bytes}_{\text{fused}} = 0 \text{ 额外 HBM 字节}$$

其计算复杂度仅增加极少量的 MUFU 三角函数时钟周期，被完全隐藏在内存加载的等待延迟中。

---

## 步骤 5：具体微基准数字推导

以 LLaMA-3 70B 模型在处理超长文本 $S = 65536$（64K 上下文），$H = 64, d = 128$、单层批大小 $B = 1$ 下评估：

1. **未熔合查表方案的 HBM 流量：**
   - 每次调用需要从 HBM 加载 $\mathbf{Q}, \mathbf{K}$ 并写回，同时加载庞大的位置编码表：
     $$\text{单层流量} \approx 4 \times (64 \times 65536 \times 128 \times 2) \approx 4.29 \text{ GB}$$
   - 全模型 80 层累计 HBM 搬运量高达 **$343.2\text{ GB}$**！
   - 在 H100（$3.35\text{ TB/s}$）上仅为了给向量转个角度，就必须白白浪费：
     $$T = \frac{343.2 \times 10^9}{3.35 \times 10^{12}} \approx 102.4 \text{ 毫秒}$$
2. **采用片上即时融合后：**
   - 该 $343.2\text{ GB}$ 的显存读写被**彻底永久抹除**！
   - 计算耗时与后续算子流水线完全重叠，**物理执行耗时几乎降为 0**！

---

## 步骤 6：核心系统工程铁律

> 永远不要在全局显存中为旋转位置编码维护庞大的查找表。在片上寄存器利用硬件超越函数单元即时计算正余弦旋转，将 RoPE 紧密熔合在注意力加载的前哨，是实现长上下文无损高吞吐的必然之选。
