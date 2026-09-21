# 第 E07 章：大规模 GEMM 与张量核心：脉动阵列、分块平铺与维度量化对齐

## 步骤 1：物理直觉

想象你要手工核算两本厚达 1000 页的巨型账本的交叉乘积：
- 如果你拿一支铅笔，按照一行一行、逐个数字去翻书计算，你 99% 的时间都在疲惫地来回翻页找数字；
- 现代 GPU 中的 **Tensor Core（张量核心）**就像一台由齿轮紧密咬合的**自动化流水印钞冲压机（脉动阵列 Systolic Array）**：
  - 数字像有节奏的心跳脉搏一样，整齐划一地流过网格中的微型乘加齿轮；
  - 中间计算结果永远锁在齿轮之间层层累加传递，**完全不需要拿笔写在草稿纸上（零中间显存访存）**；
  - 只要进料口源源不断送入整齐的矩阵块，冲压机就能以每秒上千亿次的速度满负荷轰鸣！

但有一个致命前提：**原材料的尺寸必须严丝合缝对齐模具（如 16 或 64 的整数倍）**。如果送进来的纸张稍微多出 1 毫米，整台机器就会瞬间停机报警（硬件流水线停顿排空）！

---

## 步骤 2：芯片微观底层执行机制

在现代 NVIDIA GPU（Ampere/Hopper）上，通用矩阵乘法（GEMM: $\mathbf{C} = \mathbf{A}\mathbf{B} + \mathbf{C}$）由底层的 **Tensor Core 硬件指令** 驱动执行：

```
分层瓦片分块金字塔 (Hierarchical Tiling):
全局显存 (HBM) 矩阵 A [M x K], B [K x N]
      │
      ├── 1. 线程块分块 (Threadblock Tile: 例如 128 x 256 x 64)
      ▼
片上共享内存 (SRAM / Shared Memory)
      │
      ├── 2. 线程束分块 (Warp Tile: 例如 64 x 64 x 64)
      ▼
片上寄存器文件 (Registers)
      │
      ├── 3. 指令级微元切片 (MMA Sub-tile: 16 x 8 x 16 硬件脉动矩阵指令)
      ▼
Tensor Core 物理执行单元 (单时钟周期完成密集乘加做功)
```

### 维度量化与对齐陷阱（Dimension Quantization）：
- 硬件 MMA 指令要求矩阵微块尺寸必须是 16 的倍数，而高效的 SRAM 向量加载指令（如 `LDG.E.128` 一次性搬运 16 字节）要求内存首地址与跨度必须 **128 字节对齐**；
- 若词表大小 $|V|$ 没有对齐（例如原生词表为 32,001），在最后的解码反向投影 $\mathbf{W}_{\text{vocab}}$ 时，矩阵边缘将产生大量无法被 Tensor Core 打包的残差碎片。GPU 只能降级启动低效的通用 CUDA Core 处理边缘，导致整体算力利用率暴跌 30% 以上！

---

## 步骤 3：跨组件相互耦合机制

1. **词表填充（Vocabulary Padding）：**  
   在模型架构设计之初，明智的工程师会主动将词表大小向上填充至 64 或 128 的整数倍（例如将 32,001 填充至 32,064 或 32,128）。虽然多占用了几十 KB 的空闲参数显存，但消除了 GEMM 边界不对齐停顿，使全网最终的 Softmax 预测投影始终维持在 80% 以上的硬件峰值效率。
2. **批大小与 Prefill 吞吐：**  
   在提示词预填（Prefill）阶段，更大的 Token 批大小 $M = B \times S$ 能够铺满更多的 SM 线程块，使 Tensor Core 的占用率（Occupancy）迅速饱和。

---

## 步骤 4：精确性能数学公式

对于形状为 $\mathbf{A} \in \mathbb{R}^{M \times K}$ 与 $\mathbf{B} \in \mathbb{R}^{K \times N}$ 的矩阵乘法：

理论浮点运算量为：

$$\text{GEMM FLOPs} = 2 M N K$$

Tensor Core 的实际硬件运算效率（Efficiency）定义为：

$$\eta_{\text{GEMM}} = \frac{2 M N K}{T_{\text{exec}} \times P_{\text{peak}}}$$

其中 $T_{\text{exec}}$ 为实际执行耗时，$P_{\text{peak}}$ 为芯片理论峰值算力。

在矩阵乘法的显存搬运中，输入与输出的理论最小数据量为：

$$\text{Bytes} = (M K + K N + M N) \times \text{sizeof(dtype)}$$

当 $M, N, K$ 均极大时，算术强度逼近物理理论极限：

$$I_{\text{GEMM}} \approx \frac{2 M N K}{(M K + K N + M N) \times 2} \approx \frac{K}{2} \text{ FLOPs/Byte} \quad (\text{当 } M=N=K \text{ 且 FP16 精度时})$$

---

## 步骤 5：具体微基准数字推导

以 H100 SXM（BF16 密集峰值算力 $P_{\text{peak}} \approx 1,000\text{ TFLOPS}$）运行 LLaMA-3 70B 单层 FFN 投影矩阵（$M = 4096, K = 8192, N = 28672$）为例：

1. **总浮点运算量：**
   $$F = 2 \times 4096 \times 8192 \times 28672 \approx 1.924 \times 10^{12} \text{ FLOPs} \approx 1.924 \text{ TFLOPs}$$
2. **硬件理论极限最小执行时间：**
   $$T_{\text{min}} = \frac{1.924 \times 10^{12}}{1.0 \times 10^{15}} \approx 0.001924 \text{ 秒} \approx 1.92 \text{ 毫秒}$$
3. **未对齐情况下的实测对比：**
   - 若 $N$ 保持 128 对齐，采用 CUTLASS 高度优化内核时，实测耗时为 $2.35\text{ ms}$（Tensor Core 效率 $\eta \approx 81.7\%$）；
   - 若人为引入奇数维度不对齐，实测耗时增加至 $3.62\text{ ms}$（效率暴跌至 $53\%$）！
- **结论：** 仅靠严格的硬件维度对齐与分块平铺，同一个物理芯片在相同代码下性能立涨 **$54\%$**！

---

## 步骤 6：核心系统工程铁律

> Tensor Core 是为严密的规则几何矩阵而生的精密机器。永远将模型的隐藏层维度、中间层维度与词表大小严格对齐至 64 或 128 的整数倍；任何打破内存对齐的奇数维度，都是在向硬件流水线注入高昂的空闲停顿。
