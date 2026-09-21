# 第 E04 章：FlashAttention (1, 2, 3) 演进：SRAM 平铺分块、在线 Softmax 与 TMA 异步传输

## 步骤 1：物理直觉

在第 E03 章中，我们看到每次做千层面都要跑到远端地下冷库搬运巨大托盘。现在，一位聪明的工程大师改写了工作流：
- 他拿来一个刚好能放在身旁备餐台（片上 SRAM）上的**小保鲜盒（Tile 分块）**；
- 他一次只从冷库取出一小叠面皮（一个 Block），在备餐台上当场完成涂抹酱汁与局部加权累加；
- 他发明了一种数学技巧（**在线 Softmax 动态更新**）：当下一盒新面皮端上来时，如果发现新面皮的温度更高，他不需要倒掉重做，而是用一个缩放系数瞬间校正碗里已经做好的半成品，把新的数据融合进来；
- 整个制作过程一气呵成，巨大的中间矩阵彻底不需要存在，所有计算都在小保鲜盒与备餐台上搞定！

这就是 **FlashAttention** 的本质：将注意力机制的显存读写复杂度，直接从 $O(S^2)$ 砍到与序列长度呈严格线性的 $O(S)$！

---

## 步骤 2：芯片微观底层执行机制

FlashAttention 的核心突破由三大物理机制协同支撑：

### 1. 瓦片分块（Tiling）与 SRAM 驻留
将 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 矩阵在序列维度切分成较小的块（如 $B_r \times d$ 和 $B_c \times d$，典型值为 $64 \times 128$ 或 $128 \times 128$），完全适配单个 SM 的 $228\text{ KB}$ SRAM 大小。

```
输入矩阵 Q, K, V (位于高带宽显存 HBM)
        │
        ├── 每次仅加载一个 Tile 分块至片上共享内存 (SRAM)
        ▼
┌────────────────────────────────────────────────────────┐
│                   单个 SM 的 SRAM 内部                 │
│  [分块 Q_i: 64x128]  @  [分块 K_j^T: 128x64]           │
│  ==> 产生微型中间矩阵 S_ij (仅 64x128 大小!)            │
│  ==> 局部在线 Softmax: 动态跟踪 max_val 与 sum_exp      │
│  ==> 局部乘加累加至局部输出 O_i                         │
└────────────────────────────────────────────────────────┘
        │
        └── 遍历完全部 K, V 块后，仅将最终注意力结果 O_i 写回 HBM!
```

### 2. 在线 Softmax（Online Softmax）数学机制
标准 Softmax 需要遍历整行求全局最大值 $m = \max(x_i)$ 和全局归一化分母 $l = \sum e^{x_i - m}$。在线 Softmax（Milakov & Gimelshein, 2018）允许在分块流式迭代中，每遇到新的局部最大值 $\tilde{m}$，利用缩放因子动态修正旧累加和：

$$
m_{\text{new}} = \max(m_{\text{old}}, \; \tilde{m})
$$

$$
l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \tilde{l} \cdot e^{\tilde{m} - m_{\text{new}}}
$$

$$
\mathbf{O}_{\text{new}} = \mathbf{O}_{\text{old}} \cdot \left(\frac{l_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}}}{l_{\text{new}}}\right) + \tilde{\mathbf{P}} \mathbf{V}_j \cdot \left(\frac{e^{\tilde{m} - m_{\text{new}}}}{l_{\text{new}}}\right)
$$

### 3. FlashAttention-1 到 FlashAttention-3 的代际跃迁
- **FlashAttention-1（Dao et al., 2022）：** 引入 SRAM 分块与在线 Softmax，消灭 HBM 中间矩阵，实现 $2\text{--}4\times$ 加速。外层循环遍历 K 和 V，内层循环遍历 Q。
- **FlashAttention-2（Dao, 2023）：** 反转内外层循环（外层遍历 Q，内层遍历 K/V），消灭不同 SM 之间写回输出 $\mathbf{O}$ 时的同步原子加锁；优化线程束内部划分，算力利用率从 $35\%$ 跃升至 $50\text{--}73\%$。
- **FlashAttention-3（Shah et al., 2024，专为 Hopper 架构打造）：** 
  - 充分利用硬件异步张量内存加速器（TMA），绕过寄存器直接将数据从 HBM 拷贝至 SRAM；
  - 采用 Warp-Specialization（线程束专业化分组），让一组 Warp 专职负责异步数据搬运，另一组 Warp 专职在 Tensor Core 上满负荷运算，实现数据加载与矩阵乘法的完全重叠；
  - 支持 FP8 低精度，算力利用率突破 $75\text{--}80\%$。

---

## 步骤 3：跨组件相互耦合机制

1. **反向传播中的以计算换显存（Activation Recomputation）：**  
   FlashAttention 在反向传播时不从 HBM 读取庞大的 Softmax 概率矩阵 $\mathbf{P}$，而是仅保留小巧的标量统计量 $(m, l)$，在反向传播时于 SRAM 中极速**重新计算**局部小块的 $\mathbf{S}$ 和 $\mathbf{P}$。显存占用由 $O(S^2)$ 骤降为 $O(S)$，为增大训练批大小释放了数十吉字节的空间。
2. **头维度 $d$ 与 SRAM 容量的博弈：**  
   若单头维度从 $d = 64$ 扩大到 $d = 128$ 再到 $d = 256$，单个 Tile 所占用的 SRAM 会成倍增加。当 $d = 256$ 时，单个 SM 无法再放下较大的分块，必须缩小 Tile 尺寸，导致共享内存读取冲突增加。
3. **因果掩码（Causal Masking）的计算短路：**  
   在因果自回归模型中，对于注意力矩阵右上角全为 0 的无效分块，FlashAttention 可以直接从调度层面整块跳过，直接节省近 $50\%$ 的 Tensor Core 运算量。

---

## 步骤 4：精确性能数学公式

FlashAttention 的全局显存（HBM）总访存量为：

$$
M_{\text{flash}} = 2 B H S d \text{ (读取 Q, 写入 O)} + \frac{2 B H S^2 d}{M_{\text{sram}}} \text{ (多次分块流式加载 K, V)}
$$

其中 $M_{\text{sram}}$ 为单个 SM 的片上可用共享内存容量（以元素计）。

对比朴素注意力与 FlashAttention 的显存读写量：

$$
\frac{M_{\text{flash}}}{M_{\text{naive}}} \approx \frac{O(S)}{O(S^2)} = \frac{4 d}{S} \quad (\text{当 } S \gg d \text{ 时})
$$

当序列长度 $S = 8192$、单头维度 $d = 128$ 时：

$$
\frac{M_{\text{flash}}}{M_{\text{naive}}} \approx \frac{4 \times 128}{8192} = \frac{512}{8192} = \frac{1}{16}
$$

**FlashAttention 将物理显存总线上的数据搬运量缩减到了原来的 $\frac{1}{16}$（降低了 $93.75\%$）！**

其算术强度由此跃升为：

$$
I_{\text{flash}} \approx \frac{4 B H S^2 d}{2 B H S d + \frac{2 B H S^2 d}{M_{\text{sram}}}} \propto M_{\text{sram}} \gg I_{\text{naive}}
$$

使得算子在长序列下成功脱离访存受限区间，昂贵的 Tensor Core 终于得以全速运转。

---

## 步骤 5：具体微基准数字推导

以 H100 SXM 运行 LLaMA-3 8B 序列长度 $S = 8192$、批大小 $B = 1$ 的单层注意力为例：

1. **总浮点运算量：** $F \approx 1.10 \times 10^{12} \text{ FLOPs}$。
2. **FlashAttention 实际 HBM 数据读写量：**
   - 读取 $\mathbf{Q}$: $1 \times 32 \times 8192 \times 128 \times 2 \approx 67.1 \text{ MB}$
   - 写回 $\mathbf{O}$: $67.1 \text{ MB}$
   - 分块流式读取 $\mathbf{K}, \mathbf{V}$：在 $M_{\text{sram}} \approx 100\text{ KB}$ 条件下，有效重用倍率使得读取量仅约 $0.98\text{ GB}$。
   - **HBM 总数据量：** $M \approx 1.11 \text{ GB}$（而朴素实现高达 $17.18\text{ GB}$！）。
3. **访存传输耗时：**
   $$T_{\text{memory}} = \frac{1.11 \times 10^9}{3.35 \times 10^{12}} \approx 0.00033 \text{ 秒} = 0.33 \text{ 毫秒}$$
4. **计算耗时（在 H100 实际利用率 60% 即 600 TFLOPS 下）：**
   $$T_{\text{compute}} = \frac{1.10 \times 10^{12}}{600 \times 10^{12}} \approx 1.83 \text{ 毫秒}$$
5. **实际性能对比：**
   - 朴素注意力物理耗时：$\ge 5.1 \text{ ms}$（显存带宽卡死）
   - FlashAttention 物理耗时：$\approx 1.9 \text{ ms}$（计算密集型，数据搬运时间已被计算完全隐藏）
   - **实测端到端加速比：约 $2.7\times$ 到 $4\times$！**

---

## 步骤 6：核心系统工程铁律

> 算法优化的最高境界不是减少数学乘除法次数，而是改变数据驻留的物理层级。FlashAttention 通过在线 Softmax 巧妙重构代数流程，将原本需要在 HBM 中流转的二次方中间矩阵彻底囚禁在片上 SRAM 中，完成了大模型系统从访存墙向算力充分释放的决定性飞跃。
