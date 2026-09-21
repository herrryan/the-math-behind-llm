# 第二主线：现代主流架构黄金四件套——从标准 Transformer 到 LLaMA 范式

> 本篇精读现代开源大模型事实标准的 5 篇核心文献：
> 1. *RMSNorm: Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)
> 2. *GLU Variants Improve Transformer (SwiGLU)* (Shazeer, 2020)
> 3. *RoFormer: Rotary Position Embedding (RoPE)* (Su et al., 2021)
> 4. *GQA: Grouped-Query Attention* (Ainslie et al., 2023)
> 5. *The LLaMA Family Technical Reports* (Meta, 2023-2024)
>
> 学习目标：像资深架构师一样看清：2017 年的原始 Transformer 究竟有哪些暗伤？工业界是如何通过这四把手术刀逐一修复，最终拼装出风靡当今开源界的 LLaMA / Qwen 终极架构的。

---

## 现代大模型架构演进对比全览

```
[ 2017 年标准 Transformer ]              [ 当代主流开源标准 (LLaMA/Qwen) ]
1. Post-LN (后置层归一化，训练极易发散)   ==> 1. Pre-RMSNorm (前置无均值归一化，极速且稳健)
2. 传统 ReLU / GELU 前馈全连接          ==> 2. SwiGLU 双通道门控线性单元 (更强表达力)
3. 正余弦绝对位置编码 (长文本无法外推)    ==> 3. RoPE 旋转位置编码 (相对距离保持，超强外推)
4. MHA 多头注意力 (KV Cache 显存暴涨)    ==> 4. GQA 分组查询注意力 (显存立省 75%，吞吐暴增)
```

---

## 核心改进 1：Pre-RMSNorm——砍掉多余的算术均值

- **文献链接**：[[arXiv:1910.07467](https://arxiv.org/abs/1910.07467)] · [[PDF 官方直达](https://arxiv.org/pdf/1910.07467.pdf)]
- **作者团队**：Biao Zhang, Rico Sennrich (2019)

### 老师讲透原理：
传统的 **LayerNorm** 公式为：
$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$
其中必须先计算序列的均值 $\mu = \frac{1}{d} \sum x_i$，然后再算方差 $\sigma^2$。

**论文的惊人洞察**：
神经网络之所以需要归一化，是为了**控制激活值的尺度（Scale）**，防止数值在前向和反向传播中指数级爆炸或衰减。至于“把中心平移到 0”（减均值 $\mu$），对梯度的稳定性几乎没有任何贡献，反而在硬件底层增加了两次数据遍历（一次求和算均值，一次做差）！

**RMSNorm 的极简公式**：
直接丢弃均值，只除以向量的均方根（Root Mean Square）：
$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot \gamma
$$
- **收益**：不仅数学上更加干净，而且省去了偏置 $\beta$ 和均值计算，在现代 GPU 上算子执行速度提升了 10% 到 50%，现已成为 LLaMA、Mistral、Gemma、Qwen 的绝对标配。

---

## 核心改进 2：SwiGLU——给思考神经加上可控的“水龙头”

- **文献链接**：[[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)] · [[PDF 官方直达](https://arxiv.org/pdf/2002.05202.pdf)]
- **作者团队**：Noam Shazeer (Google, 2020)

### 老师讲透原理：
传统 Transformer 的 FFN 层采用粗暴的“单通道”两层映射：
$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$
这就好比水流流过管道，ReLU 只能机械地决定“小于 0 的水截断，大于 0 的放行”。

**门控机制（Gated Linear Units, GLU）的哲学**：
把 FFN 拆成两个平行的支路：
1. **内容通道**：由矩阵 $W_{\text{gate}}$ 投影；
2. **门阀通道**：由矩阵 $W_1$ 投影，并通过 Swish 激活函数生成一个介于 0 到 1 之间的连续旋钮；
3. **两路点乘（Hadamard Product）**：
$$
\text{SwiGLU}(x) = \Big(\text{Swish}(x W_1) \otimes (x W_{\text{gate}})\Big) W_2
$$
- **直觉**：内容支路负责提取复杂的概念特征，而门阀支路像一个智能水龙头，精准控制当前特征要以多大比例输出给下一层。大量的实证表明，参数量相当时，SwiGLU 在各项基准测试中均无情碾压传统 ReLU/GELU。

---

## 核心改进 3：RoPE 旋转位置编码——复数平面的神奇时针

- **文献链接**：[[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)] · [[PDF 官方直达](https://arxiv.org/pdf/2104.09864.pdf)]
- **作者团队**：Jianlin Su (苏剑林), Yu Lu, Shengfeng Pan et al. (2021)

### 老师讲透原理：
在没有位置编码时，词向量只有语义坐标，模型根本无法区分“狗咬人”与“人咬狗”。
- **绝对位置编码的硬伤**：直接在第 $m$ 个词向量上加一个固定常数向量 $P_m$。当模型要在 8k 长度推理，但训练时只见过 2k 长度时，$P_{3000}$ 对于模型而言完全是个陌生外星人，模型直接崩溃胡言乱语。

**苏剑林团队的数学神作（RoPE）**：
我们不需要告诉模型“你绝对站在操场的第 5 米还是第 10 米”，我们只需要保证：**第 $m$ 个词的 Query 和第 $n$ 个词的 Key 做内积时，其结果只取决于它们的相对距离 $(m - n)$！**

**实现方式**：
把向量中每两个相邻维度 $(x_1, x_2)$ 视为复数平面上的一个点，把位置 $m$ 转化为一个旋转角度 $m\theta$：
$$
R_{\Theta, m}^d x = \begin{pmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_1 \sin(m\theta) + x_2 \cos(m\theta) \end{pmatrix}
$$
由于两个复数相乘等于角度相加，做内积时角度自然相减：
$$
\langle R_m q, R_n k \rangle = q^\top R_{n-m} k
$$
- **惊人优雅**：不用显式增加任何额外参数，只通过正余弦旋转变换，就让模型完美感知到了相对位置差！更赋予了模型强大的**长文本扩展（RoPE 线性外推、YaRN 插值）**能力。

---

## 核心改进 4：GQA（分组查询注意力）——解救暴涨的显存

- **文献链接**：[[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)] · [[PDF 官方直达](https://arxiv.org/pdf/2305.13245.pdf)]
- **作者团队**：Joshua Ainslie et al. (Google Research, 2023)

### 老师讲透原理：
在部署大模型服务时，**KV Cache 占用了海量的 GPU 显存**：
- **MHA（标准多头注意力）**：如果有 32 个 Query 头，就必须配套 32 个 Key 头和 32 个 Value 头。当并发用户多、上下文长时，几十 GB 显存瞬间被吃光；
- **MQA（多查询注意力）**：走向另一个极端——32 个 Query 头强行共用 1 个 Key 头和 1 个 Value 头。显存立省 96%，但模型的精准注意力严重受损；
- **GQA（折中智慧）**：划分为 4 或 8 个小组，每组内部的几个 Query 头共享同一对 Key/Value 头。
- **结论**：以极小的精度损失代价，将 KV Cache 显存开销直接砍下 75% 以上，推理并发吞吐成倍暴涨。

---

## 终极集大成者：The LLaMA Family

- **文献链接**：
  - LLaMA 1: [[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)]
  - LLaMA 2: [[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)]
  - LLaMA 3: [[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)]
- **作者团队**：Meta AI (2023 - 2024)

Meta 的三篇技术报告并不是提出了多么奇特的新数学算法，而是以极高的工业水准**将上述四大组件彻底固化为全行业标配**，并向全世界证明了：**不要轻易更改这套架构，把精力投入到 15T+ 高质量合成数据清洗、长文本退火训练与高质量 DPO 对齐中，才是现代大模型的通天大道。**
