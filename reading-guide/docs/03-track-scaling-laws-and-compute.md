# 第三主线：预训练物理定律——算力、数据与参数的 Scaling Laws 革命

> 本篇精读统治现代大语言模型算力投资与训练规划的两大神级论文：
> 1. *Scaling Laws for Neural Language Models* (Kaplan et al., OpenAI, 2020)
> 2. *Training Compute-Optimal Large Language Models (Chinchilla)* (Hoffmann et al., DeepMind, 2022)
>
> 学习目标：像顶尖实验室的首席科学家一样，学会计算 FLOPs 算力账本，推导参数量 $N$ 与训练数据量 $D$ 的最佳配比，深刻理解为什么“小模型大灌水”成为了当今开源大模型的致胜法宝。

---

## 训练算力的通用物理账本（必须背下的基石公式）

在大模型预训练中，总计算量用浮点运算次数（FLOPs, Floating Point Operations）来衡量。
对于一个非词嵌入参数量为 $N$、训练数据 Token 数量为 $D$ 的标准自回归 Transformer：

$$
C \approx 6 N D
$$

### 老师讲透来源：为什么乘数是 6？
- **前向传播（Forward Pass）**：每个 Token 的输入需要经过矩阵乘法，一次加法加一次乘法算 2 次浮点运算，因此前向传播消耗约为 $2 N$ FLOPs；
- **反向传播（Backward Pass）**：反向传播既要计算对输入的梯度，又要计算对权重的梯度，计算量精确等于前向传播的 2 倍，即 $4 N$ FLOPs；
- **合计**：$2 N + 4 N = 6 N$ FLOPs / Token。
- 乘以总数据量 $D$，即可精准估算出一次完整训练所消耗的理论算力总量！

---

## 论文 1：大模型领域的开普勒定律——OpenAI Kaplan 2020

- **文献链接**：[[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)] · [[PDF 官方直达](https://arxiv.org/pdf/2001.08361.pdf)]
- **作者团队**：Jared Kaplan et al. (OpenAI, 2020)

### 核心贡献与公式拆解：
在 2020 年之前，深度学习被嘲讽为“玄学炼丹”。没有人知道模型做大到底有没有用，还是会很快过拟合。
Kaplan 团队训练了几百个不同尺寸的模型，首次定量给出了三大幂律（Power-law）法则：

1. **模型损失（Loss $L$）只与三个宏观数字强相关**：
   - 模型参数量 $N$；
   - 训练 Token 数 $D$；
   - 总计算预算 $C$。
   至于你的 Transformer 到底是设计得更宽还是更深、注意力头数多几个还是少几个，在对数坐标轴下**统统无关紧要**！

2. **幂律公式**：
$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$
在双对数坐标系（Log-Log Plot）下，Loss 与参数量、数据量呈现出完美的笔直斜线。这向全世界资方与工程师宣告：**只要持续堆算力和高质量数据，智能水平就可以被稳定预测并持续增长！**

### Kaplan 论文的历史局限（踩坑点）：
在这篇论文中，OpenAI 得出了一个错误的推论：
他们认为，如果算力预算增加，**应该优先把参数量 $N$ 做大（占 73%），而训练数据 $D$ 只需要微调少加一点（占 27%）**。
这一错误结论直接导致了后来的 GPT-3（175B 参数，但仅仅只喂了 300B Token，严重营养不良！）。

---

## 论文 2：DeepMind 的历史纠错——Chinchilla（2022）

- **文献链接**：[[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)] · [[PDF 官方直达](https://arxiv.org/pdf/2203.15556.pdf)]
- **作者团队**：Jordan Hoffmann, Sebastian Borgeaud et al. (DeepMind, 2022)

### 老师讲透纠错过程：
DeepMind 团队敏锐地发现：OpenAI 实验之所以得出“参数比数据更重要”，是因为 OpenAI 在实验中采用了一个固定的学习率余弦调度（Cosine Schedule）步长，导致很多较小的模型在还没完全收敛时就被强行测了 Loss！

DeepMind 重新设计了极其严谨的 400 多个模型跨算力对比实验，构建了参数与数据的双变量凸优化函数：
$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$
在给定算力预算 $C = 6 N D$ 的约束下，利用拉格朗日乘数法求极值，得到了震惊业界的**最优扩展定律**：
$$
N \propto C^{0.5}, \quad D \propto C^{0.5}
$$
- **结论**：**参数量 $N$ 与 数据量 $D$ 应该以 1:1 的完全相等比例同时扩张！**
- **黄金换算比**：对于一个计算最优的模型，**每一个模型参数，至少需要喂养约 20 个 Token 的数据！**

### Chinchilla 的震撼实践：
- 当时业内的明星模型 **Gopher**（280B 参数，喂了 300B Token，严重吃不饱）；
- DeepMind 依据最优定律，直接把参数砍掉四分之三，造出了仅有 **70B 参数的 Chinchilla**，但给它狠狠灌入了 **1.4T Token** 的数据；
- **战果**：70B 的 Chinchilla 在各项下游跑分上全面爆锤 280B 的 Gopher 和 175B 的 GPT-3！

---

## 当代开源的“超饱和预训练”（Beyond Compute-Optimal）

Chinchilla 探讨的是**“在一次性预训练计算预算固定下的最优配比”**。
但在现实商业中，人们发现：**模型训练只是一次性的，而推理部署要跑几千万次！**
- 如果模型做得太大（如 70B），哪怕预训练算力省了，部署到生产环境需要 4 张 80G 显卡，每秒推理极其昂贵；
- 如果把模型做小（如 8B），哪怕远远超过 1:20 的 Chinchilla 上限，狠狠给它喂入 **15T Token（参数数据比达到了惊人的 1:1875）**，模型不但不会过拟合，能力还在疯狂变强！
- **现代启示**：以 Llama 3 8B、Qwen2.5-7B 为代表的“小钢炮”，用极致的超饱和数据灌溉，造就了消费级单卡即可畅跑的超强智力体。
