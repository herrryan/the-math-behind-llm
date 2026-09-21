# 02. 淘汰与过时篇：哪些昔日明星方法已经被彻底取代？

大模型领域的快速发展，不仅体现在新技术的诞生，更体现在对旧技术的“无情淘汰”。
本章系统盘点已被现代大语言模型**彻底淘汰或不再建议深入研读**的 7 大方向，每篇经典均附带论文出处与 **PDF 链接**，帮助你明辨历史脉络、果断断舍离。

---

## 淘汰榜单概览速查表

| 技术领域 | 曾经的明星代表论文（附 PDF 直达） | 当前已被什么彻底取代？ | 淘汰原因与状态判定 |
| :--- | :--- | :--- | :--- |
| **词表示** | [*Word2Vec* (Mikolov 2013)](https://arxiv.org/pdf/1301.3781.pdf), [*GloVe* (Pennington 2014)](https://aclanthology.org/D14-1162.pdf) | 动态上下文词嵌入（Transformer Embedding） | **已彻底淘汰**。无法解决一词多义，语义固定静态，现代模型已完全不用。 |
| **序列骨架** | [*LSTM* (Hochreiter 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf), [*GRU* (Cho 2014)](https://arxiv.org/pdf/1406.1078.pdf) | Transformer 因果自注意力机制 | **通用领域已淘汰**。无法利用 GPU 并行，梯度消失，长程记忆衰减严重。 |
| **预训练目标** | [*BERT* (Devlin 2018)](https://arxiv.org/pdf/1810.04805.pdf), [*RoBERTa* (Liu 2019)](https://arxiv.org/pdf/1907.11692.pdf) | 纯 Decoder-only 自回归自监督接龙 (Next-Word Prediction) | **生成时代已淘汰**。双向编码器无法高效生成文本，已被因果大模型统领。 |
| **模型拓扑** | [*T5* (Raffel 2019)](https://arxiv.org/pdf/1910.10683.pdf), [*BART* (Lewis 2019)](https://arxiv.org/pdf/1910.13461.pdf) | 纯自回归 Decoder-only 架构 (GPT, LLaMA) | **主干已淘汰**。双塔结构增加推理复杂度与 KV 缓存冗余，参数效率不及 Decoder。 |
| **注意力加速** | [*Performer* (2020)](https://arxiv.org/pdf/2009.14794.pdf), [*Linformer* (2020)](https://arxiv.org/pdf/2006.04768.pdf), [*Reformer* (2020)](https://arxiv.org/pdf/2001.04451.pdf) | 硬件感知精确计算 ([*FlashAttention*](https://arxiv.org/pdf/2205.14135.pdf)) | **已被彻底证伪**。低秩逼近不仅破坏语义精度，而且在现代 GPU 上并不快。 |
| **位置编码** | Sinusoidal 正余弦绝对编码, 可学习绝对位置编码 | 旋转位置编码 ([*RoPE*](https://arxiv.org/pdf/2104.09864.pdf)) 与 YaRN | **绝对编码已过时**。无法自然外推长文本，相对位置感知力低下。 |
| **训练与对齐** | Post-LN (后置归一化), [*InstructGPT 复杂四模型 PPO* (2022)](https://arxiv.org/pdf/2203.02155.pdf) | Pre-RMSNorm, [*DPO*](https://arxiv.org/pdf/2305.18290.pdf) / [*GRPO*](https://arxiv.org/pdf/2402.03300.pdf) | **工程落地已淘汰**。Post-LN 极易梯度爆炸；传统 PPO 显存庞大、极难调参。 |

---

## 深度剖析：为什么它们会被淘汰？

### 1. 静态词向量：Word2Vec 与 GloVe
- **代表论文**：
  - *Efficient Estimation of Word Representations in Vector Space* (Mikolov et al., 2013) [[arXiv:1301.3781](https://arxiv.org/abs/1301.3781)] · [[PDF](https://arxiv.org/pdf/1301.3781.pdf)]
  - *GloVe: Global Vectors for Word Representation* (Pennington et al., 2014) [[PDF](https://aclanthology.org/D14-1162.pdf)]
- **当时的作用**：2013 年首次用低维连续向量表达词语，实现了著名的向量代数奇迹（“国王” - “男人” + “女人” = “王后”）。
- **淘汰原因**：
  静态词向量给每个词分配了全局唯一的固定向量。面对多义词“苹果”（水果 vs 手机科技公司），静态向量只能被迫取两者的平均折中，表达能力严重受限。
- **当前替代方案**：
  现代 LLM 的词向量仅仅是第 0 层的初始锚点，一进入后续 80 层的注意力机制，词向量会根据周围的所有单词动态演变为完全契合当前语境的**深层上下文动态表征**。
- **学习建议**：理解“向量空间距离”的哲学概念即可，完全不需要花时间死磕 Skip-Gram、CBOW 或负采样的繁琐数学推导。

---

### 2. 序列传话筒：RNN、LSTM 与 GRU
- **代表论文**：
  - *Long Short-Term Memory* (Hochreiter & Schmidhuber, 1997) [[PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)]
  - *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (GRU)* (Cho et al., 2014) [[arXiv:1406.1078](https://arxiv.org/abs/1406.1078)] · [[PDF](https://arxiv.org/pdf/1406.1078.pdf)]
- **淘汰原因**：
  1. **串行计算致命缺陷**：计算第 100 个词必须等前 99 个词算完，导致上千张现代 GPU 的并行矩阵乘法硬件（Tensor Core）大部分时间处于空闲等待状态；
  2. **长程遗忘与信息压缩瓶颈**：随着距离拉长，早期的上下文信息被反复有损压缩，即使有门控机制（LSTM），在超过几百个字后依然不可避免地发生语义失真。
- **学习建议**：知道“传话筒存在记忆丢失与排队等待”作为对比背景即可，无需推导遗忘门、输入门、候选隐状态公式。

---

### 3. 掩码双向语言模型：BERT 与 RoBERTa
- **代表论文**：
  - *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018) [[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)] · [[PDF](https://arxiv.org/pdf/1810.04805.pdf)]
  - *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (Liu et al., 2019) [[arXiv:1907.11692](https://arxiv.org/abs/1907.11692)] · [[PDF](https://arxiv.org/pdf/1907.11692.pdf)]
- **淘汰原因**：
  1. **无法做高效的自然文本生成**：BERT 天生依赖双向全知视角，无法像自回归模型那样一个词接一个词地流式吐字；
  2. **大一统时代的降维打击**：学术界曾经认为“判别任务用 BERT，生成任务用 GPT”，但随着自回归模型 Scaling 到百亿千亿参数，其少样本与思维链推理能力在判别任务上也彻底碾压了 BERT。
- **学习建议**：BERT 是现代双向编码器的绝唱。除非你从事极其特异的纯文本分类轻量级边缘部署，否则不要在大模型主干学习中花费超过 1 个小时。

---

### 4. 早期线性与稀疏注意力：Performer、Linformer、Reformer
- **代表论文**：
  - *Rethinking Attention with Performers* (Choromanski et al., 2020) [[arXiv:2009.14794](https://arxiv.org/abs/2009.14794)] · [[PDF](https://arxiv.org/pdf/2009.14794.pdf)]
  - *Linformer: Self-Attention with Linear Complexity* (Wang et al., 2020) [[arXiv:2006.04768](https://arxiv.org/abs/2006.04768)] · [[PDF](https://arxiv.org/pdf/2006.04768.pdf)]
  - *Reformer: The Efficient Transformer* (Kitaev et al., 2020) [[arXiv:2001.04451](https://arxiv.org/abs/2001.04451)] · [[PDF](https://arxiv.org/pdf/2001.04451.pdf)]
- **为什么这是一条“纸上谈兵的死胡同”？**
  1. **破坏模型精度**：低秩逼近和近似截断大幅破坏了大模型在长程依赖、代码括号匹配、数学检索等需要精细定位任务上的表现；
  2. **违反 GPU 物理真实硬件法则**：现代 GPU 的核心瓶颈不是浮点算力（FLOPs），而是**显存访问带宽（Memory Bandwidth）**。那些看似数学上只要 $O(N)$ 计算量的复杂核算法，由于引入了大量不规整的非连续内存读写与中间张量，在真实显卡上跑起来往往比朴素的矩阵乘法还要慢得多！
- **终结者**：**FlashAttention** [[PDF](https://arxiv.org/pdf/2205.14135.pdf)]。FlashAttention 证明了一件事：**根本不需要做任何有损数学近似，直接通过硬件层级的内存分块（Tiling）算法，就能让标准精确注意力在显卡上跑出数倍的极速，并将显存压缩到极致！**
- **学习建议**：彻底跳过所有线性近似注意力论文，直接精读 FlashAttention。
