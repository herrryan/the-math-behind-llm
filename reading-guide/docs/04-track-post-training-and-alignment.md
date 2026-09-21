# 第四主线：后训练与对齐洗牌——从 3 阶段复杂 RLHF 到极简 DPO 与高效微调

> 本篇精读让大模型走出象牙塔、拥有人类仆从品格与专业能力的 4 篇里程碑论文：
> 1. *InstructGPT (Training language models to follow instructions with human feedback)* (Ouyang et al., OpenAI, 2022)
> 2. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)* (Rafailov et al., Stanford, 2023)
> 3. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
> 4. *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)
>
> 学习目标：搞明白为什么预训练出来的模型满嘴胡话？OpenAI 的三阶段 RLHF 为何让工程师夜不能寐？斯坦福如何用极其优美的数学闭式解（DPO）消灭强化学习？以及消费级显卡如何通过低秩矩阵实现四两拨千斤。

---

## 为什么预训练大模型不能直接拿来当对话助手？

刚刚完成预训练的原始模型（Base Model），其本质只是一个**冷酷的互联网文本续写机**。
如果你问它：`"请帮我写一封辞职信。"`
它很可能不会给你写辞职信，而是顺着互联网上的论坛语料继续续写：
- `"……辞职信写完后，老板勃然大怒，接着发生了以下事情……"`
因为它认为你在写小说！

为了让模型从“野生续写机”蜕变成“谦逊、诚实、有用（Helpful, Honest, Harmless）的助手”，必须经过**后训练（Post-Training）**。

---

## 论文 1：ChatGPT 背后的真正功臣——InstructGPT (RLHF)

- **文献链接**：[[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)] · [[PDF 官方直达](https://arxiv.org/pdf/2203.02155.pdf)]
- **作者团队**：Long Ouyang et al. (OpenAI, 2022)

### 老师讲透经典三步走流水线：

```
[ 第一阶段：SFT (监督微调) ]
  人类专家撰写几万条问答样例，通过交叉熵损失教模型：“遇到提问，要这样礼貌回答”。
            │
            ▼
[ 第二阶段：训练 Reward Model (奖励裁判模型) ]
  模型针对同一个问题生成 4-9 个回答，人类对其进行优劣排序 (A > B > C > D)。
  训练一个独立的评分模型，让胜出回答的得分显著高于落败回答。
            │
            ▼
[ 第三阶段：PPO (近端策略优化强化学习) ]
  用奖励模型的打分作为反馈信号，通过 PPO 算法指导语言模型自发调整参数，
  同时引入 KL 散度惩罚，防止模型投机取巧彻底跑偏。
```

### 为什么工业界对 PPO 叫苦不迭？
在进行 PPO 训练时，GPU 显存中必须**同时驻留 4 个庞大模型**：
1. **Actor Model（正在训练的策略模型）**：负责吐词生成回答；
2. **Critic Model（价值评估模型）**：负责预测当前状态的长期期望得分；
3. **Reference Model（参考模型，通常是 SFT 权重）**：负责计算 KL 散度，防止 Actor 越轨；
4. **Reward Model（奖励模型）**：负责给完整的回答打分。
- **痛点**：显存直接翻 4 倍，且 PPO 对超参数极其敏感，一旦策略梯度估算出现偏差，模型会陷入“奖励黑客（Reward Hacking）”——例如疯狂吐标点符号或迎合裁判，导致回答逻辑彻底崩溃。

---

## 论文 2：后训练的极简革命——DPO (直接偏好优化)

- **文献链接**：[[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)] · [[PDF 官方直达](https://arxiv.org/pdf/2305.18290.pdf)]
- **作者团队**：Rafael Rafailov, Archit Sharma, Eric Mitchell et al. (Stanford University, 2023)

### 第一步：大白话直觉——消灭中间商
斯坦福团队提出了一个灵魂拷问：
**为什么非要在中间训练一个专门打分的奖励模型，再拿强化学习去逼着语言模型迁就它？语言模型自己本身不就是一个天然能算概率的打分器吗？**

### 第二步：数学解析解的神来之笔
在强化学习最优解中，可以精确推导出真实隐式奖励函数与最优策略概率之间的代数关系：
$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$
将这个奖励表达式直接代入人类偏好的 Bradley-Terry 模型中，神奇的事情发生了：**复杂的归一化项 $Z(x)$ 在做差时被完全抵消了！**

由此诞生了不需要任何独立 Reward 模型、不需要任何强化学习采样循环的 **DPO 损失函数**：
$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$
- $y_w$ 是人类偏好的优质回答（Winner）；
- $y_l$ 是人类嫌弃的劣质回答（Loser）；
- **物理含义**：如果当前模型对于好回答的相对概率提升、对坏回答的相对概率下降，损失函数就急剧减小！
- **时代影响**：把复杂的 RLHF 简化成了一个普通的多 GPU 有监督分类任务，成为今天开源界后训练的首选对齐基准。

---

## 论文 3 & 4：微调的平民救星——LoRA 与 QLoRA

- **文献链接**：
  - LoRA: [[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)]
  - QLoRA: [[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)]
- **作者团队**：Edward Hu et al. (Microsoft, 2021) / Tim Dettmers et al. (UW, 2023)

### 老师讲透低秩分解的几何直觉：
全量微调一个 70B 模型需要修改 700 亿个参数，连同 Adam 优化器状态需要几百 GB 显存。
微软团队提出了**内在维度（Intrinsic Dimension）假说**：
虽然模型的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 处于高维空间，但我们在微调特定下游任务时，所需要的**知识更新量 $\Delta W$ 实际上完全落在了一个极低维度的流形（子空间）上！**

**数学构造**：
冻结原始权重 $W_0$，将更新量分解为两个极瘦的高窄矩阵：
$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)
$$
- $A \in \mathbb{R}^{r \times k}$，用高斯随机数初始化；
- $B \in \mathbb{R}^{d \times r}$，初始完全为 0（保证训练刚开始时 $\Delta W = 0$，模型行为与原始基座完全一致）；
- 秩 $r$ 通常仅仅取 8 或 16。
- **QLoRA 再次加码**：引入了 **4-bit NormalFloat (NF4)**，把被冻结的基座权重压缩进 4 位整型显存中，使得一张普通的 RTX 3090/4090 显卡就能直接微调 33B 甚至 70B 级别的超级大模型！
