# 04. 训练与对齐洗牌：预训练数据清洗、DPO 与 GRPO 的三次大革命

后训练（Post-training）决定了大模型的价值观与逻辑分寸感。
在对齐领域，技术的迭代速度甚至超越了底层架构。每篇核心突破均附带官方 **PDF 直达链接**。

---

## 1. 预训练数据演进：从垃圾堆狂吞到精准营养学

- **合成数据与教科书假设论文**：*Textbooks Are All You Need* (Phi-1, Gunasekar et al., Microsoft, 2023) [[arXiv:2306.11644](https://arxiv.org/abs/2306.11644)] · [[PDF](https://arxiv.org/pdf/2306.11644.pdf)]
- **核心认知颠覆**：
  低质数据不仅带来不了智能，反而会导致模型过早产生幻觉。利用特级教师模型将互联网脏数据重写为逻辑缜密的“教材级教科书”，能用极小数据量击败传统十倍规模的大模型。

---

## 2. 监督微调（SFT）的顿悟：LIMA 与“少即是多”假说

- **代表论文**：*LIMA: Less Is More for Alignment* (Zhou et al., Meta, 2023) [[arXiv:2305.11206](https://arxiv.org/abs/2305.11206)] · [[PDF](https://arxiv.org/pdf/2305.11206.pdf)]
- **科学启示**：
  Meta 用仅仅 **1,000 条** 顶级指令问答数据微调 65B 模型，效果媲美数万条普通标注。
  **模型的全部常识在预训练阶段早就学完了！SFT 的唯一使命是规范回答格式和语气，切忌在微调阶段硬灌新事实。**

---

## 3. 强化学习对齐的三次范式洗牌

```
【第一代: 复杂笨拙的四模型 PPO 体系】
  InstructGPT (Ouyang et al., 2022) ──► Actor, Critic, Reward, Reference 四模型常驻
  [arXiv:2203.02155] · [PDF: https://arxiv.org/pdf/2203.02155.pdf]

                                │
                                ▼ 范式革命 1: 消除强化学习环境与价值估计
                                
【第二代: 闭式隐式对齐 DPO / ORPO】
  DPO (Rafailov et al., 2023) ──► 无需训练独立奖励模型, 直接用偏好对计算二分类 Loss
  [arXiv:2305.18290] · [PDF: https://arxiv.org/pdf/2305.18290.pdf]
  ORPO (Hong et al., 2024)    ──► 无需参考模型, 单阶段完成 SFT + 偏好对齐
  [arXiv:2403.07691] · [PDF: https://arxiv.org/pdf/2403.07691.pdf]

                                │
                                ▼ 范式革命 2: 面向推理与严谨领域的去 Critic 纯 RL
                                
【第三代: 群组相对策略优化 GRPO (DeepSeek)】
  DeepSeekMath (Shao et al., 2024) ──► 提出 GRPO, 彻底废弃 Critic 价值网络
  [arXiv:2402.03300] · [PDF: https://arxiv.org/pdf/2402.03300.pdf]
```

### 为什么 GRPO 正在重塑后训练格局？
1. **显存开销缩减近半**：彻底砍掉了占用极其巨大的 Critic（价值网络）显存；
2. **逻辑客观任务上的确定性奖励**：在数学解题和代码编写中，答案对不对是客观的，用编译器和单元测试就能给出 100% 可信的“规则奖励”，彻底告别了人工主观打分裁判（Reward Model）的幻觉欺骗（Reward Hacking）。
