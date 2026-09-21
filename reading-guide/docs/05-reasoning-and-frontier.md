# 05. 前沿推理新纪元：测试时计算扩展与纯强化学习反思涌现

当业界还在忧虑“互联网人类高质量文本即将耗尽”、“预训练 Scaling Law 是否遭遇天花板”时，OpenAI o1 与 DeepSeek-R1 开辟了全新赛道——**测试时计算扩展（Test-Time Compute Scaling Law）与推理模型新范式**。每篇前沿文献均附带官方 **PDF 直达链接**。

---

## 1. 经典必读论文：DeepSeek-R1 的纯强化学习奇迹

- **论文标题**：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- **文献链接**：[[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)] · [[PDF 官方直达](https://arxiv.org/pdf/2501.12948.pdf)]
- **核心里程碑拆解**：
  - **R1-Zero**：拿最原始的 DeepSeek-V3-Base 基座模型，**完全不喂任何人工写好的问答对或思维链（Zero Cold-start SFT）**；
  - 仅给两道极简规则判卷：规则 1 格式标签，规则 2 标答/测试用例客观判分；
  - **顿悟时刻（Aha Moment）**：在没有人类教授的情况下，模型自发学会了在思维链中进行反思重算：“等等，让我重新核算刚才的方程……啊，之前忽略了负号！”，见证了硅基智能对自省机制的独立觉醒。

---

## 2. 测试时计算扩展与过程验证核心论文

- **推理扩展代表作 A**：*Large Language Monkeys: Scaling Inference Compute with Verifiers* (Brown et al., 2024)
  - 文献链接：[[arXiv:2407.21787](https://arxiv.org/abs/2407.21787)] · [[PDF 官方直达](https://arxiv.org/pdf/2407.21787.pdf)]
  - 阐述了在测试时通过重复采样与验证器（Verifiers）扩展计算量，模型性能如何呈现指数级跃升。
- **推理思考代表作 B**：*Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking* (Zelikman et al., 2024)
  - 文献链接：[[arXiv:2403.09629](https://arxiv.org/abs/2403.09629)] · [[PDF 官方直达](https://arxiv.org/pdf/2403.09629.pdf)]
  - 探索了大模型在吐出下一个字之前，在静默状态下自发生成多条内部思考路径的机制。
- **过程奖励模型 PRM 代表作**：*Let's Verify Step by Step* (Lightman et al., OpenAI, 2023)
  - 文献链接：[[arXiv:2305.20050](https://arxiv.org/abs/2305.20050)] · [[PDF 官方直达](https://arxiv.org/pdf/2305.20050.pdf)]
  - 提出不仅对最终答案打分，更对解题推理的每一个逻辑步骤进行逐步监督与评分，奠定了严谨数学推理的基础。
