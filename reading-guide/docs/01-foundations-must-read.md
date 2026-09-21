# 01. 必读基石篇：奠定当代 LLM 体系的传世经典

本篇精选的 7 篇论文，是现代大语言模型演进史上真正的里程碑。无论技术如何演化，现代主流大模型（LLaMA、Mistral、Qwen、DeepSeek 等）的骨架全部基于这些论文所确立的底层法则。

---

## 论文 1：现代 AI 的创世纪

- **论文标题**：*Attention Is All You Need*
- **作者团队**：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Research)
- **发表年份**：2017 年
- **精读推荐指数**：五星（全篇必须逐字精读）
- **为什么它不可替代**：
  它首次彻底废弃了串行循环（RNN/LSTM）与卷积（CNN），提出了基于自注意力（Self-Attention）与多头注意力（MHA）的纯并行架构。
- **重点精读段落**：
  - Section 3.2: Attention（Scaled Dot-Product Attention 与 Multi-Head Attention 的物理定义与维度推导）；
  - Section 3.3: Position-wise Feed-Forward Networks（前馈网络的维度膨胀与收缩设计）；
  - Section 3.5: Positional Encoding（早期正余弦绝对位置编码的设计逻辑，用于对比后续的 RoPE）。
- **当今现状**：虽然原始论文采用的是 Encoder-Decoder 架构，且采用了现在已被淘汰的 Post-LN 与正余弦位置编码，但其核心的 QKV 机制与残差连接骨架至今未动摇。

---

## 论文 2：预训练的物理定律——缩放定律

- **论文标题**：*Scaling Laws for Neural Language Models* (Kaplan et al., 2020) 与 *Training Compute-Optimal Large Language Models (Chinchilla)* (Hoffmann et al., DeepMind, 2022)
- **精读推荐指数**：五星（工程立项与预算评估的圣经）
- **核心贡献与认知颠覆**：
  - **Kaplan 2020 (OpenAI)**：首次定量揭示了模型性能（交叉熵 Loss）与计算量 $C$、参数量 $N$、数据量 $D$ 之间呈现严格的幂律（Power-law）关系，证明了无脑堆算力和参数能持续带来智能收益；
  - **Chinchilla 2022 (DeepMind)**：修正了 OpenAI 的参数/数据配比错误，指出此前的大模型（如 175B 的 GPT-3）严重“参数过大、数据喂养不足”（Undertrained），证明最佳配比应该是**参数量与训练 Token 数量以 1:1 的等比例同时扩张**（约为 1 个参数配 20 个 Token，而现代开源甚至推向 1:200）。
- **必须掌握的核心结论**：
  不要盲目做大参数，把充分的高质量 Token 灌入体量合理的模型中，是在固定算力预算下榨取最大性能的核心法则。

---

## 论文 3：通用智能与涌现能力的起点

- **论文标题**：*Language Models are Few-Shot Learners* (GPT-3)
- **作者团队**：Tom B. Brown et al. (OpenAI)
- **发表年份**：2020 年
- **精读推荐指数**：四星半（理解大模型思维模式必读）
- **核心贡献**：
  - 正式证明了纯 Decoder-only 自回归大模型在参数扩展到 175B 时，无需针对下游任务更新任何参数梯度，仅仅依靠上下文提示词（In-Context Learning），就能展现出惊人的零样本（Zero-shot）与少样本（Few-shot）泛化能力；
  - 终结了传统自然语言处理中“预训练 + 针对具体任务微调（Task-specific Fine-tuning）”的分裂范式，将 NLP 统一为“提示词对话”单一大道。

---

## 论文 4：硬件感知注意力的工业革命

- **论文标题**：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022) 与 *FlashAttention-2* (2023)
- **精读推荐指数**：五星（做系统与工程的必读杰作）
- **核心贡献**：
  - 彻底改变了深度学习算法设计理念：**算法设计不能只看浮点计算次数（FLOPs），更要看芯片内存传输开销（Memory IO）**；
  - 揭示了标准注意力之所以慢且吃显存，是因为频繁在 GPU 高带宽显存（HBM）与片上高速缓存（SRAM）之间读写那个巨大的 $N 	imes N$ 注意力矩阵；
  - 提出了基于分块平铺（Tiling）与在线 Softmax（Online Softmax）算法，将计算完全锁在片上 SRAM 中，不仅实现精确注意力零精度损失，更将训练速度提升数倍、长文本显存开销从平方级骤降至线性。

---

## 论文 5：现代开源大模型的工程基石

- **论文标题**：*Llama: Open and Efficient Foundation Language Models* (Llama 1 / 2) 与 *The Llama 3 Herd of Models* (Meta, 2023-2024)
- **精读推荐指数**：五星（现代大模型落地的标准答案）
- **核心贡献**：
  - 确立了现代大模型架构的“黄金四件套”事实标准：**Pre-RMSNorm + SwiGLU + RoPE (旋转位置编码) + GQA (分组查询注意力)**；
  - Llama 3 更是公开了 15T+ 海量数据预训练与高质量后训练（合成数据配比、长文本退火、偏好对齐迭代）的全部工业细节，是当今所有大模型训练实践的教科书。

---

## 论文 6：后训练偏好对齐的极简革命

- **论文标题**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (DPO)
- **作者团队**：Rafael Rafailov et al. (Stanford University)
- **发表年份**：2023 年
- **精读推荐指数**：四星半
- **核心贡献**：
  - 颠覆了传统 RLHF 中需要同时维持语言模型、奖励模型（Reward Model）、价值网络（Critic）的脆弱复杂四模型管线；
  - 从数学上巧妙证明：语言模型本身的隐式概率，在代数上完全等价于奖励模型。直接通过人类偏好对比数据（胜出回复 vs 落败回复）计算交叉熵二分类损失，即可实现稳定的偏好对齐。

---

## 论文 7：中国开源震撼世界的双子星

- **论文标题**：*DeepSeek-V3 Technical Report* 与 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- **作者团队**：DeepSeek-AI
- **发表年份**：2024 - 2025 年
- **精读推荐指数**：五星（现代最极致的性价比与推理革命）
- **核心贡献**：
  - **V3 架构创新**：提出了 **MLA（多头潜在注意力）**，在推理时将 KV 缓存压缩至原始 MHA 的几分之一；采用细粒度混合专家（DeepSeekMoE，256 专家 + 1 共享专家）与多 Token 预测（MTP），实现了前所未有的显存与计算极致效率；
  - **R1 范式革命**：提出通过极简规则奖励函数（格式 + 答案正确性）配合 **GRPO（群组相对策略优化）**，无需冷启动人工标注数据，直接在纯强化学习环境中激发模型的长思维链推理、自我纠错与顿悟能力。
