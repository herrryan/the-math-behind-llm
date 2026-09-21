# 03. 架构演进实录：现代主流大模型组件进化路线图

现代大语言模型（如 LLaMA-3、Mistral、Qwen-2.5、DeepSeek-V3）虽然依然自称属于 Transformer 家族，但其内部的每一个关键零件，都已经历了多轮翻天覆地的优胜劣汰迭代。
本章为你梳理各核心组件的工业进化脉络，并附带所有突破性论文的官方 **PDF 链接**。

---

## 现代大模型标准架构演进总览

```
  组件类别       原始 Transformer (2017)             现代工业标准 (2024-2025)
 ───────────────────────────────────────────────────────────────────────────
  注意力机制 ──► Vanilla MHA (多头注意力)     ──► GQA (Llama 3) / MLA (DeepSeek-V3)
  位置编码   ──► Sinusoidal (绝对正余弦)      ──► RoPE (旋转位置编码) + YaRN
  层归一化   ──► Post-LN (后置 LayerNorm)     ──► Pre-RMSNorm (前置均方根归一化)
  前馈激活   ──► ReLU (单通道线性阈值)        ──► SwiGLU (门控双通道自适应)
  计算拓扑   ──► Dense (全参数密集激活)       ──► Fine-Grained MoE (细粒度混合专家)
 ───────────────────────────────────────────────────────────────────────────
```

---

## 1. 注意力机制演化：从 MHA 到 GQA，再到 MLA

注意力机制的演化史，本质上是**“围剿推理阶段 KV Cache（键值缓存）显存占用”**的壮烈史诗。

### 第一代：原生多头注意力（Vanilla MHA）
- **代表出处**：*Attention Is All You Need* (Vaswani et al., 2017) [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)] · [[PDF](https://arxiv.org/pdf/1706.03762.pdf)]
- **痛点**：在多用户高并发推理时，每个 token 生成都要缓存所有层的 Key 和 Value。对于 70B 模型，仅仅存 KV Cache 就能在短时间内把 80GB 的 A100 显存彻底吃爆！

### 第二代：多查询注意力（MQA）与分组查询注意力（GQA）
- **MQA 论文**：*Fast Transformer Decoding: One Write-Head is All You Need* (Shazeer, 2019) [[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)] · [[PDF](https://arxiv.org/pdf/1911.02150.pdf)]
  所有注意力头共享同一套 Key 和 Value，KV 缓存暴降为原来的 $1/H$，但模型精度受到轻微损伤。
- **GQA 论文**：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023) [[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)] · [[PDF](https://arxiv.org/pdf/2305.13245.pdf)]
  折中智慧（Llama-2/3 标配）：将注意力头分为 8 个小组，每组内部共享一套 Key 和 Value。在几乎零精度损失的前提下，将 KV Cache 显存占用压低了 8 倍！

### 第三代：多头潜在注意力（MLA，DeepSeek 终极杀手锏）
- **代表论文**：*DeepSeek-V2 Technical Report* (2024) [[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)] · [[PDF](https://arxiv.org/pdf/2405.04434.pdf)]
- **物理机制**：不再简单粗暴地按头分组，而是使用**低秩矩阵投影压缩**：
  在把 Key 和 Value 存入显存之前，先将其压缩为一个极其短小的“潜在向量（Latent Vector）”；
  在真正计算点积时，利用矩阵结合律，直接在压缩空间内完成变换与旋转！
- **战果**：将推理显存占用压缩到了原始 MHA 的 **不到六分之一**，为百万长文本高并发吞吐奠定了坚实基础。

---

## 2. 位置编码演化：从绝对刻度到相对旋转罗盘

- **RoPE 代表论文**：*RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021) [[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)] · [[PDF](https://arxiv.org/pdf/2104.09864.pdf)]
- **YaRN 长文本外推论文**：*YaRN: Efficient Context Window Extension of Large Language Models* (Peng et al., 2023) [[arXiv:2309.00071](https://arxiv.org/abs/2309.00071)] · [[PDF](https://arxiv.org/pdf/2309.00071.pdf)]
- **为什么 RoPE 一统天下**：
  1. 形式优雅，完全通过二维复数旋转夹角差刻画相对距离；
  2. 随着距离增加，旋转引起的相对衰减自然符合自然语言近密远疏的规律；
  3. 配合 YaRN 外推算法，能以极小的训练代价将上下文窗口从 4K 轻松外推至 128K 乃至 1M。

---

## 3. 归一化与激活函数：从 Post-LN/ReLU 到 Pre-RMSNorm/SwiGLU

- **RMSNorm 代表论文**：*Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019) [[arXiv:1910.07467](https://arxiv.org/abs/1910.07467)] · [[PDF](https://arxiv.org/pdf/1910.07467.pdf)]
  发现传统 LayerNorm 中耗费大量计算的“减去均值中心化”对稳定性毫无贡献，仅保留均方根缩放，计算吞吐大幅提升。
- **SwiGLU 代表论文**：*GLU Variants Improve Transformer* (Noam Shazeer, Google, 2020) [[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)] · [[PDF](https://arxiv.org/pdf/2002.05202.pdf)]
  用 Swish 激活构筑双通道自适应门控，成为现代百亿千亿大模型的标准前馈层。

---

## 4. 拓扑结构演化：从稠密走向细粒度混合专家（MoE）

- **早期粗粒度 MoE**：*Switch Transformers: Scaling to Trillion Parameter Models* (Fedus et al., 2021) [[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)] · [[PDF](https://arxiv.org/pdf/2101.03961.pdf)]
- **现代细粒度 MoE 标杆**：*DeepSeekMoE: Towards Ultimate Expertise in Mixture-of-Experts Language Models* (Dai et al., 2024) [[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)] · [[PDF](https://arxiv.org/pdf/2401.06066.pdf)]
  将专家切得更细（如 256 专家激活 8 个），并专门设置固定共享专家兜底常识，配合无辅助损失自适应路由，激活 37B 算力却达到 671B 稠密模型的智力！
