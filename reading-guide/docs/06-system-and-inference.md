# 06. 系统工程与推理解码：冲破硅基内存墙的必读文献

在当代大模型的生命周期中，**99% 的计算成本与商业开销发生在推理部署阶段**。
不懂硬件底层的算法研究员，做出来的模型往往根本无法在服务器上以可控成本跑起来。
本章为你梳理解决 GPU 显存墙、吞吐墙与延迟墙的核心工程杰作，全部提供官方 **PDF 链接**。

---

## 1. 突破显存碎片的必读工业里程碑：vLLM 与 PagedAttention

- **论文标题**：*Efficient Memory Management for Large Language Model Serving with PagedAttention*
- **文献链接**：[[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)] · [[PDF 官方直达](https://arxiv.org/pdf/2309.06180.pdf)]
- **作者团队**：Woosuk Kwon et al. (UC Berkeley / LMSYS, 2023)
- **核心贡献**：
  巧妙借用操作系统**虚拟内存分页管理（Paging）**思想，将 KV Cache 拆分为固定大小的物理块，通过逻辑页表动态映射。几乎实现了零显存碎片浪费，吞吐提升 2~4 倍，成为当今全球主流推理服务引擎的核心地基。

---

## 2. 打破逐字解码延迟的奇思妙想：投机采样（Speculative Decoding）

- **论文标题**：*Fast Inference from Transformers via Speculative Decoding*
- **文献链接**：[[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)] · [[PDF 官方直达](https://arxiv.org/pdf/2211.17192.pdf)]
- **作者团队**：Yaniv Leviathan et al. (Google Research, 2023)
- **运行机制**：
  1. 小模型快速起草 5 个候选词；
  2. 大模型利用强大的并行矩阵计算，在单次前向传播中同时对 5 个候选词进行概率校验；
  3. 基于拒绝采样数学证明，在保证输出分布 100% 毫无变形的前提下，直接带来 2~3 倍的生成端到端加速！

---

## 3. 分布式并行经典体系（千卡训练的交通枢纽）

- **张量与流水线并行**：*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., NVIDIA, 2019)
  - 文献链接：[[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)] · [[PDF 官方直达](https://arxiv.org/pdf/1909.08053.pdf)]
- **状态分片数据并行**：*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., Microsoft, 2020)
  - 文献链接：[[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)] · [[PDF 官方直达](https://arxiv.org/pdf/1910.02054.pdf)]
  - 透彻掌握 ZeRO-1/2/3 显存节省机制（对应 PyTorch FSDP）。
- **百万长序列并行**：*RingAttention with Block Pipelining for Up to Millions of Tokens* (Liu et al., UC Berkeley, 2023)
  - 文献链接：[[arXiv:2310.01889](https://arxiv.org/abs/2310.01889)] · [[PDF 官方直达](https://arxiv.org/pdf/2310.01889.pdf)]

---

## 4. 极致部署量化路线：从 FP16 到 FP8 / INT4

- **SmoothQuant (W8A8)**：*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (Xiao et al., 2023)
  - 文献链接：[[arXiv:2211.10438](https://arxiv.org/abs/2211.10438)] · [[PDF 官方直达](https://arxiv.org/pdf/2211.10438.pdf)]
- **AWQ (4-bit 激活感知量化)**：*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2023)
  - 文献链接：[[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)] · [[PDF 官方直达](https://arxiv.org/pdf/2306.00978.pdf)]
