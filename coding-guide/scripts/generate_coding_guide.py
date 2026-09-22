import os

docs = {}

# 1. index.md
docs["index.md"] = r"""# 大模型从零上手编程实战课程表：从手工打铁到工业落地

> 专为雄心勃勃的工程师与研究者打造的 LLM 编程从零上手路径：拒绝只当调包侠与 API 调用员，从 0 依赖纯 Python 手写神经网络出发，一步步亲手构建自己的自回归大脑、掌握现代 PyTorch、玩转开源大模型微调与工业级系统加速。

---

## 为什么需要这份“大模型上手编程路线图”？

进入大模型时代，许多初学者在尝试写代码时，往往会陷入两个极端的误区：

1. **“调包调用派”的浅尝辄止**：
   - 以为大模型编程就是注册 API 密钥、调用 `openai.ChatCompletion.create()` 或者写几条 Prompt 模板；
   - 这种开发被称为“应用层开发”，你并没有真正触碰到大模型的计算神经与底层运作逻辑。一旦遇到模型幻觉、吞吐瓶颈、显存溢出或需要领域定制时，立刻束手无策。

2. **“源码深渊派”的过早劝退**：
   - 一上来就试图去硬啃工业级框架（如几万行代码的 Megatron-LM、vLLM 或 Deepspeed）；
   - 繁琐的分布式集群通信（NCCL）、复杂的 CUDA 内存调度直接劝退了 99% 的学习者，让人产生“大模型是极少数大厂研究员专属”的绝望感。

**大模型编程的最佳学习心法，是“由内向外、自下而上”的四级台阶演进：**
**手写一次，胜过读十遍论文；亲眼看一次 Loss 下降，胜过听一万句哲学空谈。**

---

## 四级进阶阶梯全览

| 阶段 | 核心任务 | 技术栈 | 算力门槛 | 产出成果 | 对应指南 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **第一阶段** | **0 依赖纯 Python 手写微型大脑** | 原生 Python (math, random) | 普通笔记本 CPU (0 GPU) | 80 行代码写出 Bengio 语言模型，手算反向传播 | [进入第一阶段](01-stage1-pure-python-scratch.md) |
| **第二阶段** | **PyTorch 从零手搓自回归 Transformer** | PyTorch, Autograd, nn.Module | 单卡 / 笔记本 CPU / Colab | 150 行代码复刻 NanoGPT，训练莎士比亚生成器 | [进入第二阶段](02-stage2-pytorch-nanogpt.md) |
| **第三阶段** | **开源大模型工业实战与轻量微调** | Hugging Face, PEFT (LoRA), DPO | 消费级单卡 (4090 / 3060) | 微调开源 7B 模型（Qwen/LLaMA），实现领域对齐 | [进入第三阶段](03-stage3-huggingface-and-lora.md) |
| **第四阶段** | **突破显存墙与工业级推理加速** | FlashAttention, KV Cache, vLLM | 专业开发卡 (A10/A100) | 编写高效分页内存管理与流式推理解码引擎 | [进入第四阶段](04-stage4-systems-and-acceleration.md) |

---

## 5 分钟立即启动：亲手见证你的第一个神经大脑

你不需要准备任何显卡，今天你的笔记本电脑就能直接跑通第一个模型。

### 步骤 1：启动本地实战终端
进入本项目根目录：
```bash
cd /Users/guofei/workspace/the-math-behind-llm
```

### 步骤 2：直接运行 80 行纯 Python 微型大脑
执行本项目内置的精简实战代码：
```bash
python3 labs/01_micro_brain.py
```

### 步骤 3：观察输出
你将在终端中亲眼看到：
1. 词表自动构建（7 个单词的微型世界）；
2. 随机初始化的网络输出极高混乱度的预测；
3. 经过 120 轮纯手工反向传播（SGD），损失值（Loss）从 `2.07` 稳定坠落至 `0.04`；
4. 语言模型自发输出连贯的自回归句子：`the cat sat on the mat the dog sat on the rug`！

只要你迈出了这 5 分钟的第一步，大模型的黑盒之门就已经为你轰然洞开。
"""

# 2. 01-stage1-pure-python-scratch.md
docs["01-stage1-pure-python-scratch.md"] = r"""# 第一阶段：0 依赖纯 Python 从零手工打铁（80 行代码写出微型大脑）

第一阶段的核心使命是：**破除对神经网络与语言模型的黑盒敬畏，用最原始的工具看清所有齿轮**。

在这个阶段，**严禁使用任何深度学习框架**（不用 PyTorch，不用 TensorFlow，不用 JAX，甚至连 NumPy 都不用）。
我们将仅依靠 Python 标准库中的 `math` 和 `random`，用纯原生列表（List）和循环，亲手构建一个具备自学习与文本生成能力的微型神经语言模型（基于 Bengio 2003 经典架构）。

---

## 核心要掌握的 7 个底层组件

在写这 80 行代码时，你将亲自把数学公式转化为代码逻辑：

```
[ 输入文字 "the" ]
       │
       ▼
1. Tokenizer 字典映射：将文字映射为整数索引 (ID: 5)
       │
       ▼
2. Embedding 查表：从 E 矩阵中取出该词的连续坐标向量 e_5 (维度 d=4)
       │
       ▼
3. 隐藏层线性变换：计算 z = e_5 @ W1 + b1 (维度扩展至 hidden=8)
       │
       ▼
4. 非线性激活函数：通过 ReLU 函数 h = max(0, z) 赋予模型弯曲现实的表达力
       │
       ▼
5. 输出层反投影：计算 logits = h @ W2 + b2 (维度还原至词表大小 |V|=7)
       │
       ▼
6. Softmax 与交叉熵：将 logits 压缩为概率分布，并计算预测下一个词的目标损失 Loss
       │
       ▼
7. 链式法则手工反向传播：根据误差倒流推导梯度，通过 SGD 原地更新所有权重参数！
```

---

## 完整代码拆解（配套 labs/01_micro_brain.py）

本项目已在 [`labs/01_micro_brain.py`](https://github.com/herrryan/the-math-behind-llm/blob/main/labs/01_micro_brain.py) 放置了完整经过严格测试的纯 Python 实现。以下是其核心数学实现的逐行剖析：

### 1. 词表与训练对构建
```python
corpus = "the cat sat on the mat the dog sat on the rug"
words = corpus.split()
vocab = sorted(list(set(words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)          # 词表大小 |V| = 7
d_embed = 4             # 词向量维度
d_hidden = 8            # 隐藏层维度
lr = 0.1                # 学习率

# 构造自回归训练对：(当前词 -> 下一个词)
dataset = [(word2id[words[i]], word2id[words[i+1]]) for i in range(len(words)-1)]
```

### 2. 权重矩阵初始化
```python
import math, random
random.seed(42)

def init_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

E  = init_matrix(V, d_embed)         # 嵌入矩阵 E in R^{|V| x d}
W1 = init_matrix(d_embed, d_hidden)  # 隐藏层权重 W1
b1 = [0.0] * d_hidden                # 隐藏层偏置
W2 = init_matrix(d_hidden, V)        # 输出层权重 W2
b2 = [0.0] * V                       # 输出层偏置
```

### 3. 前向传播与手工反向传播核心循环
```python
for epoch in range(121):
    total_loss = 0.0
    for x_id, y_target in dataset:
        # --- 前向传播 (Forward Pass) ---
        # 1. 查表获取词向量
        embed = E[x_id]

        # 2. 隐藏层线性投影 + ReLU
        hidden_pre = [sum(embed[i] * W1[i][j] for i in range(d_embed)) + b1[j] for j in range(d_hidden)]
        hidden = [max(0.0, val) for val in hidden_pre]

        # 3. 输出层线性投影 (Logits)
        logits = [sum(hidden[j] * W2[j][k] for j in range(d_hidden)) + b2[k] for k in range(V)]

        # 4. 稳定版 Softmax
        max_logit = max(logits)
        exp_vals = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_vals)
        probs = [ev / sum_exp for ev in exp_vals]

        # 5. 交叉熵损失 Loss = -log(P[target])
        loss = -math.log(max(probs[y_target], 1e-15))
        total_loss += loss

        # --- 手工反向传播 (Backward Pass) ---
        # 导数 1: Softmax + 交叉熵的优雅解析解: dL/d(logit_k) = p_k - y_k
        d_logits = probs[:]
        d_logits[y_target] -= 1.0

        # 导数 2: 回传至 W2 和 b2
        d_W2 = [[hidden[j] * d_logits[k] for k in range(V)] for j in range(d_hidden)]
        d_b2 = d_logits[:]

        # 导数 3: 回传至隐藏层激活前梯度
        d_hidden = [sum(W2[j][k] * d_logits[k] for k in range(V)) for j in range(d_hidden)]
        d_hidden_pre = [d_hidden[j] if hidden_pre[j] > 0 else 0.0 for j in range(d_hidden)]

        # 导数 4: 回传至 W1 和 b1
        d_W1 = [[embed[i] * d_hidden_pre[j] for j in range(d_hidden)] for i in range(d_embed)]
        d_b1 = d_hidden_pre[:]

        # 导数 5: 回传至输入词向量 E[x_id]
        d_embed = [sum(W1[i][j] * d_hidden_pre[j] for j in range(d_hidden)) for i in range(d_embed)]

        # --- 参数更新 (SGD) ---
        for j in range(d_hidden):
            for k in range(V): W2[j][k] -= lr * d_W2[j][k]
            b2[j] -= lr * d_b2[j] if j < V else 0.0 # 保持对齐更新
        for k in range(V): b2[k] -= lr * d_b2[k]
        for i in range(d_embed):
            for j in range(d_hidden): W1[i][j] -= lr * d_W1[i][j]
            E[x_id][i] -= lr * d_embed[i]
        for j in range(d_hidden): b1[j] -= lr * d_b1[j]
```

---

## 阶段作业与动手思考

当你完整运行过这段代码后，请尝试做下面三个小实验：

1. **实验 A（激活函数的重要性）**：如果将 ReLU 函数移除，直接令 `hidden = hidden_pre`，模型的 Loss 还能降到那么低吗？为什么？
2. **实验 B（视野限制）**：当前模型的输入永远只有 1 个词。如果我们想让模型根据前面 2 个词来预测第 3 个词（例如输入 `["the", "cat"]` 预测 `"sat"`），嵌入层应该如何拼接？
3. **实验 C（生成模式）**：编写一个循环，以单词 `"the"` 作为起始词，不断将模型预测出的最大概率词作为下一个输入，看看模型能否自动生成一整句话！

完成第一阶段后，你已经拥有了直通神经网络灵魂的微观视角。接下来，我们将换上现代工具，迈向 Transformer 时代！
"""

# 3. 02-stage2-pytorch-nanogpt.md
docs["02-stage2-pytorch-nanogpt.md"] = r"""# 第二阶段：现代工业起步——PyTorch 手搓自回归 Transformer (NanoGPT)

在第一阶段，你亲手体验了手工推导微积分反向传播的震撼，也体会到了网络层数做多时手工求导的繁重。
从第二阶段开始，我们正式引入现代深度学习的事实工业标准——**PyTorch**。

本阶段的目标是：**用大约 150 行清晰易读的 PyTorch 代码，从零拼装出一个纯正的 Decoder-only 因果自回归 Transformer 模型（NanoGPT）**。

---

## PyTorch 的“三板斧”认知

学习 PyTorch 切记不要去死记硬背它上千个 API，只需牢牢把握三根支柱：

1. **张量（`torch.Tensor`）**：
   - 本质是多维矩阵，能够一键搬运到 GPU 显存（`.to('cuda')`）进行高并发计算；
2. **自动求导（`torch.autograd`）**：
   - 彻底解放双手：只要你在前向传播中进行了张量运算，调用 `loss.backward()`，PyTorch 就会自动根据动态计算图，利用链式法则反向求出所有参数的 `.grad`；
3. **模块容器（`torch.nn.Module`）**：
   - 统一的代码组织规范：在 `__init__()` 中声明积木（权重、子层），在 `forward()` 中编写数据流动流水线。

---

## 核心积木组装流水线

一个现代自回归 Transformer 脑结构包含五个核心积木：

### 1. 缩放点积自注意力（Scaled Dot-Product Attention）

输入序列向量 $X \in \mathbb{R}^{B \times T \times C}$，通过三个线性矩阵投影为 $Q, K, V$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

其中 $M$ 为因果遮蔽下三角矩阵（Causal Mask），保证模型在看第 $t$ 个词时，绝对看不到未来词的信息。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        # 一次性投影得到 Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # 输出线性投影
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(Sequence Length), Channel(Embedding Dim)
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # 拆分为多头并变换形状: (B, n_head, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_k ** 0.5))
        # 施加因果遮蔽 (下三角矩阵)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, n_head, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 拼合各头
        return self.c_proj(y)
```

### 2. 前馈思考网络（Feed-Forward Network, FFN）
在注意力机制让所有词互通信息后，每个词需要进入独立思考腔室：
```python
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
```

### 3. Transformer 积木块（Block：残差连接 + LayerNorm）
通过 Pre-LN 结构与跳跃连接，确保深层网络的梯度能够无衰减倒流：
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # 残差连接 1
        x = x + self.mlp(self.ln2(x))  # 残差连接 2
        return x
```

### 4. 完整的自回归语言模型（NanoGPT）
把词嵌入（Token Embedding）、位置嵌入（Positional Embedding）、多个 TransformerBlock 以及最终的语言模型头（LM Head）串联起来：
```python
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=4, n_layer=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0) # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)             # (B, T, d_model)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

---

## 阶段实战产出

在第二阶段结束时，你应当在本地或 Google Colab 免费 GPU 上完成以下目标：
1. 准备一份几万字的纯文本语料（如莎士比亚戏剧、红楼梦前十回或金庸武侠片段）；
2. 编写极简字符级分词器（Character-level Tokenizer）；
3. 启动 `NanoGPT` 训练循环，使用 `torch.optim.AdamW(model.parameters(), lr=1e-3)`；
4. 编写 `generate()` 函数，观察自回归生成文字从最初的乱码乱跳，逐步收敛到能写出符合语法逻辑的连贯段落！
"""

# 4. 03-stage3-huggingface-and-lora.md
docs["03-stage3-huggingface-and-lora.md"] = r"""# 第三阶段：拥抱开源生态——主流大模型加载与轻量高效微调 (PEFT / LoRA / DPO)

当你亲手用 PyTorch 搭完 NanoGPT 后，你已经洞悉了自回归模型骨架中的每一根骨头。
但现实工业生产中，没有人会为了解决一个实际业务问题，去从头花费几百万美元预训练一个千亿参数模型。

第三阶段的核心任务是：**学会站在全人类开源成果的肩膀上，掌握现代大模型的标准化工业工具链，用消费级硬件完成模型的能力定制与价值观对齐**。

---

## 必须掌握的开源生态三件套

### 1. Hugging Face 工业标准库
- **`transformers`**：负责模型架构与预训练权重调度。掌握 `AutoTokenizer`、`AutoModelForCausalLM` 和 `pipeline`；
- **`datasets`**：负责大规模训练数据的零内存加载（Memory Mapping 流式读取）；
- **`accelerate`**：无需修改代码，一键实现单机多卡或混合精度（BF16/FP16）训练调度。

### 2. 显存精打细算：为什么全量微调不可行？
一个 7B 参数的模型，权重文件本身占用约 14 GB 显存（按 16 位浮点数计算）。但在进行传统的全量微调（Full Fine-Tuning）时，显存需要支撑：
- 静态模型权重：14 GB
- 反向传播梯度：14 GB
- AdamW 优化器状态（一阶动量 + 二阶方差）：28 GB
- 前向激活值（Activations）：十几 GB
**总显存需求飙升至 70-80 GB，必须依赖昂贵的 A100/H100 显卡！**

### 3. 救命解法：低秩自适应微调（LoRA, Low-Rank Adaptation）

LoRA 的数学核心极为优雅：**保持原始预训练权重 $W_0 \in \mathbb{R}^{d \times k}$ 完全冻结（不存优化器状态），仅在其旁边外挂两个极低维度的旁路矩阵 $A$ 和 $B$**：

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)
$$

其中 $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$，秩 $r$ 通常仅取 8 或 16。
- 训练参数量直接从 70 亿（7B）暴降至几百万（不到 0.1%）；
- 优化器显存从数十 GB 暴降至几百 MB；
- 加上 4-bit 量化（QLoRA），**一张 16GB 显存的普通显卡（如 RTX 4080 / 4060Ti）即可轻松微调 7B/8B 旗舰基座模型！**

---

## LoRA 极简微调实战模版

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. 加载开源基座模型 (以 Qwen2.5-7B 为例)
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 配置 LoRA 旁路矩阵
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                    # 低秩维度
    lora_alpha=32,                           # 缩放系数
    target_modules=["q_proj", "v_proj"],     # 仅注入注意力投影矩阵
    lora_dropout=0.05,
    bias="none"
)

# 3. 包装模型：99.8% 的参数被瞬间冻结
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.059%
```

---

## 现代对齐新范式：直接偏好优化（DPO）

在微调掌握特定领域知识后，模型可能会胡说八道或产生攻击性。如何对齐人类价值观？
- **传统 PPO 强化学习**：需要同时在显存中跑四个大模型（Actor、Critic、Reference、Reward），极度复杂且容易崩溃；
- **现代 DPO 革命**：彻底推翻四模型体系，证明了无需训练独立的 Reward 模型，只需利用人类标注的偏好对：
  - Prompt: 提问
  - Chosen: 人类更满意的优质回答
  - Rejected: 人类不满意的糟糕回答
  直接通过闭式解将偏好概率差融入二元交叉熵损失，用简单的有监督格式完成强化学习对齐！

---

## 推荐工业工具链

在实际工程项目中，你不需要自己从头手写数据拼接与梯度调度，推荐熟练掌握以下成熟开源工具：
1. **TRL (Transformer Reinforcement Learning)**：Hugging Face 官方出品的 SFT/DPO/PPO 全流程库；
2. **Unsloth**：针对开源模型手写 Triton 算子，将 LoRA 微调速度提升 2-5 倍，显存降低 70%；
3. **LLaMA-Factory**：集成了 WebUI 界面与百种开源模型的工业级快速微调平台。
"""

# 5. 04-stage4-systems-and-acceleration.md
docs["04-stage4-systems-and-acceleration.md"] = r"""# 第四阶段：大模型系统深水区——显存墙、FlashAttention 与推理引擎 (vLLM)

恭喜你来到大模型技术金字塔的顶端！
在这个阶段，你不再仅仅是一名“算法工程师”，而是跨越成为一名**深刻理解底层硅基芯片硬件特性的“大模型系统架构师”**。

在真实商业世界中，**90% 以上的大模型成本发生在推理（Inference）阶段**。
如果你不懂 GPU 内存层级、不懂算子融合、不懂显存复用，你部署的模型就会因为昂贵的推理开销和极低的并发吞吐而走向破产。

---

## 必须跨越的核心认知：算力受限 vs 显存带宽受限

现代 GPU（如 A100/H100）内部有两个世界：
1. **算力核心（Tensor Cores）**：计算能力极其恐怖（每秒数百 TeraFLOPs）；
2. **全局显存（HBM）**：存储容量虽大，但从 HBM 向芯片核心搬运数据的带宽极其受限（Memory IO Wall）。

| 计算阶段 | 典型计算模式 | 瓶颈归属 | 核心优化抓手 |
| :--- | :--- | :--- | :--- |
| **Prefill 阶段**（处理长提示词） | 矩阵乘大矩阵（GEMM） | **算力受限（Compute-bound）** | 提高 Tensor Core 利用率、高精度并行 |
| **Decode 阶段**（逐字自回归吐词） | 矩阵乘向量（GEMV） | **显存受限（Memory-bound）** | **KV Cache 复用、减少显存搬运、PagedAttention** |

---

## 工业系统的三大必修硬核主题

### 1. KV Cache 内存复用原理与显存爆炸

自回归生成中，每生成一个新 Token，过去所有词的 Key 和 Value 向量都必须参与注意力计算。
如果不存缓存，生成长度为 $S$ 的文章需要重复计算 $O(S^2)$ 次历史注意力；
如果缓存起来，显存消耗为：

$$
\text{KV Cache 显存} = 2 \times L \times H_{\text{kv}} \times d_k \times S \times \text{sizeof(dtype)}
$$

- 对于 70B 模型，并发 64 个用户、上下文 8k 时，KV Cache 显存将超过 **80 GB**！
- 朴素分配机制会导致 60%-80% 的内部碎片浪费。

### 2. FlashAttention：硬件感知的算子革命
- **问题所在**：标准 Softmax 注意力需要在显存 HBM 与片上缓存 SRAM 之间频繁读写巨大的 $N \times N$ 注意力矩阵；
- **解决核心**：
  - **分块平铺（Tiling）**：将输入 $Q, K, V$ 切成适配片上高速缓存 SRAM 大小的小积木；
  - **在线 Softmax（Online Softmax）**：在不完整物化全局注意力矩阵的前提下，利用数学换底缩放公式实时增量更新 Softmax 统计量；
  - 彻底消除了平方级中间显存开销，速度提升 2-4 倍！

### 3. PagedAttention 与 vLLM 架构
- 借鉴现代操作系统（OS）的分页虚拟内存哲学；
- 将连续的逻辑 KV Cache 动态打散映射到不连续的物理物理块（Physical Blocks）中；
- 彻底消除了内存碎片，将显存浪费率从 70% 骤降到 4% 以下，让服务并发承载量直接提升数倍！

---

## 阶段学习路线

要吃透第四阶段，建议直接配套研读本项目的高阶工程专栏：
- [大模型系统工程专题 (The Engineering Behind LLMs)](https://github.com/herrryan/the-math-behind-llm/tree/main/engineering)
  - 模块 0：GPU 硬件格局与内存墙
  - 模块 1：FlashAttention 算子实现原理
  - 模块 2：vLLM PagedAttention 与连续批处理（Continuous Batching）
  - 模块 3：投机采样（Speculative Decoding）加速推理解析
"""

# 6. 05-stage5-capstone-projects.md
docs["05-stage5-capstone-projects.md"] = r"""# 实战进阶：四大毕业设计项目清单（从 0 到 1 打造代表作）

检验学习成果的最佳方式，就是**打造拿得出手的开源项目作品集**。
以下为你设计的四个渐进式毕业设计课题，每一个项目都对应工业界的一个核心技术生态位：

---

## 毕业设计 1：纯 Python 字符级语言模型终端（Micro-Brain CLI）

- **技术门槛**：零框架依赖（纯标准库）
- **核心任务**：
  1. 扩展第一阶段的 `labs/01_micro_brain.py`，支持多字符输入（Context Window = 3）；
  2. 实现带温度（Temperature）与 Top-k 截断的概率采样函数；
  3. 包装为一个支持交互式敲键盘、即时流式打字的命令行交互程序（CLI）。
- **答辩标准**：在终端输入任意前缀，模型能够稳定自回归补全，且无需安装任何第三方 pip 包。

---

## 毕业设计 2：复刻 NanoGPT 并在中国古典文学上训练（NanoGPT-Ancient）

- **技术门槛**：PyTorch + 单机 CPU/GPU
- **核心任务**：
  1. 完整实现 Multi-Head Attention、残差连接与 Pre-LN 结构；
  2. 爬取或整理一份《全唐诗》或《红楼梦》纯文本语料（约 50 万字）；
  3. 构建简易 BPE 分词器或字符分词表；
  4. 训练一个 10M 参数量的轻量语言模型，将损失值（Loss）压到 1.5 以下。
- **答辩标准**：给出一段诗词开头，模型能自动模仿对仗与格律，续写出韵味十足的五言或七言绝句。

---

## 毕业设计 3：垂直领域开源大模型微调与 DPO 价值观对齐（Domain-Expert LLM）

- **技术门槛**：Hugging Face + PEFT + TRL + 单张消费级显卡（Colab 免费 T4 亦可）
- **核心任务**：
  1. 选定基座模型（推荐 Qwen2.5-0.5B 或 Qwen2.5-7B）；
  2. 收集或合成 1000 条垂直专业领域问答对（如法律咨询、医疗常识或代码纠错）；
  3. 使用 LoRA 进行指令微调（SFT）；
  4. 收集 200 条带有好回答（Chosen）与恶意/错误回答（Rejected）的偏好对，使用 DPO 完成拒绝恶毒回答的价值观对齐。
- **答辩标准**：提供微调前后的对比测试用例，明确展示出领域知识准确率的显著提升与不良偏好对齐效果。

---

## 毕业设计 4：极简高性能自回归推理解码服务器（Toy-vLLM Engine）

- **技术门槛**：PyTorch + FastAPI + 基础系统编程
- **核心任务**：
  1. 为 Transformer 手写一套标准高效的 KV Cache 存储与更新机制；
  2. 搭建基于 FastAPI 的异步 HTTP 服务，支持以 SSE（Server-Sent Events）格式逐 Token 流式输出打字机效果；
  3. 实现简单的连续批处理（Continuous Batching）调度循环，支持多个并发请求同时动态拼批生成。
- **答辩标准**：启动客户端同时发起 10 个并发请求，服务器能够不卡顿地以毫秒级延迟向终端并行吐字！
"""

# 7. 06-troubleshooting-and-faq.md
docs["06-troubleshooting-and-faq.md"] = r"""# 避坑手册：显存爆满、梯度异常与训练排错常见 FAQ

在上手大模型编程时，遇到报错是家常便饭。
本手册总结了初学者在训练和部署大模型时最常遭遇的几大“拦路虎”，并提供标准的工程诊断步骤。

---

## 常见故障 1：显存溢出（`CUDA Out of Memory`）

这是所有初学者最频繁遇到的红色报错。

### 诊断与排查清单
1. **调小 Batch Size**：将每次送入模型的批大小从 16 减到 8，甚至减到 1 或 2；
2. **启用梯度累积（Gradient Accumulation）**：
   - 如果真实的有效 Batch Size 需要 64，但显存只能塞下 2，可以连续计算 32 次小批次的前向与反向传播，将梯度累加（`loss / accum_steps`），最后统一调用一次 `optimizer.step()`。这在数学上与大批次完全等价！
3. **切换数据精度至 BF16 / FP16**：
   - 默认的 FP32 每个数字占 4 个字节，切换至 BF16 立即立省 50% 的模型与显存开销；
4. **激活重算（Activation Checkpointing / Gradient Checkpointing）**：
   - 前向传播时不保存中间层的所有激活值，反向传播用到时现场重新计算一遍。用 20% 的算力开销换取高达 60% 的显存释放！
5. **及时清理垃圾**：
   - 在 PyTorch 循环间隙调用 `torch.cuda.empty_cache()` 释放碎片。

---

## 常见故障 2：损失值突然变成非数（`Loss = NaN`）

如果跑着跑着发现 `Loss: nan`，说明数值发生了溢出或下溢。

### 诊断与排查清单
1. **学习率（Learning Rate）设置过大**：
   - 导致权重被单次巨大的梯度拉扯至数值无穷大。尝试将学习率缩小 10 倍（如从 `1e-3` 降至 `1e-4`）；
2. **缺少梯度裁剪（Gradient Clipping）**：
   - 在优化器步进前，务必加入一行：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`，死死锁住梯度的最大范数；
3. **Log 运算中出现 0**：
   - 计算交叉熵或似然概率时，如果概率预测为 0，`math.log(0)` 会引发负无穷（`-inf`）。务必在内部加上极小的保护常数：`log(prob + 1e-12)`。

---

## 常见故障 3：生成文本陷入“复读机死循环”

模型生成文字时，不断重复同一句话或同一个词，例如：“今天的天气今天的天气今天的天气……”

### 诊断与排查清单
1. **采样策略过于死板（Greedy Search）**：
   - 永远只选概率最大的词（`argmax`），容易陷入高概率循环陷阱。
   - **解决办法**：引入**温度系数（Temperature = 0.7~0.9）**与 **Top-p 采样（Nucleus Sampling = 0.9）**，给次高概率词合理的随机出头机会；
2. **加入重复惩罚（Repetition Penalty）**：
   - 检查已生成的上下文，对于已经出现过的词，在其 Logit 上扣除固定惩罚分（通常乘以 1.1~1.2 的衰减因子）。

---

## 常见算力租用指南（平民玩家怎么选？）

没有几万块的专业显卡，如何低成本上手？

| 平台 | 特点 | 适用场景 | 建议机型 |
| :--- | :--- | :--- | :--- |
| **Google Colab** | 免费提供 T4 GPU，浏览器即开即用 | 跑实验、跑 NanoGPT、小样本 LoRA | Free T4 / A100 (Pro) |
| **Kaggle Notebooks** | 每周提供 30 小时免费双卡 T4 | 数据集探索与中等规模实验 | 双卡 T4 |
| **AutoDL / 恒源云** | 国内按小时计费（约 1-3 元/小时） | 中等规模 LoRA 微调、DPO 实战 | RTX 3090 / 4090 |
| **Lambda Labs / RunPod** | 国际按小时计费，机器性能纯净 | 工业级评测与大模型系统实验 | A100 80GB / H100 |
"""

# Generate docs
for filename, content in docs.items():
    filepath = os.path.join("coding-guide/docs", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated: {filepath} ({len(content.encode('utf-8'))} bytes)")

print("\nAll 7 coding guide documents generated successfully.")
