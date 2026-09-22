# 第二阶段：现代工业起步——PyTorch 手搓自回归 Transformer (NanoGPT)

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
