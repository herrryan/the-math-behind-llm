# 第一阶段：0 依赖纯 Python 从零手工打铁（80 行代码写出微型大脑）

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
