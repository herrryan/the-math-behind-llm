# 动手实验 07：纯 Python 实现 GRPO 推理引擎（200 行代码复刻自主推理与规则验证）

<fieldset id="evolution" markdown="1">
<legend><strong>Python 大模型演进全景图 &bull; 强化学习与慢思考推理前沿（第 7 阶段 / 共 7 阶段）</strong></legend>

在实验 06 中，我们实现了投机采样与 INT4 量化引擎，攻克了大模型推理中的显存带宽墙。然而，截至目前我们构建的所有模型本质上都在模仿：通过静态交叉熵无脑预测人类写过的历史词元。

作为强化学习篇章的集大成实战，我们用约 200 行零依赖纯 Python 代码（无 PyTorch、无 HuggingFace、无 NumPy）完整复刻 **GRPO 推理引擎**。我们将完整还原 DeepSeek-Math 与 DeepSeek-R1 的核心数学机制：**无评论家群组相对优势归一化（Critic-Free Group Normalization）**、**可验证规则奖励（RLVR）**、**PPO 重要性比率裁剪**以及 **Schulman 非负 KL 散度约束**。

```
[Python 大模型演化路线图 &bull; 7 阶段完整旅程]
[阶段 1]   80 行纯 Python：Bengio 2003 MLP 语言模型（嵌入层、全连接、纯手工反向传播）
        │
        ▼（健忘缺陷：固定 1 个词元的极窄上下文视窗）
[阶段 2]  140 行纯 Python：注意力机制脑（解锁 Q、K、V 投影与因果自注意力掩码）
        │
        ▼（不稳定缺陷：深层网络梯度弥散与数值震荡）
[阶段 3]  220 行纯 Python：现代 Transformer 块（Pre-RMSNorm、残差高速公路与 SwiGLU 门控）
        │
        ▼（采样缺陷：生硬贪婪循环与 O(T^2) 冗余重算）
[阶段 4]  300 行纯 Python：完整交互式 LLM 引擎（KV 缓存加速与核采样策略）
        │
        ▼（并发缺陷：静态批处理导致 60%+ 显存气泡浪费；连续张量导致内存严重碎片化）
[阶段 5]  200 行纯 Python：流式 KV 缓存与持续批处理引擎（分页注意力与迭代级调度调度器）
        │
        ▼（带宽缺陷：串行逐词元读取百 GB 权重的显存带宽墙 O(T)）
[阶段 6]  220 行纯 Python：投机采样解码与 INT4 量化加速引擎
        │
        ▼（模仿缺陷：纯监督学习无法自主发现全新解题路径，无法自发展开多步自我纠错）
[阶段 7 (推理巅峰)] 200 行纯 Python：GRPO 自主推理引擎
        │
        ▼（最终成果：纯自主探索、规则驱动、摆脱评论家的大模型强化学习推理系统！）
```
</fieldset>

---

## 步骤 1：3 岁小孩直觉（自主纠错的互助学习小组）

想象一个四人互助学习小组正在攻克一道很难的数学题：

<figure>
<pre>
[传统配家教的 RLHF 架构：沉重、脆弱、极耗显存]
学生写一个字 ──► 家教猜一个分 ──► 学生再写下一个字
(家教占用 50% 显存，而且面对长推导常常自身判断失误！)

[GRPO：高效自主的互助学习小组]
第 1 步：4 位同学针对这道题同时在草稿纸上独立写出完整推导。
第 2 步：规则裁判直接核对黑板结果：
         同学 1: &lt;think&gt;3*4=12, 12+2=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;  ──► 奖励: 1.5
         同学 2: &lt;think&gt;2+12=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;          ──► 奖励: 1.5
         同学 3: &lt;think&gt;2+3=5, 5*4=20&lt;/think&gt;&lt;answer&gt;20&lt;/answer&gt;    ──► 奖励: 0.5
         同学 4: 14 &lt;eos&gt;                                          ──► 奖励: 1.0
第 3 步：全组平均分 = 1.125。计算标准差并进行组内标准化。
第 4 步：更新模型：同学 1 和 2 获得正向超额优势；同学 3 获得负向惩罚！
         成果：完全不需要任何评论家（Critic）神经网络！
</pre>
<figcaption><strong>图 31b.1：</strong> GRPO 自主学习小组闭环：多路候选生成、规则精准校验、群组相对优势分配，零评论家开销。</figcaption>
</figure>

1. **彻底解放显存**：
   无需维系一个与主模型同样庞大的价值评估模型，同批次样本的均值与方差构成了极其精确且无偏的动态基线。
2. **绝对客观的规则判断**：
   编译器与数学符号校验器永远不会产生幻觉：思考标签是否闭合、答案是否等于 14，判定结果是非分明。
3. **防止走火入魔的数学安全绳**：
   非负 KL 散度锚定项死死约束住模型与参考基座之间的距离，保证模型在探索出新推理逻辑的同时，不丢失基础语言理解能力。

---

## 步骤 2：承前启后的关键过渡

我们如何在不调用 PyTorch、HuggingFace 或任何第三方库的前提下，仅用标准 Python 实现多项式采样、确定性规则验证、群组优势标准化（$A_i = \frac{R_i - \mu}{\sigma}$）、重要性比率计算（$\rho_t = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{旧}}}$）、PPO 裁剪更新以及 Schulman 非负 KL 散度惩罚？

---

## 步骤 3：严谨数学公式与推导

### 1. 群组优势归一化
针对规模为 $G$ 的采样组，其得分向量为 $R_1, \dots, R_G$：

$$
\mu = \frac{1}{G} \sum_{i=1}^G R_i, \quad \sigma = \sqrt{\frac{1}{G} \sum_{i=1}^G (R_i - \mu)^2 + \epsilon}
$$
$$
A_i = \frac{R_i - \mu}{\sigma}
$$

### 2. 词元级比率与 PPO 裁剪
对第 $i$ 个采样的第 $t$ 个词元：

$$
\rho_{i,t} = \frac{\pi_{\boldsymbol{\theta}}(a_{i,t} \mid s_{i,t})}{\pi_{\boldsymbol{\theta}_{\text{旧}}}(a_{i,t} \mid s_{i,t})}
$$
$$
\mathcal{L}^{\text{CLIP}}_{i,t} = \min\left( \rho_{i,t} A_i, \; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right)
$$

### 3. Schulman 非负 KL 散度约束
为了约束策略 $\pi_{\boldsymbol{\theta}}$ 与参考模型 $\pi_{\text{ref}}$ 的偏移：

$$
u = \frac{\pi_{\text{ref}}(a_{i,t} \mid s_{i,t})}{\pi_{\boldsymbol{\theta}}(a_{i,t} \mid s_{i,t})}, \quad D_{\text{KL}} = u - \log(u) - 1 \ge 0
$$

最终策略梯度按 $\frac{A_i}{|o_i|} - \beta D_{\text{KL}}$ 比例精准推升参数。

---

## 步骤 4：历史渊源与技术演进

2024 年初，DeepSeek 团队在 **DeepSeek-Math** 中首次提出 GRPO，大幅降低了长推理链的训练门槛。2025 年初，**DeepSeek-R1** 证明：只要配合极简的格式规范与准确率规则奖励，GRPO 便能彻底激发基座模型的慢思考潜能，实现全自主自我纠错与长思维链跃迁。

---

## 步骤 5：完整纯 Python 代码实现

以下是完整可直接运行的 Python 源码（[`labs/07_grpo_engine.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/07_grpo_engine.py)），可直接使用 `python3 labs/07_grpo_engine.py` 进行验证：

```python
"""
动手实验 07：纯 Python 实现 GRPO 推理引擎
Python 大模型演化路线图第 7 阶段（强化学习与慢思考推理前沿）
"""

import math
import random

# 1. 词表与简易分词器
VOCAB = [
    "<pad>", "<eos>",
    "<think>", "</think>", "<answer>", "</answer>",
    "2", "3", "4", "5", "7", "8", "12", "14", "20",
    "+", "*", "=", "correct", "wrong"
]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: tok for idx, tok in enumerate(VOCAB)}
EOS_ID = TOKEN_TO_ID["<eos>"]
PAD_ID = TOKEN_TO_ID["<pad>"]

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def sample_token(probs):
    r = random.random()
    cum = 0.0
    for idx, p in enumerate(probs):
        cum += p
        if r <= cum:
            return idx
    return len(probs) - 1

# 2. 策略网络原型
class ToyPolicy:
    def __init__(self, vocab_size, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.weights = [[(random.random() - 0.5) * 0.1 for _ in range(vocab_size)] for _ in range(vocab_size)]

    def forward(self, prev_token_id):
        return self.weights[prev_token_id]

    def get_probs(self, prev_token_id):
        return softmax(self.forward(prev_token_id))

    def clone(self):
        new_model = ToyPolicy(self.vocab_size)
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                new_model.weights[i][j] = self.weights[i][j]
        return new_model

# 3. 确定性规则验证器（RLVR）
def verify_response(tokens, expected_answer="14"):
    text_tokens = [ID_TO_TOKEN[t] for t in tokens if t not in (PAD_ID, EOS_ID)]
    format_reward = 0.0
    has_ts = "<think>" in text_tokens
    has_te = "</think>" in text_tokens
    has_as = "<answer>" in text_tokens
    has_ae = "</answer>" in text_tokens

    if has_ts and has_te and has_as and has_ae:
        if text_tokens.index("<think>") < text_tokens.index("</think>") < text_tokens.index("<answer>") < text_tokens.index("</answer>"):
            format_reward = 0.5

    acc_reward = 0.0
    if expected_answer in text_tokens:
        if has_as and has_ae:
            ans_slice = text_tokens[text_tokens.index("<answer>"):text_tokens.index("</answer>")]
            if expected_answer in ans_slice:
                acc_reward = 1.0
        else:
            acc_reward = 0.5

    return format_reward + acc_reward, format_reward, acc_reward

# 4. 群组相对优势归一化
def compute_group_advantages(rewards):
    G = len(rewards)
    mean_r = sum(rewards) / G
    variance = sum((r - mean_r) ** 2 for r in rewards) / G
    std_r = math.sqrt(variance + 1e-4)
    advantages = [(r - mean_r) / std_r for r in rewards]
    return advantages, mean_r, std_r

# 5. GRPO 训练步（集成裁剪与 Schulman KL）
def train_grpo_step(actor, ref_model, prompt_token_id, group_size=6, lr=0.05, clip_eps=0.2, beta=0.04):
    old_actor = actor.clone()
    rollouts, rewards = [], []
    
    for _ in range(group_size):
        actions = []
        tokens = [prompt_token_id]
        for _ in range(8):
            probs = old_actor.get_probs(tokens[-1])
            action = sample_token(probs)
            tokens.append(action)
            actions.append(action)
            if action == EOS_ID:
                break
        tot_r, fmt_r, acc_r = verify_response(actions, expected_answer="14")
        rollouts.append(actions)
        rewards.append(tot_r)
        
    advantages, mean_r, std_r = compute_group_advantages(rewards)
    
    for i in range(group_size):
        actions = rollouts[i]
        adv = advantages[i]
        seq_len = max(len(actions), 1)
        prev = prompt_token_id
        for action in actions:
            p_curr = actor.get_probs(prev)[action]
            p_old = old_actor.get_probs(prev)[action]
            p_ref = ref_model.get_probs(prev)[action]
            
            ratio = p_curr / (p_old + 1e-12)
            clipped_ratio = max(min(ratio, 1.0 + clip_eps), 1.0 - clip_eps)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            
            u = p_ref / (p_curr + 1e-12)
            kl = u - math.log(u + 1e-12) - 1.0
            
            grad_scale = (adv / (p_old + 1e-12) if (surr1 <= surr2 or (adv > 0 and ratio < 1 + clip_eps)) else 0.0)
            grad_scale -= beta * kl
            
            step_factor = lr * (grad_scale / seq_len)
            probs_curr = actor.get_probs(prev)
            for j in range(VOCAB_SIZE):
                grad_logit = (1.0 if j == action else 0.0) - probs_curr[j]
                actor.weights[prev][j] += step_factor * grad_logit
                
            prev = action
            
    return mean_r, std_r
```

---

## 步骤 6：核心精髓总结

<fieldset markdown="1">
<legend><strong>核心精髓总结</strong></legend>

通过 200 行标准 Python 代码亲手实现 GRPO，你已经彻底揭开了 DeepSeek-R1 的算法底层：大模型通往通用慢思考推理的道路并不依赖数十亿参数的评论家网络，也不依赖无休止的人工打分，而是建立在纯粹的群组相对优势归一化、坚固的规则判别以及严格的策略裁剪边界之上。
</fieldset>
