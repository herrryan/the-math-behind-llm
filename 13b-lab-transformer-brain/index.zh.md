# 动手实验 03：用 220 行纯 Python 实现现代 Transformer 大脑

<fieldset id="evolution">
<legend><strong>Python 大脑演化链 &bull; 第 3 阶段（共 4 阶段）</strong></legend>
<p>在实验 02 中，我们构建了注意力大脑，成功摆脱了马尔可夫短视健忘症。然而，当深度学习先驱们尝试通过简单串联多层纯注意力层来构建深层网络时，遭遇了一个致命的数学困境：<strong>深层不稳定性（Depth Instability）</strong>。反向传播梯度要么彻底归零，要么发生数值爆炸，且多次加权平均使得最初的输入词汇特征被冲刷殆尽。</p>
<p>在第三个动手实验中，我们将从第一性原理出发，用约 220 行纯标准库 Python 打造一个<strong>现代前沿 Transformer 块（Modern Transformer Block）</strong>（复刻 LLaMA 3、Gemma、Mistral 和 DeepSeek 的核心前向与解析反向传播架构）。全程零外部依赖：无需 PyTorch、无需 TensorFlow、无需 NumPy。</p>
<pre>
[Python 大脑演化路线图]
[第 1 阶段] 80 行纯 Python：Bengio 2003 MLP 语言模型（嵌入、全连接层、手工反向传播）
       │
       ▼ (致命瓶颈：1 词上下文视界；对更早的从句完全健忘)
[第 2 阶段] 140 行纯 Python：注意力大脑（解锁 Q、K、V 投影与因果注意力机制）
       │
       ▼ (致命瓶颈：直接堆叠深层注意力会引发梯度消失与数值爆炸)
[第 3 阶段 (当前)] 220 行纯 Python：现代 Transformer 块（Pre-RMSNorm、残差连接与 SwiGLU 门控）
       │
       ▼ (致命瓶颈：僵硬贪婪解码导致死板重复的机械文本死循环)
[第 4 阶段 (第 17 章)] 300 行纯 Python 终极版：工业级推理引擎（KV 缓存与 Top-p 采样）
</pre>
</fieldset>

---

## 步骤 1：3岁孩子也能懂的直觉（隔音音乐厅、高架旁路与双重保险库）

想象整个神经网络是一座忙碌的大型音乐学院，学生们沿着长长的走廊依次传递故事纸条：

<figure>
<pre>
[现代 Transformer 块数据流]

输入的原始词汇特征 ──X0──────────────────────────────────────────────┐ (高架直通高速公路)
                         │                                               │
                         ▼                                               │
               [ Pre-RMSNorm 音量限幅器 ]                                │
                         │                                               │
                         ▼                                               │
               [ 多头因果自注意力机制 ]                                  │
                         │                                               │
                         ▼                                               │
                         └───────────────────────────► ( + ) ◄───────────┘
                                                        │
中间特征表征 ────────────X1─────────────────────────────┴────────────────┐ (高架直通高速公路)
                                                        │                │
                                                        ▼                │
                                              [ Pre-RMSNorm 限幅器 ]     │
                                                        │                │
                                                        ▼                │
                                              [ SwiGLU 知识记忆库 ]      │
                                                        │                │
                                                        ▼                │
                                                        └────► ( + ) ◄───┘
                                                                │
                                              X2 ──► [ 终层 RMSNorm ] ──► [ LM Head ] ──► 下一个词
</pre>
<figcaption><strong>图 13b.1:</strong> 现代 Transformer 块通过两条直通旁路（残差连接）与三道音量控制器（Pre-RMSNorm），确保超深层网络稳定流通。</figcaption>
</figure>

1. **音量平衡控制器（Pre-RMSNorm）**：
   如果走廊里的一个学生大声咆哮，而另一个学生窃窃私语，音乐厅的音响系统要么被震耳欲聋的高音彻底击穿烧毁（数值爆炸），要么根本听不清低音（数值消失）。在踏入任何高难度的教室之前，每个信号都先经过一个音量旋钮：计算当前向量的平均能量大小，并将其等比例缩放到稳定舒适的听觉区间。

2. **高架直通高速公路（残差连接 Residual Highways）**：
   在过去没有走廊的旧教学楼里，所有学生必须排队穿过每一个房间。一旦某间教室的门被卡住，整个学校的交通就会彻底瘫痪。**残差连接**在每间教室旁边修建了一条永远开通的高速直通通道：
   $$
   \mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \text{教室加工}(\mathbf{x}_{\text{norm}})
   $$
   学生带着最原始的纸条直接从高架路走过。如果教室里的老师觉得没有什么需要修改的，输出为零即可，原始信息完好无损地直达终点！

3. **双重保险库门控网络（SwiGLU 前馈网络）**：
   注意力机制只负责决定“此时此刻谁应该看谁”，它本身并不储存海量的事实知识百科。**前馈网络（FFN）**才是学院的总资料档案馆。现代大模型采用 **SwiGLU** 架构：
   - **Up 钥匙（$\mathbf{W}_{\text{up}}$）**：从书架上取出备选的事实资料。
   - **Gate 钥匙（$\mathbf{W}_{\text{gate}}$）**：操纵一个平滑的物理阀门（$\text{SiLU}$）。它审视当前的问题，精确决定这个抽屉应该拉开多大的缝隙。
   - **Down 钥匙（$\mathbf{W}_{\text{down}}$）**：把挑选确认后的知识压缩打包，直接送回高架高速公路上。

---

## 步骤 2：连接理论与纯 Python 的数学桥梁

第 11、12、13 与 14 章中的所有核心公式，均可一对一无缝映射为纯 Python 列表推导式：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 13b.1:</strong> 现代 Transformer 核心数学公式与纯 Python 逐行对应</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="18%">架构模块</th>
      <th scope="col" align="left" width="14%">对应章节</th>
      <th scope="col" align="left" width="34%">数学公式</th>
      <th scope="col" align="left" width="34%">纯 Python 实现（零依赖）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>前置 RMSNorm</strong></th>
      <td>第 13 章</td>
      <td>$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum x_j^2 + \epsilon}} \odot \boldsymbol{\gamma}$</td>
      <td><code>rms = math.sqrt(sum(v**2 for v in x)/d + 1e-5); [v/rms*g for v, g in zip(x, gamma)]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>残差直通公路</strong></th>
      <td>第 12 章</td>
      <td>$\mathbf{X}_1 = \mathbf{X}_0 + \operatorname{Sublayer}(\mathbf{X}_{0, \text{norm}})$</td>
      <td><code>X1 = [[X0[i][j] + sub[i][j] for j in range(d)] for i in range(T)]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>SiLU 激活函数</strong></th>
      <td>第 04 章</td>
      <td>$\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$</td>
      <td><code>def silu(x): return x / (1.0 + math.exp(-x))</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>SwiGLU 门控前馈</strong></th>
      <td>第 14 章</td>
      <td>$\operatorname{SwiGLU}(\mathbf{x}) = \left(\operatorname{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}$</td>
      <td><code>H_swiglu = [[silu(Hg[i][j]) * Hu[i][j] for j in range(d_ffn)] for i in range(T)]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>终层映射输出</strong></th>
      <td>第 03 章</td>
      <td>$\mathbf{Z} = \operatorname{RMSNorm}(\mathbf{X}_2)\mathbf{W}_{\text{head}} \in \mathbb{R}^{T \times |V|}$</td>
      <td><code>Z = matmul(rmsnorm_forward(X2, gamma_f)[0], W_head)</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>残差高速梯度</strong></th>
      <td>第 12 章</td>
      <td>$\frac{\partial \mathcal{L}}{\partial \mathbf{X}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}_1} + \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{0, \text{norm}}}$</td>
      <td><code>dX0 = [[dX1[i][j] + dX0_norm_back[i][j] for j in range(d)] for i in range(T)]</code></td>
    </tr>
  </tbody>
</table>

---

## 步骤 3：完整代码实现与解剖

本实验同样提供两个完全对齐的脚本：
- **引导式练习模板**：[`labs/03_transformer_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/03_transformer_brain_exercise.py)（同步存放于 [`13b-lab-transformer-brain/transformer_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/13b-lab-transformer-brain/transformer_brain_exercise.py)），留出核心数学运算并通过内置单元测试逐一验证。
- **完整参考实现**：[`labs/03_transformer_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/03_transformer_brain.py)（同步存放于 [`13b-lab-transformer-brain/transformer_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/13b-lab-transformer-brain/transformer_brain.py)）。

完整可直接执行的代码如下：

<figure>
<pre>
# =====================================================================
# 第 3 阶段：现代 Transformer 大脑（220 行纯 Python 实现）
# 依赖项：零外部库（仅使用标准库 math 和 random）
# =====================================================================
import math
import random

sentences = [
    "the cat sat on the mat .",
    "the dog sat on the rug .",
    "the cat walked on the mat .",
    "the dog walked on the rug ."
]

all_words = (" ".join(sentences)).split()
vocab = sorted(list(set(all_words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}

V = len(vocab)          # |V| = 9
d_model = 8             # 残差流维度
d_ffn = 16              # SwiGLU 隐藏扩展维度
lr = 0.1                # 学习率
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in sentences:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

def init_matrix(rows, cols, scale_init=0.2):
    return [[random.gauss(0, scale_init) for _ in range(cols)] for _ in range(rows)]

def matmul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_row(row):
    max_val = max(row)
    exp_r = [math.exp(v - max_val) for v in row]
    sum_r = sum(exp_r)
    return [v / sum_r for v in exp_r]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

def silu(x):
    return x * sigmoid(x)

def silu_deriv(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))

def rmsnorm_forward(X, gamma, eps=1e-5):
    T, d = len(X), len(X[0])
    X_norm = []
    rms_list = []
    for i in range(T):
        ms = sum(v * v for v in X[i]) / d
        rms = math.sqrt(ms + eps)
        rms_list.append(rms)
        X_norm.append([X[i][j] / rms * gamma[j] for j in range(d)])
    return X_norm, rms_list

def rmsnorm_backward(dX_norm, X, gamma, rms_list):
    T, d = len(X), len(X[0])
    dX = []
    dgamma = [0.0] * d
    for i in range(T):
        rms = rms_list[i]
        sum_gamma_dxn_x = sum(gamma[j] * dX_norm[i][j] * X[i][j] for j in range(d))
        row_dx = []
        for j in range(d):
            dgamma[j] += dX_norm[i][j] * (X[i][j] / rms)
            term1 = gamma[j] * dX_norm[i][j]
            term2 = (X[i][j] / (d * rms * rms)) * sum_gamma_dxn_x
            row_dx.append((term1 - term2) / rms)
        dX.append(row_dx)
    return dX, dgamma

random.seed(42)
params = {
    "E":           init_matrix(V, d_model),
    "gamma1":      [1.0] * d_model,
    "W_q":         init_matrix(d_model, d_model),
    "W_k":         init_matrix(d_model, d_model),
    "W_v":         init_matrix(d_model, d_model),
    "W_o":         init_matrix(d_model, d_model),
    "gamma2":      [1.0] * d_model,
    "W_gate":      init_matrix(d_model, d_ffn),
    "W_up":        init_matrix(d_model, d_ffn),
    "W_down":      init_matrix(d_ffn, d_model),
    "gamma_final": [1.0] * d_model,
    "W_head":      init_matrix(d_model, V),
}

for epoch in range(201):
    total_loss = 0.0
    for inputs, targets in dataset:
        T = len(inputs)
        X0 = [params["E"][idx][:] for idx in inputs]

        # 1. 第一次 Pre-RMSNorm
        X0_norm, rms1 = rmsnorm_forward(X0, params["gamma1"])

        # 2. 因果自注意力计算
        Q = matmul(X0_norm, params["W_q"])
        K = matmul(X0_norm, params["W_k"])
        V_mat = matmul(X0_norm, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                if j &gt; i:
                    scores[i][j] = -1e9

        A = [softmax_row(scores[i]) for i in range(T)]
        O_raw = matmul(A, V_mat)
        Attn_out = matmul(O_raw, params["W_o"])

        # 3. 第一次残差直通公路
        X1 = [[X0[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

        # 4. 第二次 Pre-RMSNorm
        X1_norm, rms2 = rmsnorm_forward(X1, params["gamma2"])

        # 5. SwiGLU 门控前馈网络
        H_gate = matmul(X1_norm, params["W_gate"])
        H_up   = matmul(X1_norm, params["W_up"])
        H_silu = [[silu(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]
        H_swiglu = [[H_silu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        FFN_out = matmul(H_swiglu, params["W_down"])

        # 6. 第二次残差直通公路
        X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

        # 7. 终层 RMSNorm
        X2_norm, rms_f = rmsnorm_forward(X2, params["gamma_final"])

        # 8. 语言模型头与损失计算
        Z = matmul(X2_norm, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]
        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # 解析反向传播 (Exact Analytical Backprop)
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]
        dW_head = matmul(transpose(X2_norm), dZ)
        dX2_norm = matmul(dZ, transpose(params["W_head"]))

        dX2, dgamma_f = rmsnorm_backward(dX2_norm, X2, params["gamma_final"], rms_f)
        dX1_res2 = dX2[:]
        dFFN_out = dX2[:]

        dW_down = matmul(transpose(H_swiglu), dFFN_out)
        dH_swiglu = matmul(dFFN_out, transpose(params["W_down"]))
        dH_up = [[dH_swiglu[i][j] * H_silu[i][j] for j in range(d_ffn)] for i in range(T)]
        dH_silu = [[dH_swiglu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        dH_gate = [[dH_silu[i][j] * silu_deriv(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]

        dW_gate = matmul(transpose(X1_norm), dH_gate)
        dW_up   = matmul(transpose(X1_norm), dH_up)

        dX1_norm = [[sum(dH_gate[i][k] * params["W_gate"][j][k] + dH_up[i][k] * params["W_up"][j][k] for k in range(d_ffn))
                     for j in range(d_model)] for i in range(T)]
        dX1_from_norm, dgamma2 = rmsnorm_backward(dX1_norm, X1, params["gamma2"], rms2)
        dX1 = [[dX1_res2[i][j] + dX1_from_norm[i][j] for j in range(d_model)] for i in range(T)]

        dX0_res1 = dX1[:]
        dAttn_out = dX1[:]

        dW_o = matmul(transpose(O_raw), dAttn_out)
        dO_raw = matmul(dAttn_out, transpose(params["W_o"]))
        dV_mat = matmul(transpose(A), dO_raw)
        dA = matmul(dO_raw, transpose(V_mat))

        dScores = [[0.0] * T for _ in range(T)]
        for i in range(T):
            sum_dA_A = sum(dA[i][k] * A[i][k] for k in range(T))
            for j in range(T):
                if j &lt;= i:
                    dScores[i][j] = A[i][j] * (dA[i][j] - sum_dA_A) * scale

        dQ = matmul(dScores, K)
        dK = matmul(transpose(dScores), Q)
        dW_q = matmul(transpose(X0_norm), dQ)
        dW_k = matmul(transpose(X0_norm), dK)
        dW_v = matmul(transpose(X0_norm), dV_mat)

        dX0_norm = [[sum(dQ[i][k] * params["W_q"][j][k] + dK[i][k] * params["W_k"][j][k] + dV_mat[i][k] * params["W_v"][j][k] for k in range(d_model))
                     for j in range(d_model)] for i in range(T)]
        dX0_from_norm, dgamma1 = rmsnorm_backward(dX0_norm, X0, params["gamma1"], rms1)
        dX0 = [[dX0_res1[i][j] + dX0_from_norm[i][j] for j in range(d_model)] for i in range(T)]

        # SGD 梯度下降更新
        for i in range(d_model):
            for v in range(V):
                params["W_head"][i][v] -= lr * dW_head[i][v]
            params["gamma_final"][i] -= lr * dgamma_f[i]
            params["gamma2"][i]      -= lr * dgamma2[i]
            params["gamma1"][i]      -= lr * dgamma1[i]
            for k in range(d_model):
                params["W_o"][i][k] -= lr * dW_o[i][k]
                params["W_q"][i][k] -= lr * dW_q[i][k]
                params["W_k"][i][k] -= lr * dW_k[i][k]
                params["W_v"][i][k] -= lr * dW_v[i][k]
            for k in range(d_ffn):
                params["W_gate"][i][k] -= lr * dW_gate[i][k]
                params["W_up"][i][k]   -= lr * dW_up[i][k]
        for k in range(d_ffn):
            for j in range(d_model):
                params["W_down"][k][j] -= lr * dW_down[k][j]
        for i in range(T):
            idx = inputs[i]
            for j in range(d_model):
                params["E"][idx][j] -= lr * dX0[i][j]
</pre>
<figcaption><strong>图 13b.2:</strong> 220 行纯 Python 实现的完整现代 Pre-RMSNorm、SwiGLU 门控 Transformer 块及反向传播。</figcaption>
</figure>

---

## 步骤 4：源自何处？（现代大模型架构的演进史）

当代主流生产级开源大模型（LLaMA 3、Gemma、Mistral、Qwen）相比 2017 年最初的原始论文，在三大关键环节完成了深度进化：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 13b.2:</strong> 2017 原始 Transformer 与 2024+ 现代大模型架构对比</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="22%">核心设计</th>
      <th scope="col" align="left" width="38%">原始论文 (Vaswani 2017)</th>
      <th scope="col" align="left" width="40%">现代主流架构 (LLaMA / Gemma)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>归一化位置</strong></th>
      <td><strong>Post-LN</strong>：$\operatorname{LN}(\mathbf{x} + \operatorname{Sublayer}(\mathbf{x}))$。需要极其精密的预热学习率策略，极易早期梯度爆炸。</td>
      <td><strong>Pre-LN</strong>：$\mathbf{x} + \operatorname{Sublayer}(\operatorname{Norm}(\mathbf{x}))$。从第 1 层到第 128 层，主干残差公路完全畅通，训练极度平稳。</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>归一化算法</strong></th>
      <td><strong>标准 LayerNorm</strong>：计算均值 $\mu$ 与方差 $\sigma^2$。需要两次扫描内存。</td>
      <td><strong>RMSNorm (Zhang 2019)</strong>：舍弃均值漂移项，直接按均方根能量缩放。节约约 7% 的显存与计算时间，效果完全持平。</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>前馈激活函数</strong></th>
      <td><strong>ReLU / 标准 GELU</strong>：单分支固定激活：$\operatorname{ReLU}(\mathbf{x} \mathbf{W}_1)\mathbf{W}_2$。</td>
      <td><strong>SwiGLU 门控 (Shazeer 2020)</strong>：双线性门控机制，通过专属的 Gate 矩阵动态决定事实释放量。</td>
    </tr>
  </tbody>
</table>

---

## 步骤 5：具体数字演练与单步算术跟踪

让我们用一个简单的 2 维微型向量，亲手检验 Pre-RMSNorm 与残差公路的算术细节：

1. **RMSNorm 缩放算术**：
   假设主干特征向量 $\mathbf{x} = [3.0, 4.0]$，缩放因子 $\boldsymbol{\gamma} = [1.0, 1.0]$：
   $$
   \operatorname{MS}(\mathbf{x}) = \frac{3.0^2 + 4.0^2}{2} = \frac{9 + 16}{2} = 12.5
   $$
   $$
   \operatorname{RMS}(\mathbf{x}) = \sqrt{12.5} \approx 3.5355
   $$
   $$
   \mathbf{x}_{\text{norm}} = \left[\frac{3.0}{3.5355}, \frac{4.0}{3.5355}\right] = [0.8485, 1.1314]
   $$
   向量的能量被极其平滑地校准到了单位尺度，全程完全无需计算均值！

2. **残差高速公路保护机制**：
   假设自注意力层处理归一化后的 $\mathbf{x}_{\text{norm}}$ 并输出了微小调整量 $\Delta \mathbf{x} = [0.1, -0.2]$。
   残差主干上的更新结果为：
   $$
   \mathbf{x}_1 = \mathbf{x} + \Delta \mathbf{x} = [3.0 + 0.1, 4.0 - 0.2] = [3.1, 3.8]
   $$
   在反向传播时，当一个误差梯度信号 $d\mathbf{x}_1 = [1.0, 1.0]$ 到达时，它直接分流到两个分支：
   $$
   d\mathbf{x}_{\text{highway}} = [1.0, 1.0], \quad d\mathbf{x}_{\text{sublayer}} = [1.0, 1.0]
   $$
   直通高架路分支在向后传递时**不经过任何权重矩阵的连续乘法**，彻底从根源上消除了梯度消失！

---

## 步骤 6：核心收获与致命瓶颈

<fieldset>
<legend><strong>核心教学总结</strong></legend>
<p>现代 Transformer 大脑通过<strong>无阻碍的残差高架通道</strong>与 <strong>Pre-RMSNorm 动态音量阀门</strong>的精妙配合，实现了跨越上百层深度的数学稳定性。再加上 <strong>SwiGLU 门控前馈网络</strong>对事实知识的动态提取，奠定了当今世界所有顶级前沿大模型的核心基石。</p>
</fieldset>

### 致命瓶颈：机械生成的局限性（Mechanical Generation）

我们现在的第 3 阶段 Transformer 大脑已经能够平稳训练至任意深度，并在测试样本上达到极低的交叉熵损失。
然而，观察此时模型的生成方式：
$$
x_{t+1} = \arg\max_{v} P(v \mid x_{\le t})
$$
如果完全依赖**贪婪解码（Greedy Decoding）**（即每一步只选概率最高的单字），模型会暴露出严重的工业短板：
1. **死板循环与重复死锁**：一旦模型踏入某个循环句式，贪婪解码会陷入无尽的复读机死循环（<samp>"the cat sat on the mat on the mat on the mat..."</samp>）。
2. **毫无创造力与表达多样性**：模型无法调节语调、探索备选方案或进行多样化头脑风暴。
3. **推理性能瓶颈（$O(T^2)$ 延迟）**：在生成第 $t$ 个词时，重新计算此前所有历史词的注意力会浪费极其昂贵的计算资源。

为了将这个纯粹的神经网络大脑升级为一个真正流畅、灵活、极速的工业级对话引擎，我们需要：
- **温度采样（Temperature）、Top-$k$ 与 Top-$p$（核采样 Nucleus Sampling）**：第 18 章
- **KV 缓存加速（Key-Value Cache）**：第 17 章

这一切的终极成果，将在 **第 4 阶段：完整大语言模型（实验 04）** 中彻底绽放！
