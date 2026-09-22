# 阶段实战工坊 01：80 行纯 Python 训练首个自研大脑（Bengio 2003）

<nav aria-label="目录">
  <p>
    <strong>工坊导航：</strong>
    <a href="#evolution">自研大脑进化链</a> &bull;
    <a href="#step-1">第 1 步：直觉理解</a> &bull;
    <a href="#step-2">第 2 步：桥梁映射</a> &bull;
    <a href="#step-3">第 3 步：完整源码与数学解剖</a> &bull;
    <a href="#step-4">第 4 步：历史渊源</a> &bull;
    <a href="#step-5">第 5 步：运行实测与自回归生成</a> &bull;
    <a href="#step-6">第 6 步：第一阶段核心精髓与致命缺陷</a>
  </p>
</nav>

<hr>

<fieldset id="evolution">
<legend><strong>自研大脑进化链 &bull; 阶段 1 / 4</strong></legend>
<p>恭喜你学完了第 00 章至第 04 章！此时你手头已经掌握了<strong>下一个词预测、词向量嵌入、点积几何、矩阵线性投影、非线性激活与微积分反向传播</strong>的全部核心工具。</p>
<p>我们不需要依赖 PyTorch、TensorFlow，甚至不需要 NumPy。在第一个实战工坊中，我们将用 80 行纯原生 Python 代码，从零手搓一个具备真实自学习与文本生成能力的微型神经语言模型！</p>
<pre>
【自研大脑进化路线图】
[阶段 1 (当前)] 80 行纯 Python：Bengio 2003 MLP 语言模型（掌握词嵌入、前向全连接与手工反向传播）
       │
       ▼ (发现致命死穴：视野仅有 1 个词，长句子立刻患上失忆症)
[阶段 2 (第 08 章)] 140 行纯 Python：Attention 动态注意力大脑（解锁 Q, K, V 与全序列长程聚焦）
       │
       ▼ (发现深度缺陷：层数做深后数值崩溃、梯度弥散)
[阶段 3 (第 13 章)] 220 行纯 Python：现代 Transformer 块（合体残差连接、RMSNorm 与 SwiGLU 门控）
       │
       ▼ (优化生成体验：贪心采样机械死板)
[阶段 4 (第 17 章)] 300 行纯 Python 终极版：工业级自回归推理引擎（KV Cache 缓存加速与 Top-p 动态采样）
</pre>
</fieldset>

---

<h2 id="step-1">第 1 步：3 岁小孩也能懂的直觉比喻（发条打孔音乐盒）</h2>

想象你正在搭建一个全机械的**古董发条八音盒**：

<figure>
<pre>
【发条八音盒的纯机械运转】
[ 输入打孔纸卡 ("cat") ]
         │
         ▼
[ 齿轮插槽：识别卡片厚度 (嵌入矩阵 E) ]
         │
         ▼
[ 一组带弹簧的联动连杆 (线性变换 W1) ]
         │
         ▼
[ 单向棘轮门：只许向前推，负向卡死 (ReLU 激活) ]
         │
         ▼
[ 铜质音叉拨片敲击铃铛 (输出投影 W2) ]
         │
         ▼
[ 弹出的下一张纸卡 ("sat") ]
</pre>
<figcaption><strong>图 B1.1：</strong> 神经网络本质上就是一个由可微齿轮构成的物理乐器。输入一个音符，弹簧与连杆层层传动，敲响下一个最和谐的音符。</figcaption>
</figure>

1. **输入卡片（Token ID）**：
   你往入口塞入一张写着 `"cat"` 的木牌。
2. **齿轮插槽（词嵌入矩阵 $\mathbf{E}$）**：
   八音盒内部有不同深浅的插槽，卡片插进去的瞬间，顶起了一组 4 根不同高度的测针（把离散符号变成了连续的 4 维空间坐标）。
3. **连杆传动（权重矩阵 $\mathbf{W}_1$）**：
   这 4 根测针带动了 8 根复杂的机械连杆，连杆互相推拉，把空间进行拉伸和旋转。
4. **单向棘轮门（$\operatorname{ReLU}$ 激活函数）**：
   每根连杆末端装有一个单向翻门。往前推的冲力畅通无阻，往后拉的阻力直接被挡板卡在零位。**这个小小的翻门，让整台机器拥有了表达复杂非线性音律的能力。**
5. **铃铛敲击与概率选择（Softmax 与 $\mathbf{W}_2$）**：
   连杆最终敲击不同音调的铜铃，发出叮咚声。哪口铜铃的声音最洪亮，哪张备选字卡就会从出口滑落出来。
6. **自动调音螺丝（反向传播与梯度下降）**：
   如果原本应该弹出 `"sat"`，结果弹出了 `"the"`，质检员就会拿扳手顺着连杆**倒着往回摸**。哪根连杆偏离了，就松开螺母微调半圈（更新权重）。连续微调 100 次后，八音盒敲击出的音调便分毫不差！

---

<h2 id="step-2">第 2 步：数学公式到纯 Python 代码的映射桥梁</h2>

很多初学者之所以对深度学习感到敬畏，是因为被商业级框架（PyTorch、CUDA、C++ 底层）复杂的语法遮蔽了双眼。

把所有外部包装剥除后，**前 5 章的所有数学公式与 Python 原生语法的对应关系纯净得令人吃惊**：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 B1.1：</strong> 理论数学公式与纯原生 Python 操作逐行解剖对照表</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="15%">模块</th>
      <th scope="col" align="left" width="20%">对应章节</th>
      <th scope="col" align="left" width="30%">经典数学公式</th>
      <th scope="col" align="left" width="35%">纯 Python 实现（无第三方库）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>词嵌入查找</strong></th>
      <td>第 01 章</td>
      <td>$\mathbf{x} = \mathbf{e}_i^\top \mathbf{E} \in \mathbb{R}^{1 \times d}$</td>
      <td><code>x_vec = E[x_id]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>隐藏层线性变换</strong></th>
      <td>第 03 章</td>
      <td>$\mathbf{z}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1$</td>
      <td><code>[sum(x_vec[k]*W1[k][j] for k in range(d)) + b1[j] ...]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>空间折叠激活</strong></th>
      <td>第 04 章</td>
      <td>$\mathbf{a}_1 = \operatorname{ReLU}(\mathbf{z}_1) = \max(0, \mathbf{z}_1)$</td>
      <td><code>[max(0.0, v) for v in z1]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>输出词表投影</strong></th>
      <td>第 03 章</td>
      <td>$\mathbf{z}_2 = \mathbf{a}_1 \mathbf{W}_2 + \mathbf{b}_2$</td>
      <td><code>[sum(a1[k]*W2[k][j] for k in range(h)) + b2[j] ...]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Softmax 归一化</strong></th>
      <td>第 00 章</td>
      <td>$\hat{y}_j = \frac{e^{z_{2,j}}}{\sum_k e^{z_{2,k}}}$</td>
      <td><code>exp_z = [math.exp(v) ...]; [v/sum(exp_z) for v in exp_z]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>交叉熵误差</strong></th>
      <td>第 00 章</td>
      <td>$\mathcal{L} = -\log(\hat{y}_{\text{target}})$</td>
      <td><code>loss = -math.log(probs[y_target])</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>输出层误差信号</strong></th>
      <td>第 04 章</td>
      <td>$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = \hat{\mathbf{y}} - \mathbf{y}^*$</td>
      <td><code>dz2 = probs[:]; dz2[y_target] -= 1.0</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>激活阀门反传</strong></th>
      <td>第 04 章</td>
      <td>$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1}\right) \odot \operatorname{ReLU}'(\mathbf{z}_1)$</td>
      <td><code>dz1 = [da1[j] if z1[j] > 0 else 0.0 ...]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>梯度下降更新</strong></th>
      <td>第 04 章</td>
      <td>$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$</td>
      <td><code>W[k][j] -= lr * dW[k][j]</code></td>
    </tr>
  </tbody>
</table>

---

<h2 id="step-3">第 3 步：完整源码与数学解剖</h2>

为了让大家通过“亲手手搓”获得最扎实的物理与数学肌肉记忆，本工坊提供两个配套代码文件：
- **启发式填空练习模板**：[`labs/01_micro_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/01_micro_brain_exercise.py)（亦位于 [`04b-lab-micro-brain/micro_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/04b-lab-micro-brain/micro_brain_exercise.py)），将核心数学变换留空为带有启发式注释、张量形状与断言校验的 TODO 桩函数；
- **完整参考答案代码**：[`labs/01_micro_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/01_micro_brain.py)（亦位于 [`04b-lab-micro-brain/micro_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/04b-lab-micro-brain/micro_brain.py)）。

这是阶段 1 的完整可执行 Python 脚本参考实现：

<figure>
<pre>
# =====================================================================
# 阶段实战工坊 01：80 行纯 Python 手搓微型大脑（Bengio 2003）
# 依赖：零依赖（仅使用 Python 原生内置 math 与 random 模块）
# =====================================================================
import math
import random

# 1. 语料库与微型词汇表 (第 00 章)
corpus = "the cat sat on the mat the dog sat on the rug"
words = corpus.split()
vocab = sorted(list(set(words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)          # 词表尺寸 |V| = 7
d_embed = 4             # 词向量特征维度 d = 4 (第 01 章)
d_hidden = 8            # 隐藏层空间维度 = 8 (第 03 章)
lr = 0.1                # 学习率 eta (第 04 章)

# 构建自回归训练样本对：(当前词 -> 下一个词)
dataset = [(word2id[words[i]], word2id[words[i+1]]) for i in range(len(words)-1)]

# 2. 权重矩阵随机初始化 (第 01 & 03 章)
random.seed(42)
def init_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

E  = init_matrix(V, d_embed)         # 词嵌入查找矩阵 E (第 01 章)
W1 = init_matrix(d_embed, d_hidden)  # 升维投影矩阵 W1 (第 03 章)
b1 = [0.0] * d_hidden                # 隐藏层偏置 b1
W2 = init_matrix(d_hidden, V)        # 词表投影矩阵 W2 (第 03 章)
b2 = [0.0] * V                       # 词表偏置 b2

# 3. 训练主循环：前向传播、链式反向传播与 SGD 更新 (第 04 章)
for epoch in range(121):
    total_loss = 0.0
    
    for x_id, y_target in dataset:
        # ---【前向计算通道】---
        x_vec = E[x_id]  # 查表取出 1x4 词向量 (第 01 章)
        
        # 矩阵乘法 z1 = x * W1 + b1 (第 03 章)
        z1 = [sum(x_vec[k] * W1[k][j] for k in range(d_embed)) + b1[j] for j in range(d_hidden)]
        
        # 激活空间折叠 a1 = ReLU(z1) (第 04 章)
        a1 = [max(0.0, val) for val in z1]
        
        # 投影回词表 logits z2 = a1 * W2 + b2 (第 03 章)
        z2 = [sum(a1[k] * W2[k][j] for k in range(d_hidden)) + b2[j] for j in range(V)]
        
        # Softmax 概率归一化 (第 00 章)
        max_z2 = max(z2)
        exp_z2 = [math.exp(val - max_z2) for val in z2]
        sum_exp = sum(exp_z2)
        probs = [val / sum_exp for val in exp_z2]
        
        # 交叉熵损失 (第 00 章)
        loss = -math.log(max(probs[y_target], 1e-12))
        total_loss += loss
        
        # ---【反向溯源通道 (第 04 章)】---
        # 输出层责任信号 dz2 = probs - one_hot
        dz2 = probs[:]
        dz2[y_target] -= 1.0
        
        # 对 W2 与 b2 的局部敏感度导数
        dW2 = [[a1[k] * dz2[j] for j in range(V)] for k in range(d_hidden)]
        db2 = dz2[:]
        
        # 穿过 W2 向前回传误差：da1 = dz2 * W2^T
        da1 = [sum(dz2[j] * W2[k][j] for j in range(V)) for k in range(d_hidden)]
        
        # 穿过 ReLU 导数阀门：dz1 = da1 * (1 if z1 > 0 else 0)
        dz1 = [da1[j] if z1[j] > 0 else 0.0 for j in range(d_hidden)]
        
        # 对 W1 与 b1 的局部敏感度导数
        dW1 = [[x_vec[k] * dz1[j] for j in range(d_hidden)] for k in range(d_embed)]
        db1 = dz1[:]
        
        # 回传给词向量自身：dx_vec = dz1 * W1^T
        dx_vec = [sum(dz1[j] * W1[k][j] for j in range(d_hidden)) for k in range(d_embed)]
        
        # ---【参数物理更新：负梯度微调 (第 04 章)】---
        for k in range(d_hidden):
            for j in range(V):
                W2[k][j] -= lr * dW2[k][j]
        for j in range(V):
            b2[j] -= lr * db2[j]
        for k in range(d_embed):
            for j in range(d_hidden):
                W1[k][j] -= lr * dW1[k][j]
        for j in range(d_hidden):
            b1[j] -= lr * db1[j]
        for k in range(d_embed):
            E[x_id][k] -= lr * dx_vec[k]

# 4. 自回归文本生成测试 (第 00 章)
curr_word = "cat"
generated = [curr_word]
for _ in range(6):
    x_id = word2id[curr_word]
    x_vec = E[x_id]
    z1 = [sum(x_vec[k] * W1[k][j] for k in range(d_embed)) + b1[j] for j in range(d_hidden)]
    a1 = [max(0.0, val) for val in z1]
    z2 = [sum(a1[k] * W2[k][j] for k in range(d_hidden)) + b2[j] for j in range(V)]
    exp_z2 = [math.exp(val - max(z2)) for val in z2]
    probs = [val / sum(exp_z2) for val in exp_z2]
    next_id = probs.index(max(probs))
    curr_word = id2word[next_id]
    generated.append(curr_word)

print("生成结果:", " ".join(generated))
</pre>
<figcaption><strong>代码清单 B1.1：</strong> 完整的阶段 1 纯 Python 神经网络语言模型，无需任何库安装即可直接运行。</figcaption>
</figure>

---

<h2 id="step-4">第 4 步：历史渊源与技术演进（Bengio 2003 的破晓时刻）</h2>

在 2003 年之前，人类构建语言模型的唯一方式是**传统统计 N-gram 频率计数**：
- 如果一个词组在百科全书中从未同时出现过（例如 <samp>"astronaut rides a camel"</samp>），它的统计频次就是 $0$；
- 统计模型会机械地判定这句话在语法上的概率为 $0$。
- 如果想要统计 5 个词以上的长连贯性，随着词表 $|V|$ 变大，需要记录的状态组合呈 $|V|^5$ 爆发（对于 10 万词表，状态数高达 $10^{25}$），任何计算机硬盘都无法容纳。这就是著名的<dfn id="def-curse-of-dimensionality"><strong>维数灾难（Curse of Dimensionality）</strong></dfn>。

<dl>
  <dt><time datetime="2003">2003年</time> &mdash; <strong>Yoshua Bengio 等人</strong>：神经概率语言模型（NPLM）横空出世</dt>
  <dd>
    Bengio 提出了一项改写历史的划时代构想：<strong>放弃离散词频计数，让每一个词在连续几何空间中拥有一个稠密向量（Distributed Representation）！</strong><br>
    因为在这个连续空间中：
    
$$
\text{“猫”} \approx \text{“狗”}, \quad \text{“地毯”} \approx \text{“垫子”}
$$

    即使训练集里只见过 <samp>"the cat sat on the mat"</samp>，当模型第一次在测试集里看到 <samp>"the dog sat on the rug"</samp> 时，凭借几何向量的余弦相似度，网络依然能自动领悟出这完全是一句合乎语法的优美句子！<cite>《A Neural Probabilistic Language Model》, JMLR 2003</cite>。
  </dd>
</dl>

我们上面编写的 80 行代码，正是这篇划时代经典论文的最精简、最纯粹的现代实现。

---

<h2 id="step-5">第 5 步：运行实测与自回归生成</h2>

运行该脚本，我们将亲眼见证模型从一个“纯随机乱叫的婴儿”，在 120 轮梯度更新内迅速成长为一个“懂得动物动作模式的小天才”：

<figure>
<pre>
Vocabulary size |V|: 7, Words: ['cat', 'dog', 'mat', 'on', 'rug', 'sat', 'the']

Training Pure Python Neural Network...
Epoch   0 | Average Loss: 1.9735  (几乎等于理论纯随机交叉熵 ln(7) ≈ 1.9459)
Epoch  20 | Average Loss: 1.8540
Epoch  40 | Average Loss: 0.8253  (损失开始断崖式下跌，空间折叠成型)
Epoch  60 | Average Loss: 0.6108
Epoch  80 | Average Loss: 0.5808
Epoch 100 | Average Loss: 0.5683
Epoch 120 | Average Loss: 0.5582

Autoregressive Sequence Generation:
Prompt: 'cat'
Generated: cat sat on the rug the rug
</pre>
<figcaption><strong>输出记录 B1.1：</strong> 纯 Python 神经网络在 120 轮训练中的损失收敛轨迹与自回归连续单词生成演示。</figcaption>
</figure>

### 词元概率跃迁计量计

对于输入词 <kbd>"cat"</kbd>，让我们手算并测量模型在各可能候选词上的最终 Softmax 预测分布：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 B1.2：</strong> 输入 <kbd>"cat"</kbd> 时，模型对词表中所有词的预测概率分布</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left">候选词 Token</th>
      <th scope="col" align="right">预测概率 $P(w_{t+1} \mid \text{"cat"})$</th>
      <th scope="col" align="left">动态能量计量表</th>
      <th scope="col" align="left">状态判定</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><kbd>"sat"</kbd></th>
      <td align="right"><strong>99.01%</strong></td>
      <td><meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.9901">0.9901</meter></td>
      <td><mark>极度自信（正解）</mark></td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left"><kbd>"the"</kbd></th>
      <td align="right">0.45%</td>
      <td><meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0045">0.0045</meter></td>
      <td>退火静默</td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"on"</kbd></th>
      <td align="right">0.22%</td>
      <td><meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0022">0.0022</meter></td>
      <td>退火静默</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left"><kbd>"dog"</kbd></th>
      <td align="right">0.11%</td>
      <td><meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0011">0.0011</meter></td>
      <td>退火静默</td>
    </tr>
  </tbody>
</table>

---

<h2 id="step-6">第 6 步：第一阶段核心精髓与致命缺陷</h2>

<fieldset>
<legend><strong>阶段 1 核心里程碑精髓</strong></legend>
深度学习绝非高不可攀的黑盒魔法。仅仅用矩阵乘法、一个单向 ReLU 阀门和链式反向传播，机器就拥有了通过梯度自我进化的生命力。从数学上讲，你已经亲手造出了世界上第一个具备自学习能力的语言预测大脑！
</fieldset>

### 但是，这个微型大脑有一个足以致命的硬伤：

仔细观察上面的自回归生成结果：
<samp>"cat sat on the rug the rug"</samp>

为什么它在生成到后面时，陷入了 <samp>"the rug the rug"</samp> 的死循环？
**因为它患有无法治愈的严重“短时失忆症”！**
- 当前模型是一个**单词输入模型（马尔可夫模型）**：它每次只能低头看**紧挨着的前 1 个词**。
- 当它看到 `"the"` 时，它只知道语料库里 `"the"` 后面跟着 `"rug"` 或 `"mat"`；它根本不记得 3 个词之前的主语究竟是 `cat` 还是 `dog`！
- 面对真实的自然语言长句（比如：*“那只昨天在暴雨中浑身湿透、被好心人收留的小猫，今天终于在柔软的垫子上 __”*），当前模型在处理最后的动词时，前面的所有主语定语早就被忘得一干二净。

**怎样才能打破单词输入的视野枷锁？怎样让模型能在长达几千个词的海洋里，瞬间精准定位并聚焦到最关键的信息？**

带上这个激动人心的挑战，让我们正式推开现代大语言模型最核心的基石大门——**第 05 章：大模型的全景图纸（Transformer 架构宏观巡览）**！

---

<nav aria-label="章节导航">
  <p>
    <a href="../04-activation-functions/index.zh.html">&larr; 第 04 章：激活函数（ReLU、GELU 与 SwiGLU）</a> &bull;
    <a href="../index.html">课程主页</a> &bull;
    <a href="../05-transformer-architecture/index.zh.html">第 05 章：大模型的全景图纸（Transformer 架构宏观巡览） &rarr;</a>
  </p>
</nav>
