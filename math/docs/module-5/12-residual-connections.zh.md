# 第 12 章：无阻的特快高架桥（残差连接 Residual Connections）


## 第 1 步：3 岁孩子也能懂的直觉（透明描摹纸与神奇的特快高架桥） {: #step-1 }

!!! note "3岁小孩的直觉: 透明描摹纸与神奇的特快高架桥"
    想象你在幼儿园的美术课上，用铅笔认认真真画了一只非常可爱的小狗轮廓。

    这时候，老师邀请了 5 位不同的小画家来帮你的画补充细节：
    - 画家 1 想把小狗的毛涂成漂亮的金色。
    - 画家 2 想给小狗画一个红色的项圈。
    - 画家 3 想在小狗脚下画绿油油的草地。
    - 画家 4 想在头顶画蓝天和白云。
    - 画家 5 想在周围画几颗闪闪发光的小星星。

    试想一下：如果每一位画家来到桌前，都拿起橡皮擦把你原本画的线稿擦个精光，然后试图凭记忆重新画一整幅画，会发生什么？

    等轮到第 5 位画家动笔的时候，最初那只可爱的小狗早就被擦得面目全非、模糊成一团黑炭了！没有人还能认出最初画的究竟是什么。

    为了解决这个问题，老师给每位画家发了一张**完全透明的描摹塑料薄片**：

    1. 你最初画的铅笔小狗安安稳稳地躺在桌面上，**谁都不准擦**。
    2. 每一位画家把自己的透明薄片盖在上面，仅仅画上**他们自己负责的一小点新细节**（只画红项圈，或者只画草地）。
    3. 画完后，把所有透明薄片叠在一起。

    奇迹出现了：你最初画的那只可爱小狗，透过所有透明薄片依然清清楚楚地闪耀着！如果某位画家不小心画歪了一笔，你只需要轻轻拿走那一层塑料片，根本不会伤到最初的小狗分毫。

    在大语言模型中，这道永远保护初始信息的机制就叫做**残差连接（Residual Connection，也叫跳跃连接 Skip Connection）**。模型没有逼迫每一层网络去从零重塑整个句子的全部特征，而是留下了一条原封不动的**特快直通桥**，只要求每一层在原有信息的基础上，加上一丁点微调和修饰！

<figure>
<pre>
没有残差连接的传统深层网络（信息逐层被破坏磨损）：

输入 ──► [第 1 层] ──► [第 2 层] ──► ... ──► [第 80 层] ──► 输出
（到了第 80 层，最初的语义特征早就被改写得面目全非！）

残差连接架构（永不磨损的直通高架桥）：

                   原汁原味的原样直通高架桥（恒等复制）
          ┌──────────────────────────────────────────────┐
          │                                              │
          │                                              ▼
输入 ─────┴──► [ 核心计算层：注意力或 FFN ] ──► ( F(x) ) ──► ( + ) ──► 输出
  x                                                             x + F(x)
</pre>
<figcaption><strong>图 12.1：</strong> 残差连接通过一条无阻力的恒等通路，让底层的原始词元信号与反向误差梯度能够无损穿越 80 多层深层网络。</figcaption>
</figure>

---

## 第 2 步：承前启后的关键过渡 {: #step-2 }

!!! question "计算连接问题: 深度神经网络中的“梯度消失大灾难”"
    在现代大语言模型中，Transformer 层数通常多达数十甚至上百层（例如 LLaMA-7B 拥有 32 层，LLaMA-70B 拥有 80 层，DeepSeek-V3 拥有 61 层）。

    在没有快捷通道的传统深层前馈神经网络中，第 $l$ 层的输出直接是上一层输出的复合函数：

    $$
    \mathbf{x}_{l} = \mathcal{F}_l(\mathbf{x}_{l-1})
    $$

    当使用反向传播算法训练模型时，最终损失函数 $\mathcal{L}$ 对最底层输入 $\mathbf{x}_0$ 的梯度，必须根据多元微积分链式法则从第 $L$ 层逐层向前倒推：

    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \prod_{l=1}^L \mathbf{J}_l
    $$

    其中 $\mathbf{J}_l = \frac{\partial \mathcal{F}_l(\mathbf{x}_{l-1})}{\partial \mathbf{x}_{l-1}}$ 是第 $l$ 层变换的雅可比矩阵（Jacobian Matrix）。

    这导致了数学上的毁灭性灾难：
    - 如果各层雅可比矩阵的特征值略小于 $1$（例如 $0.9$），在 $80$ 层的连续连乘下：$0.9^{80} \approx 0.0002$。**梯度瞬间衰减归零（梯度消失）！** 靠近输入的底层网络根本接收不到任何更新反馈，参数彻底冻结。
    - 如果特征值略大于 $1$（例如 $1.1$），在 $80$ 层连乘下：$1.1^{80} \approx 2048$。**梯度爆炸至无穷大**，造成浮点溢出（`NaN`），训练瞬间崩溃。

    “我们究竟如何在代数层面上重塑网络层的结构，让梯度的链式法则中自带一个恒不为零的‘加法直通项 $\mathbf{I}$’，彻底打破矩阵连乘的乘法诅咒，保证误差信号能够无阻穿透 100 多层网络？”

---

## 第 3 步：严谨数学推导与公式 {: #step-3 }

### 1. 残差结构的核心公式

何恺明等人在 2015 年提出的深度残差学习（\lt dfn id="def-resnet-zh">ResNet</dfn>）中，打破了“让网络直接拟合目标映射 $\mathcal{H}(\mathbf{x})$”的传统观念，转而让网络层去学习差量映射：$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$。

其前向传播计算公式为：

$$
\mathbf{x}_{l} = \mathbf{x}_{l-1} + \mathcal{F}_l(\mathbf{x}_{l-1}, \mathbf{W}_l)
$$

其中：
- $\mathbf{x}_{l-1} \in \mathbb{R}^{T \times d_{\text{model}}}$ 是进入该子层的输入张量。
- $\mathcal{F}_l(\cdot)$ 代表该子层所包含的非线性复杂运算（在 Transformer 中即为多头自注意力层或前馈神经网络 FFN）。
- $\mathbf{x}_{l} \in \mathbb{R}^{T \times d_{\text{model}}}$ 是该子层计算完成后的输出张量。

---

### 2. 核心数学证明：为什么梯度永远不会消失？

为了在代数层面洞察这一加法结构带来的深远变革，我们考察任意一个深层 $L$ 与前面的任意浅层 $l$（满足 $l \lt L$）。

通过将前向公式递推展开：

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}_i(\mathbf{x}_i)
$$

现在利用链式法则，求标量损失函数 $\mathcal{L}$ 对浅层特征 $\mathbf{x}_l$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}_i(\mathbf{x}_i) \right)
$$

展开括号后，我们得到了**残差反向传播基本方程**：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}}_{\text{恒等直通高速通道}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left( \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}_i(\mathbf{x}_i)}{\partial \mathbf{x}_l} \right)}_{\text{受残差权重调制的反馈}}
$$

\lt details>
\lt summary>\lt strong>单位矩阵 $\mathbf{I}$ 如何彻底终结梯度消失？</strong></summary>
仔细观察这一代数形式所蕴含的优雅美感：
1. 第一项 $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}$ **与中间任何层的权重完全无关**！它从模型最顶层的损失函数出发，沿着恒等映射直接流回浅层 $l$，中途没有任何阻碍和缩减。
2. 即使中间某些层的权重尚未学好，或者其局部梯度很小导致求和项 $\sum \frac{\partial \mathcal{F}_i}{\partial \mathbf{x}_l} \approx \mathbf{0}$，总梯度的下限依然能够得到保障：
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \approx \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \mathbf{I} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \neq \mathbf{0}
   $$
3. 在乘法链条中，只要有一个因数接近 $0$，整个乘积立刻变为 $0$；而在加法通道中，只要存在一个非零常数项 $\mathbf{I}$，总和就绝不会归零！
</details>

---

### 3. Pre-LN 与 Post-LN：保持高架桥纯净的架构之争

虽然原始 Transformer 论文（Vaswani 等人，2017）引入了残差连接，但归一化层（LayerNorm）与残差加法的位置先后，对梯度高速公路的畅通程度产生了决定性的影响。

\lt figure>
\lt pre>
Post-LN 架构（原始论文 Vaswani 2017 方案）：
  x_{l} ──┬──► [ 子层变换 F ] ──► ( + ) ──► [ LayerNorm ] ──► x_{l+1}
          │                        ▲
          └────────────────────────┘
  致命缺陷：LayerNorm 恰好骑在残差主干道上！
  归一化分母的尺度缩放会严重干扰反向梯度的无阻流动，导致训练极易不稳定，必须依赖脆弱的学习率预热 (Warmup)。

Pre-LN 架构（现代工业标配：LLaMA, Mistral, DeepSeek）：
  x_{l} ──┬──► [ LayerNorm ] ──► [ 子层变换 F ] ──► ( + ) ──► x_{l+1}
          │                                          ▲
          └──────────────────────────────────────────┘
  巨大优势：主干高架桥 100% 畅通无阻！
  x_{l+1} = x_l + F(LayerNorm(x_l))。
  反向梯度可以沿着残差连接从最后一层无损倒灌回第一层，无需小心翼翼的 Warmup 也能直接稳定收敛！
</pre>
\lt figcaption>\lt strong>图 12.2：</strong> Post-LN 将归一化放在主干上阻塞了梯度通道；现代 Pre-LN 架构则将归一化移入支路，保证主干恒等通路的绝对纯净。</figcaption>
</figure>

在现代 **Pre-LN** Transformer 代码实现中：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}\left(\operatorname{LayerNorm}(\mathbf{x}_l)\right)
$$

残差跳跃路径 $\mathbf{x}_{l+1} = \mathbf{x}_l + \dots$ 没有任何多余的乘除法运算，正是这个微小的拓扑调优，使得数百层的大模型得以在第一天就顺利开跑。

---

## 第 4 步：历史源流与思考演进 {: #step-4 }

\lt figure>
\lt pre>
深度网络梯度通路的技术演进史：

1997: Hochreiter & Schmidhuber (LSTM) ──► 恒定误差轮盘 (CEC)
                                           在时间维度引入加法记忆单元：c_t = f_t * c_{t-1} + i_t * g_t。
      │
      ▼
2015 年 5 月: Srivastava 等人 (Highway Networks) ──► 门控捷径：y = H(x)*T(x) + x*(1-T(x))。
                                                      需要额外引入参数学习门控权重，计算代价较高。
      │
      ▼
2015 年 12 月: 何恺明等人 (ResNet) ──────► 纯粹恒等映射：y = x + F(x)。
                                           额外参数量完全为 0！一举训练出 152 层网络夺得 ImageNet 冠军。
      │
      ▼
2017: Vaswani 等人 (Transformer) ────────► 将每个自注意力与 FFN 子层全量套入残差：
                                           x + SubLayer(x)。
      │
      ▼
2020: 熊伟等人 / 王磊等人 (Pre-LN 变革) ─► 将 LayerNorm 移出残差主干，
                                           奠定了 GPT-3、LLaMA、DeepSeek 等现代大模型无坚不摧的基石。
</pre>
\lt figcaption>\lt strong>图 12.3：</strong> 从循环神经网络加法记忆到现代大模型 Pre-LN 残差通道的发展历程。</figcaption>
</figure>

### 1. 历史触发点：深层网络的“退化问题”（The Degradation Problem）

在 2015 年之前，学术界直觉地认为：如果一个 20 层的网络效果不错，那么一个 56 层的网络理应表现得更好或至少一样好。因为多出来的 36 层只要什么都不干，单纯学习成为**恒等映射**（$\mathcal{F}(\mathbf{x}) = \mathbf{x}$）即可。

然而实验结果令人大跌眼镜：56 层的深层网络在训练集上的误差，竟然显著高于 20 层的浅层网络！
原因在于：通过带非线性激活函数的神经层（$\sigma(\mathbf{x}\mathbf{W} + \mathbf{b}) = \mathbf{x}$）去硬生生逼近一个恒等变换极其艰难。优化器很容易在非凸鞍点中迷失。

通过直接将结构定义为 $\mathbf{x} + \mathcal{F}(\mathbf{x})$，拟合恒等变换变得不费吹灰之力——网络只需要把 $\mathcal{F}(\mathbf{x})$ 的所有参数权重置为零即可！

---

### 2. 现代拓扑方案横向评测

\lt fieldset>
\lt legend>\lt strong>网络层连接范式对比</strong></legend>

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>表 12.1：</strong> 各种短路连接方案的核心特性对比。</caption>
  \lt thead>
    \lt tr bgcolor="#f0eee6">
      \lt th align="left">架构设计</th>
      \lt th align="center">前向传播数学形式</th>
      \lt th align="center">新增参数</th>
      \lt th align="left">反向梯度传播表现</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt td>\lt strong>传统无短路网络</strong></td>
      \lt td align="center">$\mathbf{x}_{l} = \mathcal{F}(\mathbf{x}_{l-1})$</td>
      \lt td align="center">$0$</td>
      \lt td>
        \lt del>面临梯度消失与爆炸崩溃。</del> 梯度随雅可比矩阵连乘 $\prod \mathbf{J}_l$ 指数级衰减，网络通常无法突破 20 层。
      </td>
    </tr>
    \lt tr>
      \lt td>\lt strong>高速公路网络 (Highway Net)</strong></td>
      \lt td align="center">$\mathbf{x}_l = \mathcal{F}(\mathbf{x}) \odot \mathbf{T} + \mathbf{x} \odot (1 - \mathbf{T})$</td>
      \lt td align="center">需引入额外的门控矩阵 $\mathbf{W}_T$</td>
      \lt td>
        \lt del>存在门控阻碍。</del> 梯度必须通过门控系数 $(1 - \mathbf{T})$，并非百分之百自由通透。
      </td>
    </tr>
    \lt tr bgcolor="#fdfdf0">
      \lt td>\lt strong>Pre-LN 残差连接 (现代 LLM)</strong></td>
      \lt td align="center">$\mathbf{x}_l = \mathbf{x}_{l-1} + \mathcal{F}(\operatorname{LN}(\mathbf{x}_{l-1}))$</td>
      \lt td align="center">\lt ins>\lt strong>0 额外参数！</strong></ins></td>
      \lt td>
        \lt ins>\lt strong>畅通无阻的恒等高速路。</strong></ins> 公式自带 $+\mathbf{I}$ 梯度下限，使 80 层以上的大规模预训练如履平地。
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

## 第 5 步：手把手超简单数字积木（2 维向量的前向与反向纯手算） {: #step-5 }

为了让你百分之百理解残差加法与梯度的流向，我们用一组简单的二维向量，手动算一遍前向与反向过程。

### 1. 前向传播计算

设输入残差块的二维向量为：

$$
\mathbf{x}_{\text{in}} = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}
$$

假设该子层的非线性变换 $\mathcal{F}(\cdot)$ 是一个简单的全连接层，其权重矩阵为 $\mathbf{W}$（为推导清晰忽略偏置项）：

$$
\mathbf{W} = \begin{bmatrix} 0.1 & 0.2 \\ -0.1 & 0.3 \end{bmatrix}
$$

计算该子层的输出变化量 $\mathcal{F}(\mathbf{x}_{\text{in}}) = \mathbf{W}\mathbf{x}_{\text{in}}$：

$$
\mathcal{F}(\mathbf{x}_{\text{in}}) = \begin{bmatrix} 0.1(2.0) + 0.2(-1.0) \\ -0.1(2.0) + 0.3(-1.0) \end{bmatrix} = \begin{bmatrix} 0.2 - 0.2 \\ -0.2 - 0.3 \end{bmatrix} = \begin{bmatrix} 0.0 \\ -0.5 \end{bmatrix}
$$

现在执行残差加法，求出残差块的最终输出 $\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \mathcal{F}(\mathbf{x}_{\text{in}})$：

$$
\mathbf{x}_{\text{out}} = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 2.0 \\ -1.5 \end{bmatrix}
$$

\lt mark>看：原始输入向量 $[2.0, -1.0]^\top$ 构成了信号的骨架，在此基础上仅仅吸收了来自计算层的微小调整 $[-0.0, -0.5]^\top$！</mark>

---

### 2. 反向传播梯度计算

现在，假设上层网络传回的损失梯度到达本层输出 $\mathbf{x}_{\text{out}}$ 为：

$$
\mathbf{g}_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}
$$

我们来计算穿过残差块传回到输入 $\mathbf{x}_{\text{in}}$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} \left( \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_{\text{in}}} \right) = \mathbf{g}_{\text{out}}^\top (\mathbf{I} + \mathbf{W})
$$

将对应矩阵代入：

$$
\mathbf{I} + \mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.2 \\ -0.1 & 0.3 \end{bmatrix} = \begin{bmatrix} 1.1 & 0.2 \\ -0.1 & 1.3 \end{bmatrix}
$$

计算传回的梯度列向量 $\mathbf{g}_{\text{in}}$：

$$
\mathbf{g}_{\text{in}} = (\mathbf{I} + \mathbf{W})^\top \mathbf{g}_{\text{out}} = \begin{bmatrix} 1.1 & -0.1 \\ 0.2 & 1.3 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}
$$

执行简单的行列相乘：
- 第 1 个分量：$1.1(1.0) + (-0.1)(2.0) = 1.1 - 0.2 = 0.9$
- 第 2 个分量：$0.2(1.0) + 1.3(2.0) = 0.2 + 2.6 = 2.8$

$$
\mathbf{g}_{\text{in}} = \begin{bmatrix} 0.9 \\ 2.8 \end{bmatrix}
$$

\lt mark>将这个结果进行拆解分析：</mark>

$$
\mathbf{g}_{\text{in}} = \underbrace{\begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}}_{\text{来自恒等高架桥的无损梯度}} + \underbrace{\begin{bmatrix} -0.1 \\ 0.8 \end{bmatrix}}_{\text{来自变换支路的微调反馈}} = \begin{bmatrix} 0.9 \\ 2.8 \end{bmatrix}
$$

哪怕当前神经层的权重 $\mathbf{W}$ 全都是零，输入端依然能够 $100\%$ 完整地接收到来自上层的原始梯度 $\begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$！这就是残差连接创造的数学奇迹。

---

## 第 6 步：核心精要（一句话记住核心奥秘） {: #step-6 }

!!! tip "核心要点: 残差连接的终极心法"
    **残差连接在每个 Transformer 层旁搭建了一条畅通无阻的恒等高架桥，将深度网络的学习模式从“脆弱的乘法连乘”转变为“安全的加法累积”。**

    通过在反向传播导数中直接嵌入一个单位矩阵 $\mathbf{I}$，残差连接彻底根除了梯度消失的顽疾，让拥有上百层深度的大语言模型得以稳固训练。
