# 第 17 章：聪明的下山者（动量机制与 AdamW 优化器）

---

## 步骤 1：3 岁孩子也能懂的直觉（乒乓球与重型保龄球）

> [!INTUITION] 峡谷里乱撞的弹珠与平稳滚动的保龄球
> 想象在一条崎岖陡峭、两边全是碎石山壁的深山峡谷里，有两颗不同的小球准备向下滚向谷底：
>
> 1. **轻飘飘的乒乓球（朴素 SGD）**：
>    - 乒乓球太轻了，自身几乎没有任何重量。
>    - 只要遇到山壁上一颗微小的石头或者小坑洼，它就会被剧烈地撞飞，在左侧山壁和右侧山壁之间疯狂横跳！
>    - 它把 99% 的精力都白白浪费在左右两侧峭壁的无效弹跳上，向真正谷底的前进速度慢得像蜗牛。
>    - 如果你心急狠狠推它一把（调高学习率），它会瞬间被弹飞出山谷，彻底粉身碎骨！
>
> 2. **沉甸甸的实心保龄球（动量机制）**：
>    - 现在，我们向这条峡谷扔下一颗重达 15 磅的实心大保龄球。
>    - 虽然两边的山石依然在对它产生侧向推力，但由于保龄球已经积累了强大的**向前的运动惯性（动量）**，那些左右横向的细碎推力在滚动中被自然抵消了！
>    - 保龄球破浪前行，稳健笃定地沿着山谷谷底笔直向下冲刺，平顺地碾过细碎的石子。
>
> 3. **装有定制刹车的智能轮滑鞋（自适应学习率）**：
>    - 现在想象山谷里有整整 700 亿个登山队员同时下山。
>    - 有的队员脚下是垂直湿滑的冰川悬崖 &mdash; 他们必须踩下**重型防抱死刹车**，绝不能失控摔死！
>    - 有的队员深陷在平缓黏稠的泥潭沼泽里 &mdash; 他们需要穿上**火箭喷气动力轮滑鞋**，否则永远无法挪步！
>    - 聪明的优化器给每个队员安装了个性化雷达：谁遇到狂暴震荡的大斜坡，就自动缩小它的步长；谁处于平缓微弱的小斜坡，就自动放大它的步长。
>
> 在大语言模型中，这位全能的下山导航大师就是 **AdamW 优化器（<dfn id="def-adamw-zh">解耦权重衰减的自适应矩估计</dfn>）**。
>
> 它将保龄球的强劲动量与针对 700 亿个参数定制的自适应刹车系统融为一体，是大模型稳定训练的绝对基石！

<figure>
<pre>
朴素 SGD 与 AdamW 在病态峡谷地形中的下山轨迹对比：

朴素 SGD（在陡峭峡谷峭壁间剧烈横跳，消耗数万步原地打转）：
  \       /\       /\       /
   \     /  \     /  \     /
    \   /    \   /    \   /
     ▼ /      ▼ /      ▼ /

AdamW（动量抵消横向震荡 + 自适应缩放抚平悬崖，直扑谷底）：
  ═════════════════════════════► （平滑、直接、迅猛地沿着谷底前行！）
</pre>
<figcaption><strong>图 17.1：</strong> 朴素梯度下降在狭窄深谷两壁来回震荡；AdamW 借助动量消除杂音，并自动均衡各维度的步长。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

> [!BRIDGING] 为什么单纯的梯度下降在深层网络中会彻底瘫痪？
> 在第 16 章中，我们推导出了经典的梯度下降参数更新公式：
>
> $$
> \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t
> $$
>
> 这个公式在数学上极其优美，但在直接拿去训练 700 亿参数的 Transformer 时，会引发严重的灾难：
>
> 1. **病态峡谷曲率问题（Ill-Conditioned Ravines）**：大模型的损失曲面如同深不见底的“一线天”峡谷。横跨峡谷两壁的坡度（二阶海森矩阵的最大特征值 $\lambda_{\max}$）比顺着峡谷谷底的坡度（最小特征值 $\lambda_{\min}$）陡峭数万倍！如果全局使用统一的标量学习率 $\eta$，一旦步子稍大，就会在两壁震荡发散；一旦步子变小，顺着谷底前行的速度就会慢到令人绝望。
> 2. **罕见词元与高频词元的参数不平衡**：专业医学词汇或罕见人名可能几十万步才被反向传播光顾一次，而逗号或虚词几乎在每个批次都产生剧烈梯度。固定步长会导致罕见参数长期处于“饥饿缺乏更新”状态，而高频参数频繁被狂暴梯度踢飞。
>
> “我们如何在数学上精准追踪 700 亿个参数各自的滚动速度（一阶矩）与震荡能量（二阶矩），在启动阶段消除零初始化的拖拽偏差，并安全防止参数无节制膨胀？”

---

## 步骤 3：严谨数学推导与公式

### 1. 现代标配：AdamW 算法核心公式（Kingma & Ba, 2014; Loshchilov & Hutter, 2017）

在每个训练迭代步 $t$，给定当前模型参数向量 $\boldsymbol{\theta}_{t-1}$ 以及当前小批次计算出的梯度 $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{t-1})$：

#### 步骤 A：一阶动量估计（梯度的指数移动平均 &mdash; 模拟速度惯性）
$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t
$$

- $\mathbf{m}_t \in \mathbb{R}^P$ 追踪历史梯度的方向与动量。
- $\beta_1 \in [0, 1)$ 为一阶衰减系数（行业通用默认值：$\beta_1 = 0.9$）。

#### 步骤 B：二阶能量估计（梯度平方的指数移动平均 &mdash; 模拟震荡方差）
$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
$$

- $\mathbf{v}_t \in \mathbb{R}^P$ 追踪每个参数梯度的未中心化方差（能量大小）。$\mathbf{g}_t^2 = \mathbf{g}_t \odot \mathbf{g}_t$ 表示逐元素平方。
- $\beta_2 \in [0, 1)$ 为二阶衰减系数（大语言模型通常设为 $\beta_2 = 0.95$，经典视觉通常设为 $0.999$）。

#### 步骤 C：冷启动偏差修正（消除零初始化阻力）
因为在训练第 0 步时，算法初始化 $\mathbf{m}_0 = \mathbf{0}, \mathbf{v}_0 = \mathbf{0}$，在最初的几十步中，移动平均值会被严重拉向零。我们通过除以 $(1 - \beta^t)$ 进行动态补偿修正：

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

<details>
<summary><strong>数学推导证明：为什么除以 $1 - \beta^t$ 能够完美消除冷启动偏差？</strong></summary>

将递归式 $\mathbf{m}_t = (1 - \beta_1)\mathbf{g}_t + \beta_1 \mathbf{m}_{t-1}$ 彻底展开（设初始 $\mathbf{m}_0 = \mathbf{0}$）：

$$
\mathbf{m}_t = (1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i
$$

两边取数学期望 $\mathbb{E}[\cdot]$。假设近期的真实梯度期望均值为 $\mathbb{E}[\mathbf{g}]$：

$$
\mathbb{E}[\mathbf{m}_t] = \mathbb{E}\left[(1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i\right] = \mathbb{E}[\mathbf{g}] \cdot (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}
$$

利用有限等比数列求和公式：

$$
\sum_{i=1}^t \beta_1^{t-i} = \frac{1 - \beta_1^t}{1 - \beta_1}
$$

代入回期望式：

$$
\mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf{g}] \cdot (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[\mathbf{g}] \cdot (1 - \beta_1^t)
$$

在第一步 $t = 1$ 时（若 $\beta_1 = 0.9$），未修正的 $\mathbf{m}_1$ 实际上只有真实梯度的 $10\%$！
通过除以 $(1 - \beta_1^1) = 0.10$，刚好将其无偏放大 10 倍还原到 $100\%$！随着训练步数 $t$ 增大，$\beta^t \to 0$，修正项自然平滑淡出。
</details>

#### 步骤 D：解耦权重衰减更新步（AdamW）
$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \underbrace{\eta \lambda \boldsymbol{\theta}_{t-1}}_{\text{解耦权重衰减}} - \underbrace{\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t}_{\text{自适应动量梯度步}}
$$

其中：
- $\eta > 0$ 为学习率计划值。
- $\lambda \ge 0$ 为权重衰减超参数（LLM 预训练通常取 $0.1$）。
- $\epsilon > 0$ 是防止除以零的微小数值保护常数（通常取 $10^{-8}$ 或 $10^{-6}$）。

---

### 2. 为什么现代大模型全面采用 AdamW，彻底抛弃原生 Adam？

在 2014 年最初的 Adam 论文中，权重衰减是按照经典的 $L_2$ 正则化方式加在梯度上的：$\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda \boldsymbol{\theta}_{t-1}$。

2017 年，Loshchilov 和 Hutter 发现了这一做法的致命数学缺陷：
- 当 $\lambda \boldsymbol{\theta}$ 被直接混入梯度 $\mathbf{g}_t$ 时，它也无可避免地进入了分母的二阶矩 $\mathbf{v}_t$ 中！
- 对于频繁更新、梯度极大的核心权重，分母的 $\sqrt{\mathbf{v}_t}$ 非常大，**导致它们的权重衰减惩罚被大幅削弱**；
- 对于很少更新、梯度极其微小的参数，分母的 $\sqrt{\mathbf{v}_t}$ 很小，**导致它们的权重衰减惩罚被过度放大**！

**AdamW 的破局之道**：将权重衰减从自适应梯度除法中彻底解耦出来，在执行动量更新之前，让所有参数纯净地统一按比例收缩 $(1 - \eta \lambda)$。

---

## 步骤 4：历史源流与思考演进（从 AdaGrad 到 AdamW 的进化史）

<dl>
  <dt><time datetime="2011">2011</time> &mdash; <strong>杜奇、哈赞 与 辛格</strong>（<abbr title="Adaptive Gradient Algorithm">AdaGrad</abbr>）</dt>
  <dd>首次提出针对每个参数定制自适应学习率，分母除以历史所有梯度的平方累加和 $\sqrt{\sum g_\tau^2}$。但由于累加和单调递增，分母越来越大，学习率过早枯竭至 0，导致训练提前停滞。</dd>

  <dt><time datetime="2012">2012</time> &mdash; <strong>杰弗里·辛顿 团队</strong>（<abbr title="Root Mean Square Propagation">RMSProp</abbr>）</dt>
  <dd>用指数移动平均（$1 - \beta_2$）取代了死板的历史总和累加，赋予了优化器动态遗忘过往地形的能力，完美解决了 AdaGrad 步长过早冻结的绝症。</dd>

  <dt><time datetime="2014">2014</time> &mdash; <strong>迪德里克·金玛 与 吉米·巴</strong>（<abbr title="Adaptive Moment Estimation">Adam</abbr>）</dt>
  <dd>将 RMSProp 的自适应分母与经典动量机制（Polyak Momentum）双剑合璧，并提出了精妙的偏差修正公式，一经发布便成为深度学习最受欢迎的优化器。</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>伊利亚·洛希奇洛夫 与 弗兰克·哈特</strong>（<abbr title="Adam with Decoupled Weight Decay">AdamW</abbr>）</dt>
  <dd>揭示了 Adam 与 $L_2$ 正则化混合导致的病态现象，提出了“解耦权重衰减”架构，彻底恢复了自适应优化器的泛化性能，成为当今全世界大语言模型标准训练配置。</dd>
</dl>

---

## 步骤 5：手把手超简单数字积木（单参数两步迭代纯手算）

为了让你彻底看清动量积累、二阶方差和偏差修正的每一步数值细节，我们用最简单的数字，手算单个权重参数 $\theta$ 的两轮完整更新。

### 1. 超微型超参数设定
- 初始权重：$\theta_0 = 1.0000$
- 学习率：$\eta = 0.10$
- 一阶动量衰减：$\beta_1 = 0.90$
- 二阶方差衰减：$\beta_2 = 0.99$
- 稳定常数：$\epsilon = 10^{-8} \approx 0$
- 权重衰减系数：$\lambda = 0.05$
- 初始动量状态：$m_0 = 0, v_0 = 0$

---

### 2. 第一轮优化 $t = 1$（遭遇较大梯度 $g_1 = 2.0$）

<fieldset>
<legend><strong>计算流程清单（第 1 步）</strong></legend>
<p><input type="checkbox" checked disabled> <strong>步骤 A：</strong> 更新一阶矩 $m_1$ 和二阶矩 $v_1$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 B：</strong> 执行冷启动偏差修正 $\hat{m}_1$ 与 $\hat{v}_1$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 C：</strong> 计算自适应步进比率 $u_1 = \hat{m}_1 / \sqrt{\hat{v}_1}$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 D：</strong> 执行解耦衰减并计算新参数 $\theta_1$。</p>
</fieldset>

#### 步骤 A：原始矩移动平均
- $m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = (0.90 \times 0) + (0.10 \times 2.0) = \mathbf{0.20}$
- $v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = (0.99 \times 0) + (0.01 \times 2.0^2) = 0.01 \times 4.0 = \mathbf{0.04}$

#### 步骤 B：消除零初始化偏差
- $\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.20}{1 - 0.90} = \frac{0.20}{0.10} = \mathbf{2.0000}$
- $\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.04}{1 - 0.99} = \frac{0.04}{0.01} = \mathbf{4.0000}$

<mark>看：偏差修正发挥了神效！它把被零拉扯的 $m_1$ 从 0.20 还原回了真实的 2.00，把 $v_1$ 从 0.04 还原回了 4.00！</mark>

#### 步骤 C：计算自适应步进方向
$$
u_1 = \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = \frac{2.0000}{\sqrt{4.0000}} = \frac{2.0000}{2.0000} = \mathbf{1.0000}
$$

#### 步骤 D：更新参数（融合解耦权重衰减）
$$
\begin{aligned}
\theta_1 &= \theta_0 - \eta \lambda \theta_0 - \eta u_1 \\
&= 1.0000 - (0.10 \times 0.05 \times 1.0000) - (0.10 \times 1.0000) \\
&= 1.0000 - 0.0050 - 0.1000 = \mathbf{0.8950}
\end{aligned}
$$

---

### 3. 第二轮优化 $t = 2$（梯度骤降为 $g_2 = 0.5$）

模型向山谷走近了一步，梯度从 $2.0$ 骤跌至 $g_2 = 0.50$。

#### 步骤 A：原始矩移动平均
- $m_2 = \beta_1 m_1 + (1 - \beta_1) g_2 = (0.90 \times 0.20) + (0.10 \times 0.50) = 0.18 + 0.05 = \mathbf{0.2300}$
- $v_2 = \beta_2 v_1 + (1 - \beta_2) g_2^2 = (0.99 \times 0.04) + (0.01 \times 0.50^2) = 0.0396 + 0.0025 = \mathbf{0.0421}$

#### 步骤 B：消除零初始化偏差（$t = 2$）
- $1 - \beta_1^2 = 1 - 0.90^2 = 1 - 0.81 = 0.19$
- $\hat{m}_2 = \frac{0.2300}{0.19} \approx \mathbf{1.2105}$
- $1 - \beta_2^2 = 1 - 0.99^2 = 1 - 0.9801 = 0.0199$
- $\hat{v}_2 = \frac{0.0421}{0.0199} \approx \mathbf{2.1156}$

#### 步骤 C：计算自适应步进方向
$$
u_2 = \frac{\hat{m}_2}{\sqrt{\hat{v}_2} + \epsilon} = \frac{1.2105}{\sqrt{2.1156}} \approx \frac{1.2105}{1.4545} \approx \mathbf{0.8322}
$$

注意观察：即使单步梯度暴跌了 $75\%$（从 $2.0$ 跌到 $0.5$），但因为保龄球积累了向前的冲量，实际更新步长 $u_2 = 0.8322$ 依然保持着饱满且平稳的推进力度！

#### 步骤 D：更新参数（融合解耦权重衰减）
$$
\begin{aligned}
\theta_2 &= \theta_1 - \eta \lambda \theta_1 - \eta u_2 \\
&= 0.8950 - (0.10 \times 0.05 \times 0.8950) - (0.10 \times 0.8322) \\
&= 0.8950 - 0.00448 - 0.08322 = \mathbf{0.8073}
\end{aligned}
$$

参数平滑而扎实地从 $1.0000 \to 0.8950 \to 0.8073$ 稳健逼近目标最优解。

---

## 步骤 6：核心精要（一句话记住核心奥秘）

> [!TIP] AdamW 优化器的核心心法
> **AdamW 是复杂高维损失峡谷中的终极导航向导：动量惯性帮助模型无视横向噪音坚定前行，二阶方差为每个参数动态定制专属油门与刹车。**
>
> 配合解耦的纯净权重衰减，AdamW 保证了全网 700 亿个参数在长达数月的海量训练中既不震荡发散，也不臃肿膨胀，是大模型炼丹过程中无可替代的数学动力引擎。
