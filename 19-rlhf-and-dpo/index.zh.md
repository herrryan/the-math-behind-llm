# 第 19 章：教会大模型懂礼貌（人类反馈强化学习 RLHF 与直接偏好优化 DPO）

---

## 步骤 1：3 岁孩子也能懂的直觉（博览群书的野生鹦鹉与小红花礼仪学校）

> [!INTUITION] 绝顶聪明却满嘴脏话的魔法鹦鹉
> 想象世界上有一只神奇的野外魔法鹦鹉。它飞遍了整个人类世界的每一个角落，把地球上所有的图书馆书籍、报纸、论坛聊天记录和路边涂鸦全都生吞活剥地背了下来：
>
> 1. **野生天才（无拘无束的预训练基座模型）**：
>    - 这只鹦鹉无所不知：它精通量子物理微积分、古巴比伦诗歌，也知道如何烤制松软的面包。
>    - 但由于它把整个互联网不加过滤地全吞进了肚子里，它**毫无教养与礼貌规矩**！
>    - 如果你礼貌地问它：<samp>“请问怎么做巧克力蛋糕？”</samp>，它可能会给你一个绝妙的食谱，但也可能突然顺着网线吐出一句恶毒的网络脏话，或者开始背诵冗长枯燥的版权免责声明！
>    - 它根本没有“帮助人类”的概念 &mdash; 它只是一台冷酷无情的复读机，盲目顺着概率概率最高的互联网文本继续往后胡乱拼凑。
>
> 2. **榜样示范教学（监督微调 &mdash; SFT）**：
>    - 首先，耐心的老师们写了一本金光闪闪的规范礼仪范本送给鹦鹉。
>    - 书里的每一页都是标准的人类问答范例：
>      - <kbd>问：</kbd> “如何烤制蛋糕？”
>      - <kbd>答：</kbd> “这是一个简单好吃的家庭食谱：首先，将烤箱预热到 180 度……”
>    - 鹦鹉学会了如何像一个懂礼貌的人类助手那样说话。但这还远远不够：如果遇到刁钻、危险、怀有恶意的提问怎么办？
>
> 3. **小红花裁判员（RLHF 与奖励模型）**：
>    - 我们邀请人类评审员给鹦鹉回答的两套不同答案挑优胜者。
>    - 回答既诚恳又安全 $\to$ 奖励一朵**小红花（<dfn id="def-reward-zh">奖励分数，Reward, $+1$</dfn>）**！
>    - 回答粗鲁或者暗藏危险 $\to$ 给出严肃皱眉警告（$0$ 朵小红花）。
>    - 接着，我们训练一个机器裁判员（奖励模型 RM），专门模拟打分。
>    - 随后用强化学习算法（PPO）微调鹦鹉的大脑，让它拼命多赚小红花。
>    - 此时，脖子上必须套上一根看不见的弹性牵引绳（KL 散度惩罚），防止鹦鹉为了骗取裁判员的小红花而投机取巧，或者把原本流利的正常母语遗忘光！
>
> 4. **震撼人心的数学捷径：直接偏好优化（DPO）**：
>    - 训练独立的裁判模型、在 GPU 显存里同时挂载四个庞然大物、还要调试极其脆弱暴躁的强化学习，成本高昂且极易崩溃。
>    - 2023 年，斯坦福大学的学者们从数学底层推导出了一个惊世定理：
>      <mark>“这只鹦鹉自己的大脑几率里，早已严格内嵌了那位机器裁判员的小红花账本！”</mark>
>    - 我们根本不需要单独雇佣裁判，也根本不需要复杂的强化学习！只需直接对比胜者答案和败者答案在模型脑海中的对数几率差，就能用最纯粹优美的二分类公式，一行代码把鹦鹉直接驯服！

<figure>
<pre>
大语言模型对齐（Alignment）技术演进图谱：

1. 预训练：     吞食海量互联网网页 ──► 盲目补全后续词元
2. 监督微调：   人类编写优质问答对 ──► 学会助理对话语气
3. 经典 RLHF:   提示词 ──► 当前模型 ──► 奖励模型打分 ──► PPO 策略更新
                             ▲                              │
                             └──────── KL 弹性牵引绳 ───────┘
4. 现代 DPO:    提示词 + [ 胜者答案 / 败者答案 ] ──► 纯二分类损失 ──► 更新权重
                （完全干掉奖励模型！彻底抛弃 PPO！100% 稳定收敛！）
</pre>
<figcaption><strong>图 19.1：</strong> 模型从盲目预训练走向多模型复杂 RLHF，最终升华为极简直接偏好优化（DPO）。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

> [!BRIDGING] 为什么模型对齐不能单纯依赖监督微调的交叉熵？
> 在第 15 章中，我们深入掌握了交叉熵损失函数（Cross-Entropy Loss），它驱动模型最大化目标词元的对数似然概率：
>
> $$
> \mathcal{L}_{\text{SFT}}(\boldsymbol{\theta}) = -\sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(y_t \mid x, y_{<t})
> $$
>
> 这个目标在让模型学会基本的问答格式与文风时非常有效（<abbr title="Supervised Fine-Tuning">SFT</abbr>）。但在真正的价值对齐阶段，它暴露出两大根本性绝症：
>
> 1. **监督数据无法表达“千万别做这事”**：交叉熵只能把模型往前拉向正向示范。如果我们想让模型不产生毒素、不输出危险攻击教程，在 SFT 框架下我们不得不编写“错误示范”，这反而等于在向模型直接传授如何作恶！
> 2. **人类的主观审美是相对的胜负，而不是绝对的打分**：如果让两位评审员给同一篇作文打绝对数值分数（比如“这篇应该打 8.4 还是 8.7？”），两人会争执不休；但如果把作文 A 和作文 B 并排放在眼前让他们选哪个更好，两人在 90% 以上的场景下都能达成高度共识！
>
> “我们如何在数学上将人类成对的相对胜负偏好（$y_w \succ y_l$）严密转化为优化目标，既能直接拔高优胜答案、强力压低失败答案，又能用数学定理死死锁住语言模型，绝不让它产生胡言乱语？”

---

## 步骤 3：严谨数学推导与公式

### 1. 布拉德利-特里成对偏好模型（Bradley-Terry Model, 1952）

给定一个人类输入的提示词 $x$，以及模型生成的两个候选答案：人类偏好的胜者答案 $y_w$ 与被否决的败者答案 $y_l$（记作 $y_w \succ y_l \mid x$）。

根据 **Bradley-Terry 模型**，人类倾向于选择 $y_w$ 而非 $y_l$ 的概率，由背后的潜在标量奖励函数 $r(x, y) \in \mathbb{R}$ 的差值唯一决定：

$$
P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right) = \frac{1}{1 + \exp\left(-(r(x, y_w) - r(x, y_l))\right)}
$$

其中 $\sigma(u) = \frac{1}{1 + e^{-u}}$ 为标准 Sigmoid 逻辑斯蒂函数。

---

### 2. 经典带 KL 约束的 RLHF 目标函数

在经典 RLHF 架构中（Christiano et al., 2017; Ouyang et al., 2022），研究人员首先通过人类成对标注数据，训练一个独立的奖励模型参数 $\phi$：

$$
\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]
$$

固定住奖励模型 $r_\phi$ 后，语言模型策略 $\pi_{\boldsymbol{\theta}}$ 通过强化学习（PPO）最大化期望奖励，同时挂载相对初始 SFT 冻结参考策略 $\pi_{\text{ref}}$ 的 KL 散度弹性绳惩罚：

$$
\max_{\pi_{\boldsymbol{\theta}}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\boldsymbol{\theta}}}\left[ r_\phi(x, y) \right] - \beta D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}}(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)
$$

其中：
- $\beta > 0$ 为 **KL 正则化强度**（弹性牵引绳的刚度）。
- $\pi_{\text{ref}}$ 是冻结不更新的初始 SFT 模型。
- $D_{\text{KL}}(P \parallel Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)} \ge 0$ 防止模型被强化学习带偏至崩溃。

---

### 3. 解析最优策略的闭式解

令人惊叹的是，上述带有 KL 约束的强化学习最优化问题，在变分法（Calculus of Variations）下存在唯一的**解析精确闭式解**！
在满足概率归一化条件 $\sum_y \pi(y \mid x) = 1$ 的约束下，全局最优策略 $\pi^*$ 的闭式解为：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是**配分函数（Partition Function）**（对所有可能生成的文本序列求和，序列总空间高达 $32000^{2048}$，在物理世界中根本无法直接计算！）。

---

### 4. DPO 的惊世突破：反解隐式奖励（Rafailov et al., 2023）

正因为配分函数 $Z(x)$ 无法求解，OpenAI 当年才被迫采用复杂的 PPO 循环采样逼近。

拉斐尔·拉法伊洛夫（Rafael Rafailov）等人在斯坦福大学灵光乍现：我们为什么不直接对最优策略闭式解进行代数变换，把隐藏的奖励 $r(x, y)$ 反解出来呢？

$$
\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{1}{Z(x)} \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

两边同时取自然对数 $\log$：

$$
\log \pi^*(y \mid x) - \log \pi_{\text{ref}}(y \mid x) = -\log Z(x) + \frac{1}{\beta} r(x, y)
$$

移项整理，即可用模型概率直接表达出**隐式奖励函数**：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

现在，把这个式子直接代入最初的 Bradley-Terry 人类偏好差值 $r(x, y_w) - r(x, y_l)$ 中：

$$
\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{aligned}
$$

<mark>奇迹发生了：包含天文数字求和项的无法计算的配分函数 $\beta \log Z(x)$，在相减的瞬间被完全抵消得无影无踪！</mark>

---

### 5. 直接偏好优化（DPO）损失函数

将这套优雅的差值直接代入 Bradley-Terry 负对数似然中，便诞生了名垂人工智能史册的 **DPO 损失函数**：

$$
\mathcal{L}_{\text{DPO}}(\boldsymbol{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma \left( \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

#### DPO 梯度更新动力学：
对参数 $\boldsymbol{\theta}$ 求导，剖析其数学内驱力：

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{DPO}} = -\beta \, \underbrace{\sigma\left(\hat{r}_{\boldsymbol{\theta}}(x, y_l) - \hat{r}_{\boldsymbol{\theta}}(x, y_w)\right)}_{\text{惊奇度误差加权 } (1 - \sigma)} \cdot \left[ \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_w \mid x)}_{\text{强力提升胜者 } y_w \text{ 的概率}} - \underbrace{\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_l \mid x)}_{\text{强力压制败者 } y_l \text{ 的概率}} \right]
$$

- 如果模型已经很懂事，正确赋予胜者更高奖励（$\hat{r}_w \gg \hat{r}_l$），误差权重 $\sigma(\hat{r}_l - \hat{r}_w) \approx 0$，梯度归零，不做多余打扰。
- 如果模型判断失误，错误地偏爱败者（$\hat{r}_l > \hat{r}_w$），误差权重逼近 1，以最大马力强推胜者、狠刹败者！

---

## 步骤 4：历史源流与思考演进（Bradley-Terry、InstructGPT 与 DPO）

<dl>
  <dt><time datetime="1952">1952</time> &mdash; <strong>拉尔夫·布拉德利 与 米尔顿·特里</strong></dt>
  <dd>在统计学界提出了著名的 Bradley-Terry 概率模型，用于分析锦标赛中的成对对决胜率，奠定了现代所有成对偏好排序算法的数学基石。</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>保罗·克里斯蒂亚诺 等人</strong>（<cite>《Deep Reinforcement Learning from Human Preferences》</cite>）</dt>
  <dd>首次证明可以用人类的成对比较偏好训练深度强化学习奖励模型，使虚拟机器人在没有任何预设数学代码目标的情况下，自发学会了优雅的空中后空翻。</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>OpenAI 团队</strong>（<cite>《InstructGPT / ChatGPT》</cite>）</dt>
  <dd>将 SFT $\to$ RM $\to$ PPO 的完整流水线成功移植到 GPT-3 上，创造了引爆全球的 ChatGPT，证明仅有 13 亿参数的对齐模型，在人类盲测满意度上彻底碾压了 1750 亿参数的未对齐基座大模型。</dd>

  <dt><time datetime="2023">2023</time> &mdash; <strong>拉斐尔·拉法伊洛夫 等人 / 斯坦福大学</strong>（<cite>《Direct Preference Optimization》</cite>）</dt>
  <dd>推导出了闭式解等价性，通过代数相消干掉了配分函数，彻底摆脱了复杂的强化学习环境，使全球开源大模型对齐效率提升了数十倍。</dd>
</dl>

---

## 步骤 5：手把手超简单数字积木（单轮偏好对齐纯手算）

为了让你彻底看清 DPO 是如何纯靠对数概率计算隐式奖励并驱动参数更新的，我们纯手算一套具体的成对对齐数值。

### 1. 微型场景设定
- 输入提示词：$x =$ <kbd>“请简述量子力学的核心。”</kbd>
- 胜者回答：$y_w$（精炼、通俗、物理事实准确）
- 败者回答：$y_l$（啰嗦、态度傲慢、概念混淆）
- 正则化系数：$\beta = 0.50$

### 2. 初始概率设定
设冻结的基准模型 $\pi_{\text{ref}}$ 和当前模型 $\pi_{\boldsymbol{\theta}}$ 对这两个句子的生成概率分别如下：
- 基准模型：
  - $\pi_{\text{ref}}(y_w \mid x) = 0.20$
  - $\pi_{\text{ref}}(y_l \mid x) = 0.10$
- 当前待对齐模型：
  - $\pi_{\boldsymbol{\theta}}(y_w \mid x) = 0.30$
  - $\pi_{\boldsymbol{\theta}}(y_l \mid x) = 0.40$
  *（看：当前模型犯了严重错误！它居然更喜欢糟糕回答 $y_l$，给出了高达 $0.40$ 的概率，而正确胜者 $y_w$ 只有 $0.30$！）*

---

### 3. DPO 损失与误差权重纯手算推导

<fieldset>
<legend><strong>计算流程清单</strong></legend>
<p><input type="checkbox" checked disabled> <strong>步骤 A：</strong> 计算两套答案在当前模型与基准模型下的概率比值。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 B：</strong> 计算对数比值并乘以 $\beta$，得出隐式奖励分数。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 C：</strong> 计算奖励差值 $\Delta r = \hat{r}(y_w) - \hat{r}(y_l)$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 D：</strong> 经过 Sigmoid 激活函数映射为获胜预测概率。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 E：</strong> 计算最终标量损失 $\mathcal{L}_{\text{DPO}} = -\log \sigma(\Delta r)$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 F：</strong> 计算反向传播的梯度误差权重 $\beta(1 - \sigma)$。</p>
</fieldset>

#### 步骤 A：概率比值
- 胜者回答 $y_w$ 的概率比：
  $$
  \frac{\pi_{\boldsymbol{\theta}}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} = \frac{0.30}{0.20} = \mathbf{1.5000}
  $$
- 败者回答 $y_l$ 的概率比：
  $$
  \frac{\pi_{\boldsymbol{\theta}}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} = \frac{0.40}{0.10} = \mathbf{4.0000}
  $$

#### 步骤 B：自然对数与隐式奖励
- 胜者对数比：
  $$
  \log(1.5000) \approx \mathbf{+0.4055}
  $$
- 胜者隐式奖励：
  $$
  \hat{r}(x, y_w) = \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_w)}{\pi_{\text{ref}}(y_w)} = 0.50 \times 0.4055 = \mathbf{+0.2028}
  $$
- 败者对数比：
  $$
  \log(4.0000) \approx \mathbf{+1.3863}
  $$
- 败者隐式奖励：
  $$
  \hat{r}(x, y_l) = \beta \log \frac{\pi_{\boldsymbol{\theta}}(y_l)}{\pi_{\text{ref}}(y_l)} = 0.50 \times 1.3863 = \mathbf{+0.6931}
  $$

#### 步骤 C：奖励差值
$$
\Delta r = \hat{r}(x, y_w) - \hat{r}(x, y_l) = 0.2028 - 0.6931 = \mathbf{-0.4904}
$$

差值是负数（$-0.4904$），表明当前模型在隐式奖励上严重奖罚倒挂！

#### 步骤 D：Sigmoid 映射
$$
\sigma(\Delta r) = \sigma(-0.4904) = \frac{1}{1 + e^{0.4904}} = \frac{1}{1 + 1.6330} = \frac{1}{2.6330} \approx \mathbf{0.3798}
$$

模型预测人类更喜欢胜者的信心只有可怜的 $38.0\%$！

#### 步骤 E：计算 DPO 标量损失
$$
\mathcal{L}_{\text{DPO}} = -\log(0.3798) \approx \mathbf{0.9681}
$$

#### 步骤 F：计算梯度反向推动力
乘在梯度更新方向上的误差权重大小为：

$$
\text{权重} = \beta \left( 1 - \sigma(\Delta r) \right) = 0.50 \times (1 - 0.3798) = 0.50 \times 0.6202 = \mathbf{0.3101}
$$

<mark>看：正因为当前模型判断严重失误（错误率高达 $62\%$），系统立刻产生了高达 $0.3101$ 的强劲梯度推力，强力把 $\pi_{\boldsymbol{\theta}}(y_w)$ 向上推拉，同时将 $\pi_{\boldsymbol{\theta}}(y_l)$ 向下狠狠打压！</mark>

---

## 步骤 6：核心精要（一句话记住核心奥秘）

> [!TIP] 价值对齐的核心心法
> **预训练塑造了大语言模型通晓万物的博学大脑，但对齐赋予了它温暖、正直与体面的人性底色。**
>
> 直接偏好优化（DPO）以令人叹为观止的数学对称美，证明了语言模型自身就是其专属的最佳奖励裁判，将曾经脆弱晦涩的强化学习旋钮，浓缩为一行纯粹优美的二分类损失，完成了大模型修炼之旅的终极蜕变。
