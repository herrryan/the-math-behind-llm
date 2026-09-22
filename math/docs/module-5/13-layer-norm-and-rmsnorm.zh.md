# 第 13 章：让大家平静下来（LayerNorm 与 RMSNorm 归一化）


## 第 1 步：3 岁孩子也能懂的直觉（幼儿园合唱团的神奇智能调音台） {: #step-1 }

!!! note "3岁小孩的直觉: 神奇的调音台与智能音量旋钮"
    想象在幼儿园的汇报演出舞台上，三十个活泼可爱的小朋友正站成一排准备大合唱。

    如果舞台上的麦克风没有任何声音管理机制，现场很快就会陷入彻底的大混乱：
    - 有个害羞的小男孩嘴唇动个不停，但声音微弱得像小蚊子哼哼，台下的爸爸妈妈连一个字都听不清。
    - 旁边有个兴奋的小女孩对着麦克风拼命大喊大叫，尖锐刺耳的声音瞬间震得所有人耳朵生疼，音箱里甚至发出了刺耳的啸叫！
    - 慢慢地，其他小朋友为了不被别人的声音盖过去，也开始一个比一个扯着嗓子大吼，整场优美的合唱瞬间变成了一场失控的噪音风暴。

    为了解决这个危机，学校请来了一位经验丰富的调音师，带来了一台**智能调音台**：

    1. 只要有小朋友一开口唱歌，调音台就会在万分之一秒内听到声音的大小。
    2. 如果一个小女孩兴奋地喊出了 100 级的超大分贝，调音台就会瞬间把她的音量旋钮轻轻拧小，降到一个舒服平稳的数值（比如 5 级）。
    3. 如果一个小男孩怯生生地哼出了 0.1 级的小声音，调音台就会温柔地帮他推大音量旋钮，同样平稳地升到 5 级。
    4. 现在，台下听众听到的每一个稚嫩的声音，都处在完全一致、清澈悦耳的舒适能量区间！谁也不会淹没谁，音箱也永远不会过载失真。

    在大语言模型中，在数十层网络深处穿梭流转的数字也面临着一模一样的危险。如果没有音量旋钮，某些维度的数值就会失控暴涨到几千几万，而另一些维度则萎缩成看不见的粉尘。

    这个神奇的音量旋钮，在数学上就叫做**归一化机制**（如 <dfn id="def-layernorm-zh">LayerNorm</dfn> 与 <dfn id="def-rmsnorm-zh">RMSNorm</dfn>）。它就像静静守在每一个 Transformer 网络层大门口的专职调音师，把每一个词元身上的数字抚平到一个稳定、清晰、温和的标准状态！

<figure>
<pre>
未归一化的激活值（剧烈且危险的能量失衡）：

词向量：[ 1420.5,  -890.2,   0.001,   4500.8 ] ──► 音箱失真啸叫！
                                                    （梯度瞬间炸裂）

智能归一化调音台（温和平稳、尺度统一的标准化信号）：

词向量 ──► [ 实时测量整体能量与振幅 ]
                    │
                    ▼
           [ 整体除以离散振幅 ] ──► [ 0.81, -0.62, -0.11, 1.45 ]
                                    （平稳、可预测、易于优化！）
</pre>
<figcaption><strong>图 13.1：</strong> 归一化如同自动动态增益控制器，确保进入每个子层运算的数据尺度保持平稳，彻底消除数值失控风险。</figcaption>
</figure>

---

## 第 2 步：承前启后的关键过渡 {: #step-2 }

!!! question "计算连接问题: 为什么神经网络的数值会失控爆炸或萎缩？"
    在第 12 章中，我们发现残差连接构筑了一条加法高速公路：



    $$
    \mathbf{x}_{l} = \mathbf{x}_{l-1} + \mathcal{F}_l(\mathbf{x}_{l-1})
    $$



    由于每一层都是在原有向量的基础上**累加**新的特征增量，随着网络层数堆叠到 30 层、60 层甚至 80 层，向量 $\mathbf{x}_l$ 的整体欧氏模长（数值绝对值）往往会像滚雪球一样不断膨胀。

    当进入深层时，如果一个向量的各分量已经膨胀到了数百甚至上千：
    1. 在注意力层中，点积 $\mathbf{q}^\top \mathbf{k}$ 将会变得巨大无比，直接把 Softmax 函数推入两侧极端平坦的饱和区，导致导数几乎全为 $0.0000$。
    2. 在非线性激活函数（如 GELU、SwiGLU）中，极端数值会滑入饱和区或单一线性区，使模型丧失对细微语义差别的敏锐感知力。

    在早期的计算机视觉中，人们广泛使用**批归一化（Batch Normalization, BatchNorm）**，通过计算同一批次中多张不同图片的统计均值和方差来稳定数据。
    但在大语言模型中，BatchNorm 彻底失灵了：
    - 文本句子的长短极其多变（从 3 个词到 8192 个词不等）。
    - 在大模型生成推理时，系统是一个词元一个词元自回归输出的（Batch Size = 1），根本不存在可以统计的批次数据！

    “我们究竟如何设计一个完全聚焦于单个词元内部各维度、与批次大小及句子长度完全解耦的归一化算子？稳定模型训练最精简、最高效的数学运算又是什么？”

---

## 第 3 步：严谨数学推导与公式 {: #step-3 }

### 1. 标准层归一化（Layer Normalization, Ba 等人，2016）

<dfn id="def-ln-math-zh">层归一化（LayerNorm）</dfn>沿着每个词元自身的隐藏特征维度 $d_{\text{model}}$ 进行独立归一化，与批量维度彻底解耦。

给定一个输入词元向量 $\mathbf{x} = [x_1, x_2, \dots, x_d]^\top \in \mathbb{R}^d$（其中 $d = d_{\text{model}}$）：

#### 步骤 A：计算当前词元的均值（Mean）
求该向量所有维度的算术平均值：



$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i
$$



#### 步骤 B：计算方差（Variance）
计算各维度偏离均值的离散程度：



$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
$$



#### 步骤 C：零均值化与单位方差缩放
将特征中心平移至零，并除以标准差（加上极小微量 $\epsilon \approx 10^{-5}$ 以防除以零）：



$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$



#### 步骤 D：可学习的仿射缩放与平移变换
为防止归一化抹杀特征原本具有的独特表达分布，LayerNorm 引入了两个与维度大小相同的可学习参数：



$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$



其中：
- $\boldsymbol{\gamma} \in \mathbb{R}^d$ 是可学习的增益（缩放）向量，初始化为全 $1$。
- $\boldsymbol{\beta} \in \mathbb{R}^d$ 是可学习的偏置（平移）向量，初始化为全 $0$。

---

### 2. 现代大模型新标配：均方根归一化（RMSNorm, Zhang & Sennrich, 2019）

2019 年，张标（Biao Zhang）与 Rico Sennrich 开展了深入的理论与实验探究，得出了一个震撼业界的结论：
**层归一化中将中心平移至零均值的去均值操作（$x_i - \mu$），对保证训练稳定性几乎毫无用处！真正发挥压制数值失控作用的，完全是根据整体模长进行缩放（均方根 Root Mean Square）的能力。**

基于此，他们去掉了均值计算 $\mu$ 和偏置向量 $\boldsymbol{\beta}$，发明了 <dfn id="def-rmsnorm-math-zh">均方根归一化（RMSNorm）</dfn>。该算法现已成为 LLaMA 1/2/3、Mistral、Gemma、DeepSeek 等几乎所有前沿大模型的统一工业标准！

#### 步骤 A：计算均方根（RMS）
计算该向量各维度的二次平均值：



$$
\operatorname{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
$$



#### 步骤 B：向量缩放与增益重整
直接用原向量除以均方根，并乘上可学习的增益参数 $\gamma_i$：



$$
y_i = \frac{x_i}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma_i
$$



向量紧凑表示为：



$$
\mathbf{y} = \frac{\mathbf{x}}{\operatorname{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}
$$



<details>
<summary><strong>为什么现代大模型全面摒弃均值 $\mu$ 与偏置 $\beta$？</strong></summary>
1. **GPU 显存带宽极致优化（Memory-Bound 破局）**：
   在现代 GPU 上，归一化运算受限于**显存带宽**（即数据在板载显存 HBM 与芯片片上缓存 SRAM 之间的搬运速率），而非计算单元算力。
   - 标准 LayerNorm 需要对向量进行**两次完整的数据扫描**：第一遍算均值 $\mu$，第二遍算 $(x_i - \mu)^2$。
   - RMSNorm 只需要**单次扫描**即可直接算出 $\sum x_i^2$！
   这极大减少了 GPU 片上缓存的读写等待，在定制的 CUDA/Triton 算子中能够实现 **10% 到 50% 的实际速度提升**。
2. **彻底摒弃偏置项（No-Bias 架构）**：
   现代主流大模型在注意力投影、全连接层与归一化中，已经全面移除了偏置参数 $\mathbf{b}$。移除了 $\boldsymbol{\beta}$ 的 RMSNorm 天生强化了以零为原点的几何对齐，大幅简化了多卡张量并行（Tensor Parallelism）的通信同步。
</details>

---

## 第 4 步：历史源流与思考演进 {: #step-4 }

<figure>
<pre>
神经网络归一化技术的演进时间轴：

2015: Ioffe & Szegedy ────► 批归一化 (BatchNorm)
                            沿批次 (N) 维度统计；在可变长文本与自回归推理中折戟。
      │
      ▼
2016: Ba, Kiros, Hinton ──► 层归一化 (LayerNorm)
                            沿隐藏特征 (d) 维度独立统计，成为 GPT-2 与 BERT 的基石。
      │
      ▼
2019: Zhang & Sennrich ───► 均方根归一化 (RMSNorm)
                            证明中心化多余，单次扫描均方根提速 10%~50%；
                            成为现代 LLaMA、Mistral、DeepSeek 统一标配。
</pre>
<figcaption><strong>图 13.2：</strong> 从跨样本统计的批归一化，到片上单遍扫描的高性能 RMSNorm 演变脉络。</figcaption>
</figure>

### 1. 归一化流派横向全面对比

<fieldset>
<legend><strong>大模型归一化架构横向评测</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 13.1：</strong> 归一化方法核心属性与性能横向对比。</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">技术方案</th>
      <th align="center">数学缩放核心</th>
      <th align="center">可学习参数量</th>
      <th align="center">显存读取遍数</th>
      <th align="left">大模型工业界落地现状</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>批归一化 (BatchNorm)</strong></td>
      <td align="center">$\frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}}$</td>
      <td align="center">$\gamma, \beta$</td>
      <td align="center">多遍全局同步</td>
      <td>
        <del>完全不适用于现代大模型。</del> 无法应对动态变长文本，且在自回归单词元生成时直接失效。
      </td>
    </tr>
    <tr>
      <td><strong>经典层归一化 (LayerNorm)</strong></td>
      <td align="center">$\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$</td>
      <td align="center">$2d$（$\boldsymbol{\gamma}, \boldsymbol{\beta}$）</td>
      <td align="center">2 遍完整访存</td>
      <td>
        <del>上一代历史标准。</del> 广泛见于 GPT-2、GPT-3、BERT。收敛稳健，但带来不必要的显存带宽开销。
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>均方根归一化 (RMSNorm)</strong></td>
      <td align="center">$\frac{x}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma$</td>
      <td align="center"><ins><strong>$d$（仅 $\boldsymbol{\gamma}$）</strong></ins></td>
      <td align="center"><ins><strong>1 遍单次访存</strong></ins></td>
      <td>
        <ins><strong>现代开源与闭源前沿模型的绝对标准！</strong></ins> 全面赋能 LLaMA 1/2/3、Mistral、DeepSeek-V2/V3、Qwen。算子更轻、速度更快。
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

## 第 5 步：手把手超简单数字积木（三维向量的 LayerNorm 与 RMSNorm 纯手算） {: #step-5 }

为了让你彻底看清两者的数值差异，我们用一个微型三维向量，手动执行全部加减乘除计算。

### 1. 初始输入设定

设输入词元向量维度为 $d = 3$，当前激活值为：



$$
\mathbf{x} = \begin{bmatrix} 2.0 \\ 4.0 \\ 6.0 \end{bmatrix}
$$



为让手算过程一目了然，设定平滑项 $\epsilon = 0$，增益参数 $\boldsymbol{\gamma} = [1.0, 1.0, 1.0]^\top$，偏置参数 $\boldsymbol{\beta} = [0.0, 0.0, 0.0]^\top$。

---

### 2. 计算标准 LayerNorm

#### 步骤 A：计算均值 $\mu$


$$
\mu = \frac{2.0 + 4.0 + 6.0}{3} = \frac{12.0}{3} = 4.0
$$



#### 步骤 B：计算方差 $\sigma^2$ 与标准差 $\sigma$
计算各分量距离均值 $4.0$ 的差量平方：
- $x_1 - \mu = 2.0 - 4.0 = -2.0 \implies (-2.0)^2 = 4.0$
- $x_2 - \mu = 4.0 - 4.0 = 0.0 \implies (0.0)^2 = 0.0$
- $x_3 - \mu = 6.0 - 4.0 = 2.0 \implies (2.0)^2 = 4.0$

求平方差的均值：



$$
\sigma^2 = \frac{4.0 + 0.0 + 4.0}{3} = \frac{8.0}{3} \approx 2.6667
$$



标准差为：



$$
\sigma = \sqrt{\frac{8}{3}} \approx 1.6330
$$



#### 步骤 C：执行零均值化与缩放
各分量减去均值后除以标准差 $1.6330$：
- $\hat{x}_1 = \frac{-2.0}{1.6330} \approx -1.2247$
- $\hat{x}_2 = \frac{0.0}{1.6330} = 0.0000$
- $\hat{x}_3 = \frac{2.0}{1.6330} \approx 1.2247$



$$
\mathbf{y}_{\text{LayerNorm}} = \begin{bmatrix} -1.2247 \\ 0.0000 \\ 1.2247 \end{bmatrix}
$$



<mark>看：LayerNorm 输出的均值精确为 0，方差精确归一化为 1.0！</mark>

---

### 3. 计算现代 RMSNorm

现在，我们对完全相同的输入 $\mathbf{x} = [2.0, 4.0, 6.0]^\top$ 执行 RMSNorm。

#### 步骤 A：直接计算各分量平方和
无需任何减均值步骤，直接平方：
- $x_1^2 = 2.0^2 = 4.0$
- $x_2^2 = 4.0^2 = 16.0$
- $x_3^2 = 6.0^2 = 36.0$

求和：$4.0 + 16.0 + 36.0 = 56.0$。

#### 步骤 B：计算均方根 $\operatorname{RMS}(\mathbf{x})$


$$
\operatorname{MeanSquare} = \frac{56.0}{3} \approx 18.6667
$$





$$
\operatorname{RMS}(\mathbf{x}) = \sqrt{18.6667} \approx 4.3205
$$



#### 步骤 C：直接执行缩放
将每个原始分量直接除以 $\operatorname{RMS}(\mathbf{x}) \approx 4.3205$：
- $y_1 = \frac{2.0}{4.3205} \approx 0.4629$
- $y_2 = \frac{4.0}{4.3205} \approx 0.9258$
- $y_3 = \frac{6.0}{4.3205} \approx 1.3887$



$$
\mathbf{y}_{\text{RMSNorm}} = \begin{bmatrix} 0.4629 \\ 0.9258 \\ 1.3887 \end{bmatrix}
$$



<mark>看：所有数字依然保持着原本自然的正向相对比例，但整体二次能量被精确驯服钳位到了 1.0 左右！整个过程无需减均值，硬件计算单遍完成！</mark>

---

## 第 6 步：核心精要（一句话记住核心奥秘） {: #step-6 }

!!! tip "核心要点: 归一化技术的终极心法"
    **归一化如同立在每个 Transformer 层门前的智能调音台，将失控膨胀的特征振幅重新校准至平稳区间，使注意力点积与激活函数始终运行在最佳梯度工作点。**

    现代大模型通过拥抱**RMSNorm**，果断舍弃无实质收益的去均值计算，以极其精简的数学结构释放了宝贵的 GPU 显存带宽，实现了极致的训练稳定性与吞吐提速。
