# 第 E13 章：KV Cache 显存量化压缩：将历史上下文缓存压缩至 FP8 与 INT4 的硬件收益与保真度

## 步骤 1：物理直觉

想象你必须为每一位在图书馆自习的读者留下一张写满摘要的便利贴：
- **原始方案（BF16）：** 每张便利贴都用大号硬卡纸打印，写了 1000 页书，便利贴就堆成了一座大山，自习室的桌子全部被纸张压垮了。
- **低比特压缩方案（FP8 / INT4）：** 换用极薄的便签纸，并用微型代号速记。整整 1000 页的摘要现在只需要一个小巴掌大的便签盒就能全部装下！

在长文本问答（如 32K 乃至 128K 上下文）中，**KV Cache 消耗的显存很快就会大幅反超模型权重自身**！将 KV Cache 压缩为 FP8 或 INT4，就像给自习室腾出了 75% 的空间，允许服务引擎同时接待 4 倍以上的并发读者！

---

## 步骤 2：芯片微观底层执行机制

KV Cache 量化的核心是在写出与读入环节插入瞬时量化算子：

```
写入流程 (当前 Token 生成新的 Key / Value 向量):
[新 K/V 向量: FP16] ──► [SRAM 中计算局部最大值并量化为 FP8 / INT4] ──► [写入 PagedAttention 显存池]
                                                                        (显存写入流量减少 50% - 75%!)

读取流程 (随后的注意力计算):
[从显存池加载压缩后的 K/V] ──► [在 SRAM 中反量化或直接送入低精度 Tensor Core] ──► [计算注意力分数]
```

### 两种量化粒度权衡：
1. **Per-Tensor（张量级量化）：** 整个层共用一个缩放标量。实现极简，但面对长序列中的异常值容易产生精度崩溃。
2. **Per-Head / Per-Token（分头/逐词量化）：** 为每个注意力头或每个 Token 独立分配一个 FP32 缩放因子。能完美保留注意力分数的长尾分布，精度近乎无损。

---

## 步骤 3：跨组件相互耦合机制

1. **解除在线推理的并发上限：**  
   当 KV Cache 尺寸减半（FP8）或缩小四分之三（INT4）时，GPU 显存池能容纳的 Block 数量成倍增加，直接将推理服务的最大并发吞吐推高 $2\text{--}4\times$。
2. **FlashAttention 低精度内核适配：**  
   在 FlashAttention-3 中，如果输入与 KV Cache 均为 FP8，硬件可以直接在 Hopper 的 FP8 Tensor Core 上以 $2\times$ 算力执行点积，彻底消除反量化带来的中间开销。

---

## 步骤 4：精确性能数学公式

单 Token 的 KV Cache 显存占用公式：

$$\text{Memory}_{\text{kv\_quant}} = 2 \times b \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \text{ 字节}$$

其中数值位宽 $b$：
- BF16: $b = 2$ 字节
- FP8: $b = 1$ 字节（节省 $50\%$）
- INT4: $b = 0.5$ 字节（节省 $75\%$）

长上下文下单步解码的显存总线搬运量：

$$\text{Bytes}_{\text{decode}} = \text{Weights} + \text{Memory}_{\text{kv\_quant}}(S)$$

当上下文长度 $S$ 极大时，$\text{Memory}_{\text{kv}}$ 占支配地位，解码耗时直接与位宽 $b$ 成正比缩短！

---

## 步骤 5：具体微基准数字推导

以 LLaMA-3 70B 模型在处理上下文长度 $S = 65536$（64K 超长文本）、批大小 $B = 8$ 时手算：
- 层数 $n_{\text{layers}} = 80$，$n_{\text{kv\_heads}} = 8$，$d_{\text{head}} = 128$

### 1. BF16 精度下的 KV Cache 消耗：
$$\text{单 Token} = 2 \times 2 \times 80 \times 8 \times 128 = 320 \text{ KB}$$
$$\text{总显存} = 8 \times 65536 \times 320 \text{ KB} \approx 167.8 \times 10^6 \text{ KB} \approx 167.8 \text{ GB}!$$
*即使拥有两张 80GB 的 GPU，光是装这 8 个人的长上下文缓存就会当场 OOM 崩溃！*

### 2. 采用 INT4 KV Cache 量化后：
$$\text{单 Token} = 2 \times 0.5 \times 80 \times 8 \times 128 = 80 \text{ KB}$$
$$\text{总显存} = 8 \times 65536 \times 80 \text{ KB} \approx 41.9 \text{ GB}$$
*显存直接省出整整 $125.9\text{ GB}$！不仅彻底消除了 OOM，还允许将并发批大小进一步扩大！*

---

## 步骤 6：核心系统工程铁律

> 在长上下文大模型时代，显存容量的真正主宰是 KV Cache 而不是模型参数。对模型权重做低比特量化只能释放固定的显存空间，而对 KV Cache 实施低比特量化，才能打破长文本并发扩展的物理天花板。
