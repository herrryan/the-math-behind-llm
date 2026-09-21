# 实践实验 1：手把手带你写一个真实硬件 Roofline 性能剖析器

同学，你好！恭喜你完成了前面全部理论课程的学习！从今天开始，老师要带你挽起袖子，进入真正的代码实战工坊！

在第一模块中，我们花了大量时间学习了经典的**屋顶模型（Roofline Model）**。很多同学在纸上推导得头头是道，可一到自己的电脑上，就抓瞎了：“老师，我怎么才能测出我手头这块显卡的真实带宽和算力？怎么才能判断我写的算子到底有没有撞上内存墙？”

今天这节实验课，老师就手把手带你用 Python 和 PyTorch，从零编写一个**自动化硬件性能剖析器**，亲手画出属于你自己的屋顶模型折线图！

---

## 实验目标与产出

1. **测定硬件真身：** 测出当前 GPU 的实际物理峰值算力（TFLOPS）与 HBM/显存实际有效带宽（GB/s）；
2. **算子实测打点：** 编写微基准，测量大模型中最核心的线性层 GEMM 在不同 Batch Size 下的耗时；
3. **绘制性能天花板：** 自动计算算术强度 $I$，判断算子落入访存密集区还是计算密集区，并输出量化报告。

---

## 步骤 1：测试显存真实搬运带宽（Memory Copy Benchmark）

来，打开你的编辑器，跟老师敲下第一段测带宽的核心代码：

```python
import torch
import time

def measure_memory_bandwidth(size_bytes=1024 * 1024 * 512): # 512 MB
    # 通过纯显存数据搬运，测定真实物理带宽 (GB/s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("未检测到 GPU，跳过测试")
        return 0.0

    # 在 GPU 显存上分配源数据和目标数据
    num_elements = size_bytes // 4 # float32
    src = torch.empty(num_elements, dtype=torch.float32, device=device)
    dst = torch.empty(num_elements, dtype=torch.float32, device=device)

    # 预热 GPU
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()

    # 正式计时
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start

    # 每次 copy 包括一次读和一次写，总搬运量为 2 * size_bytes
    total_bytes_transferred = 2 * size_bytes * iters
    bandwidth_gb_s = (total_bytes_transferred / 1e9) / duration

    print(f"实测显存有效带宽: {bandwidth_gb_s:.2f} GB/s")
    return bandwidth_gb_s
```

---

## 步骤 2：测试张量核心峰值算力（Peak FLOPs Benchmark）

接下来，我们用一个巨大的矩阵乘法，把 GPU 的 Tensor Core 彻底喂饱，测出物理极限算力：

```python
def measure_peak_flops(m=8192, n=8192, k=8192):
    # 通过超大矩阵乘法测定实际峰值算力 (TFLOPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 采用标准半精度 FP16
    a = torch.randn(m, k, dtype=torch.float16, device=device)
    b = torch.randn(k, n, dtype=torch.float16, device=device)

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # 计时
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start

    # 单次 GEMM 浮点做功量: 2 * M * N * K
    total_flops = 2.0 * m * n * k * iters
    tflops = (total_flops / 1e12) / duration

    print(f"实测 FP16 峰值算力: {tflops:.2f} TFLOPS")
    return tflops
```

---

## 步骤 3：实战剖析不同工况的算术强度与瓶颈分析

现在，老师带你模拟大模型在 **Decode（小 Batch，单字生成）** 与 **Prefill（大 Batch，长提示词）** 下的表现：

```python
def profile_llm_layer(peak_tflops, bandwidth_gb_s):
    d_model = 4096
    d_ffn = 11008
    device = torch.device("cuda")

    # 屋顶模型的硬件拐点
    i_star = (peak_tflops * 1e12) / (bandwidth_gb_s * 1e9)
    print(f"\n当前硬件平衡拐点 I* = {i_star:.2f} FLOPs/Byte")

    test_cases = [
        ("自回归 Decode 阶段 (M=1)", 1),
        ("中等并发 Batch (M=32)", 32),
        ("预填 Prefill 阶段 (M=2048)", 2048)
    ]

    print(f"{'工况':<25} | {'算术强度 (FLOPs/Byte)':<20} | {'实际吞吐 (TFLOPS)':<18} | {'性能瓶颈判决'}")
    print("-" * 80)

    w = torch.randn(d_model, d_ffn, dtype=torch.float16, device=device)

    for desc, m in test_cases:
        x = torch.randn(m, d_model, dtype=torch.float16, device=device)
        
        # 预热
        for _ in range(10):
            y = torch.matmul(x, w)
        torch.cuda.synchronize()

        # 计时
        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            y = torch.matmul(x, w)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start

        # 计算理论指标
        flops = 2.0 * m * d_model * d_ffn
        # 访存量: 读 x, 读 w, 写 y (以 2 字节 FP16 计算)
        bytes_transferred = (m * d_model + d_model * d_ffn + m * d_ffn) * 2
        
        actual_i = flops / bytes_transferred
        achieved_tflops = (flops * iters / 1e12) / duration

        bound_type = "访存受限 (Memory-Bound)" if actual_i < i_star else "计算受限 (Compute-Bound)"
        print(f"{desc:<25} | {actual_i:<20.2f} | {achieved_tflops:<18.2f} | {bound_type}")

if __name__ == "__main__":
    bw = measure_memory_bandwidth()
    flops = measure_peak_flops()
    profile_llm_layer(flops, bw)
```

---

## 老师点评与课后思考题

1. **观察现象：** 当你运行这段代码时，你会震撼地发现，在 $M=1$ 的自回归 Decode 工况下，实际算力利用率通常连理论峰值的 5% 都达不到！  
2. **课后思考：** 尝试把数据精度从 `torch.float16` 换成 `torch.float32`，观察硬件平衡拐点 $I^*$ 发生了什么变化？为什么大模型一定要用低精度？
