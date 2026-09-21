# 实验 E1：GPU 屋顶模型性能分析器开发实战

## 实验目标与挑战

在实际工程开发中，90% 的初级工程师在遇到模型运行缓慢时，都会盲目尝试各种互不相干的优化技巧（量化、剪枝、多线程、重构循环）。但如果一个算子当前深陷在显存带宽受限区（Memory-Bound），任何尝试精简乘加代数运算的努力在物理上都是 100% 的无用功。

**本实验的核心任务：** 从零编写一个纯 Python/PyTorch 的轻量级**屋顶模型自动化性能分析器（Roofline Profiler）**。
1. 拦截目标 PyTorch 算子的底层前向执行；
2. 利用 `torch.profiler` 或底层计时器与张量形状推导，精确捕获其实际执行消耗的浮点运算量（FLOPs）与全局显存搬运字节数（Bytes）；
3. 计算该算子的实际算术强度 $I = \frac{\text{FLOPs}}{\text{Bytes}}$；
4. 结合当前物理 GPU（如 H100、A100 或 RTX 4090）的硬件规格，自动绘制该算子在屋顶模型性能边界上的真实坐标，并给出针对性的工业级架构优化建议。

---

## 动手实践代码框架

```python
import torch
import time

def profile_operator_roofline(op_fn, sample_inputs, hardware_peak_tflops, hardware_bw_gbs):
    # 分析指定算子的屋顶模型位置与瓶颈
    # hardware_peak_tflops: 物理单卡 FP16/BF16 峰值算力 (TFLOPS)
    # hardware_bw_gbs: 物理单卡显存吞吐带宽 (GB/s)
    # 1. 硬件平衡拐点
    hardware_balance_point = (hardware_peak_tflops * 1e12) / (hardware_bw_gbs * 1e9)
    print(f"[硬件诊断] 硬件临界拐点 I*: {hardware_balance_point:.2f} FLOPs/Byte")
    
    # 2. 预热 GPU
    for _ in range(10):
        _ = op_fn(*sample_inputs)
    torch.cuda.synchronize()
    
    # 3. 测量物理执行耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_event.record()
    for _ in range(iters):
        _ = op_fn(*sample_inputs)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_latency_ms = start_event.elapsed_time(end_event) / iters
    print(f"[基准测试] 实测平均延迟: {avg_latency_ms:.4f} ms")
    
    # 后续任务: 结合算子理论推导 FLOPs 与 Bytes，输出诊断报告!
```

---

## 思考与检验题
1. 运行实验代码测试 `torch.nn.functional.rms_norm`，观察其算术强度是否严格低于 2.0？
2. 增大批大小 $B$，观察标准矩阵乘法 `torch.matmul` 的算术强度是如何一步步从访存受限区跨越至计算受限区的。
