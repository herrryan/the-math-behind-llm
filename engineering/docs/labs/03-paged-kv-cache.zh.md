# 实践实验 3：构建 PagedAttention 虚拟内存管理器：从页表映射到零碎显存池

同学，你好！在第一模块第 E05 章中，我们被 vLLM 那套如同操作系统虚拟内存一般的 **PagedAttention 显存池**深深震撼。

今天这节实验课，老师要带你扮演一次“显存操作系统的架构师”！  
我们将用纯 Python 代码，亲手构建一套生产级推理引擎核心的**动态分页 KV Cache 内存管理器**。

---

## 实验目标与产出

1. **构建物理块池（Block Pool）：** 初始化固定大小的预分配连续物理显存块，维护空闲块双向队列；
2. **构建逻辑-物理页表（Block Table）：** 为每个并发进入的会话请求动态分配、映射和扩展物理块；
3. **实现写时复制与释放（CoW & Free）：** 模拟多轮对话分支的页表共享，以及请求结束时显存块的纳秒级回收。

---

## 步骤 1：定义物理显存块管理器（BlockAllocator）

来，跟着老师写下显存池的核心分配器：

```python
from typing import List, Dict
import collections

class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size  # 每个物理块容纳的 Token 数量 (通常设为 16)
        self.ref_count = 0            # 引用计数 (用于写时复制与前缀共享)

    def is_free(self) -> bool:
        return self.ref_count == 0

class BlockAllocator:
    # 全局显存物理块池 (模拟连续 GPU 显存池)
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.all_blocks = [PhysicalBlock(i, block_size) for i in range(num_blocks)]
        # 空闲队列
        self.free_queue = collections.deque(self.all_blocks)

    def allocate(self) -> PhysicalBlock:
        if not self.free_queue:
            raise MemoryError("GPU 显存物理块已耗尽！触发显存 OOM 熔断！")
        block = self.free_queue.popleft()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock):
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_queue.append(block)
        elif block.ref_count < 0:
            raise ValueError("严重系统逻辑错误：物理块引用计数小于 0！")

    def get_num_free_blocks(self) -> int:
        return len(self.free_queue)
```

---

## 步骤 2：为每个请求构建逻辑页表（SequenceBlockTable）

现在我们为用户请求建立动态页表：

```python
class Sequence:
    def __init__(self, seq_id: int, prompt_tokens: List[int]):
        self.seq_id = seq_id
        self.tokens = list(prompt_tokens)
        self.block_table: List[PhysicalBlock] = []

    def num_tokens(self) -> int:
        return len(self.tokens)

class PagedCacheManager:
    def __init__(self, num_blocks: int = 100, block_size: int = 16):
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.active_sequences: Dict[int, Sequence] = {}

    def register_sequence(self, seq_id: int, prompt_tokens: List[int]) -> Sequence:
        seq = Sequence(seq_id, prompt_tokens)
        # 计算初始提示词需要多少个物理块
        num_blocks_needed = (len(prompt_tokens) + self.block_size - 1) // self.block_size
        for _ in range(num_blocks_needed):
            block = self.allocator.allocate()
            seq.block_table.append(block)
        self.active_sequences[seq_id] = seq
        return seq

    def append_token(self, seq_id: int, new_token: int):
        # 自回归生成 1 个新 Token 时的动态扩容逻辑
        seq = self.active_sequences[seq_id]
        curr_len = seq.num_tokens()
        
        # 判断当前最后一个物理块是否已经装满
        if curr_len % self.block_size == 0:
            # 刚好装满，向显存池申请一个全新的空闲块！
            new_block = self.allocator.allocate()
            seq.block_table.append(new_block)
            print(f"[请求 {seq_id}] 步进触发跨页，成功挂载新物理块 ID: {new_block.block_id}")
            
        seq.tokens.append(new_token)

    def terminate_sequence(self, seq_id: int):
        # 请求生成完毕，释放全部物理块回空闲池
        seq = self.active_sequences.pop(seq_id)
        for block in seq.block_table:
            self.allocator.free(block)
        print(f"[请求 {seq_id}] 对话结束，成功无损释放 {len(seq.block_table)} 个物理块！")
```

---

## 步骤 3：完整运行模拟与零碎片验证

```python
if __name__ == "__main__":
    print("=== 启动 PagedAttention 显存虚拟化管理器实战 ===")
    manager = PagedCacheManager(num_blocks=10, block_size=4) # 设每个块放 4 个字

    print(f"初始可用空闲块数: {manager.allocator.get_num_free_blocks()}")

    # 1. 用户 A 带着 7 个字的 Prompt 进入系统
    seq_a = manager.register_sequence(seq_id=1, prompt_tokens=[10, 20, 30, 40, 50, 60, 70])
    print(f"用户 A (7 字) 分配块数: {len(seq_a.block_table)} (物理块 ID: {[b.block_id for b in seq_a.block_table]})")
    print(f"当前剩余空闲块数: {manager.allocator.get_num_free_blocks()}")

    # 2. 用户 A 持续吐字，触发边界扩页
    print("\n--- 用户 A 开始吐字 ---")
    manager.append_token(seq_id=1, new_token=80)  # 达到 8 字，刚好填满第 2 块
    manager.append_token(seq_id=1, new_token=90)  # 达到 9 字，动态申请第 3 块！

    # 3. 用户 B 带着 3 个字进入
    seq_b = manager.register_sequence(seq_id=2, prompt_tokens=[100, 200, 300])
    print(f"\n用户 B (3 字) 分配块数: {len(seq_b.block_table)} (物理块 ID: {[b.block_id for b in seq_b.block_table]})")
    print(f"当前剩余空闲块数: {manager.allocator.get_num_free_blocks()}")

    # 4. 用户 A 任务完成，退出系统
    print("\n--- 用户 A 结束对话 ---")
    manager.terminate_sequence(seq_id=1)
    print(f"释放后全系统空闲块数迅速回升至: {manager.allocator.get_num_free_blocks()}")
```

---

## 老师点评与课后思考题

1. **观察精妙之处：** 用户 A 和用户 B 的物理块在真实内存中是完全交织甚至乱序的，但每个用户眼里的逻辑页表都是连续的 0 到 $N$。系统内部**彻底消灭了外部显存碎片**！
2. **课后挑战：** 试着在 `BlockAllocator` 中实现**引用计数共享机制**：如果用户 C 和用户 A 拥有相同的前 10 个 Token，如何让它们共享同一个物理块？
