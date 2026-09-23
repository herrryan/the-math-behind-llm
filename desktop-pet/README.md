# AI 桌面精灵 (Cyber Clippy 2.0)

一个运行在 macOS / Windows 屏幕最上层、完全透明、感知你工作状态的现代 AI 陪伴桌面小精灵。

---

## 核心特性

1. **环境感知（Context Awareness）**：
   - **前台应用识别**：实时感知你当前在写代码（VS Code、Xcode、Cursor、Neovim）、敲终端（Ghostty、Terminal、iTerm2）、浏览网页还是社交摸鱼。
   - **剪贴板报错捕获**：一旦你在键盘上复制了包含 `Traceback`、`Exception`、`Panic`、`Segmentation fault` 的报错信息，精灵会立刻在屏幕上弹出吐槽与排查提醒。
   - **疲劳计时器**：高强度专注超过 45 分钟未切换窗口时，主动提醒喝水与活动颈椎。
2. **多重人格自由切换（Right Click 菜单）**：
   - 📎 **经典曲别针 (Clippy)**：热情怀旧的 Office 办公助手，热衷于帮你整理思路；
   - 🐱 **赛博猫咪 (Cyber Cat)**：高冷傲娇，把光标当老鼠抓，偶尔踩你键盘索要小鱼干；
   - ⚡ **老炮黑客 (Cyber Hacker)**：喝着双倍浓缩冰美式、对任何代码架构都挑剔的毒舌前辈；
   - 🍵 **禅意导师 (Zen Master)**：手捧热茶，提醒你深呼吸、放下焦虑、代码如浮云。
3. **纯代码程序化矢量动画（Zero External Assets）**：
   - 采用纯 PyQt6 `QPainter` 矢量绘制，无需下载任何图片或 GIF 资源，完美支持 Retina 视网膜高清屏。
   - 支持眨眼、待机呼吸浮动、耳朵轻颤、夜视镜发光与蒸汽动态。
4. **即插即用的端侧大模型接口（Pluggable Local Brain）**：
   - **离线模式（默认）**：内置超百条零延迟情境语料库，完全断网秒级响应，0 显存占用；
   - **本地大模型模式**：支持一键接入 Ollama、LM Studio、vLLM 或 llama.cpp 等本地 OpenAI 兼容端点（如 `qwen2.5:0.5b`、`smollm2:360m`）。

---

## 快速启动

在项目根目录下直接运行脚本（使用 `uv` 自动隔离环境并按需加载 PyQt6）：

```bash
./desktop-pet/run.sh
```

或者手动使用 `uv` 运行：

```bash
cd desktop-pet
uv run --with pyqt6 python3 desktop_pet.py
```

---

## 操作指南

- **鼠标左键按住拖拽**：将精灵自由移动到屏幕的任意角落，松开后自动记忆停靠坐标；
- **鼠标双击精灵**：主动让它根据你当前的软件状态给出深度思考与吐槽；
- **交互漫画气泡**：气泡底部带有输入框，输入文字按回车即可直接与它实时对话；
- **鼠标右键呼出菜单**：
  - 切换性格（曲别针 / 猫咪 / 黑客 / 禅宗）；
  - 投喂咖啡 ☕；
  - 强制小憩 / 唤醒；
  - 关闭气泡 / 退出程序。

---

## 进阶：直接在 Python 代码中 Serve 本地大模型

除了 Ollama，你可以**直接用 Python 代码本地加载并 Serve 模型**（支持 Apple Silicon GPU / MPS 加速、PyTorch、MLX 或纯 Python 模拟）：

### 方式 1：一键启动内置 Python Model Server

我们提供了开箱即用的 [`serve_model.py`](file:///Users/guofei/workspace/the-math-behind-llm/desktop-pet/serve_model.py)，它提供标准 OpenAI 兼容的 `/v1/chat/completions` 接口（端口 `8765`）：

```bash
# 1. 快速测试模式（无需下载权重，0 依赖，响应 0ms）：
python3 desktop-pet/serve_model.py --engine mock

# 2. 真实神经模型模式（使用 PyTorch 在 Mac GPU/MPS 上加载 Qwen 0.5B 或你微调的 checkpoint）：
uv run --with "transformers>=4.40.0" --with "torch" python3 desktop-pet/serve_model.py \
    --engine transformers \
    --model Qwen/Qwen2.5-0.5B-Instruct

# 3. Apple MLX 极速模式（Mac 原生 4-bit 量化加速）：
uv run --with "mlx-lm" python3 desktop-pet/serve_model.py \
    --engine mlx \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

启动后，桌面精灵在检测到 `http://localhost:8765` 在线时会**全自动优先使用你的 Python 模型服务端**，无需任何额外配置！

---

### 方式 2：在 Python 代码里直接 Import 模型进行推理

如果你希望把大模型逻辑完全内嵌在自己的 Python 脚本或后台任务中：

```python
from serve_model import LocalNeuralModel

# 1. 加载模型（支持 mock / transformers / mlx）
bot = LocalNeuralModel(engine="mock") 
# 或者加载真实 PyTorch 模型：
# bot = LocalNeuralModel(engine="transformers", model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")

# 2. 直接在 Python 中执行对话生成
messages = [
    {"role": "system", "content": "你是一只贴心的桌面宠物，用简短可爱的语气回答。"},
    {"role": "user", "content": "主人在敲代码时遇到了段错误，怎么安慰？"}
]

reply = bot.generate(messages, max_tokens=50, temperature=0.7)
print("精灵回复:", reply)
```

---

### 方式 3：接入外部 Ollama / LM Studio

如果你更习惯使用现有工具链：

```bash
# 启动 Ollama 本地超轻量模型 (占用显存/内存仅 300MB)
ollama run qwen2.5:0.5b

# 启动桌面精灵并指定本地大模型端点
export PET_LLM_URL="http://localhost:11434/v1/chat/completions"
export PET_LLM_MODEL="qwen2.5:0.5b"
./desktop-pet/run.sh
```

---

## 窗口置顶机制（macOS 特别优化）

为了确保精灵在切换全屏应用或跨桌面 Space 时永不消失，本程序通过 `ctypes` 调用了 macOS Cocoa 原生 API：
- **`setLevel: 1000`**：提升至 `kCGOverlayWindowLevel`，始终悬浮在所有窗口之上；
- **`canJoinAllSpaces` + `fullScreenAuxiliary`**：无论你在触控板如何左右横滑桌面（Spaces），或者全屏看代码，精灵都会如影随形跟随在屏幕角落。

