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

## 进阶：接入本地 0.5B 自训大模型

如果你本地运行了 Ollama，只需要在终端中设置环境变量后启动：

```bash
# 启动 Ollama 本地超轻量模型 (占用显存/内存仅 300MB)
ollama run qwen2.5:0.5b

# 启动桌面精灵并指定本地大模型端点
export PET_LLM_URL="http://localhost:11434/v1/chat/completions"
export PET_LLM_MODEL="qwen2.5:0.5b"
./desktop-pet/run.sh
```

一旦接入，桌面精灵的所有气泡台词都将由本地 0.5B 神经模型根据你的实时上下文实时生成！
