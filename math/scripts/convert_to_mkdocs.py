#!/usr/bin/env python3
"""
convert_to_mkdocs.py

Converts 'The Math Behind Large Language Models' markdown sources into
an interactive, fully-functional MkDocs Material documentation portal
located in the math/ folder.

Handles:
1. Directory discovery & module mapping (Module 0 through Module 8 + Labs)
2. Stripping legacy manual navigation bars (<nav aria-label="...">)
3. Converting raw HTML step headings to standard Markdown headings with ID anchors
4. Transforming GitHub-style callouts (> [!INTUITION] etc.) into MkDocs Material admonitions
5. Sanitizing LaTeX math formulas for MathJax compatibility
6. Rewriting internal relative chapter links to point to .md destinations
7. Generating bilingual index.md and curriculum.md pages
"""

import os
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MATH_DIR = REPO_ROOT / "math"
DOCS_DIR = MATH_DIR / "docs"

CHAPTER_MAP = [
    ("00-next-word-prediction", "module-0/00-next-word-prediction"),
    ("01-vectors-and-spaces", "module-1/01-vectors-and-spaces"),
    ("02-dot-product-and-similarity", "module-1/02-dot-product-and-similarity"),
    ("03-matrix-multiplication", "module-2/03-matrix-multiplication"),
    ("04-activation-functions", "module-2/04-activation-functions"),
    ("04b-lab-micro-brain", "module-2/04b-lab-micro-brain"),
    ("05-transformer-architecture", "module-3/05-transformer-architecture"),
    ("06-queries-keys-values", "module-3/06-queries-keys-values"),
    ("07-softmax-function", "module-3/07-softmax-function"),
    ("08-attention-formula", "module-3/08-attention-formula"),
    ("09-causal-masking", "module-4/09-causal-masking"),
    ("10-positional-encodings", "module-4/10-positional-encodings"),
    ("11-multi-head-attention", "module-4/11-multi-head-attention"),
    ("12-residual-connections", "module-5/12-residual-connections"),
    ("13-layer-norm-and-rmsnorm", "module-5/13-layer-norm-and-rmsnorm"),
    ("14-feed-forward-networks", "module-5/14-feed-forward-networks"),
    ("15-cross-entropy-loss", "module-6/15-cross-entropy-loss"),
    ("16-gradients-and-backpropagation", "module-6/16-gradients-and-backpropagation"),
    ("17-adam-optimizer", "module-6/17-adam-optimizer"),
    ("18-sampling-and-temperature", "module-7/18-sampling-and-temperature"),
    ("19-rlhf-and-dpo", "module-8/19-rlhf-and-dpo"),
]

ADMONITION_MAP_EN = {
    "INTUITION": ("note", "3-Year-Old Intuition"),
    "BRIDGING": ("question", "The Bridging Question"),
    "MATH": ("example", "Exact Math & Formulas"),
    "ORIGIN": ("quote", "Historical Origin & Derivation"),
    "EXAMPLE": ("tip", "Toy Example Walkthrough"),
    "TIP": ("tip", "Key Insight"),
    "NOTE": ("note", "Note"),
    "WARNING": ("warning", "Warning"),
    "TAKEAWAY": ("success", "Core Takeaway"),
}

ADMONITION_MAP_ZH = {
    "INTUITION": ("note", "3岁小孩的直觉"),
    "BRIDGING": ("question", "计算连接问题"),
    "MATH": ("example", "数学公式与符号剖析"),
    "ORIGIN": ("quote", "历史渊源与设计必然"),
    "EXAMPLE": ("tip", "超简单玩具算例"),
    "TIP": ("tip", "核心要点"),
    "NOTE": ("note", "注解"),
    "WARNING": ("warning", "警告"),
    "TAKEAWAY": ("success", "核心一句话要点"),
}


def strip_manual_navs(md_text: str) -> str:
    """Strip legacy manual Table of Contents and Chapter Navigation."""
    md_text = re.sub(
        r'<nav\s+aria-label=[\'"][^\'"]*[\'"]>[\s\S]*?</nav>\s*(?:<hr\s*/?>|---)?',
        '',
        md_text,
        flags=re.IGNORECASE
    )
    # Remove leading <hr> or --- if directly beneath title
    md_text = re.sub(r'^(#[^\n]+\n+)(?:<hr\s*/?>|---)\s*\n+', r'\1\n', md_text)
    return md_text


def adapt_headings(md_text: str, lang: str = 'en') -> str:
    """
    Converts raw HTML <h2 id="step-N"> headings into standard Markdown headings
    with attribute list anchor ids (e.g. ## Step N: ... {: #step-N }), and ensures
    any existing markdown ## Step N headings also receive the exact anchor ID.
    """
    def repl_h2(m):
        elem_id = m.group(1) or ""
        inner = m.group(2).strip()
        inner = inner.replace('&amp;', '&')

        # Check for Step N (English)
        m_en = re.search(r'Step\s*(\d+)[:：]?\s*(.*)', inner, re.IGNORECASE)
        # Check for 第 N 步 (Chinese)
        m_zh = re.search(r'(?:第\s*(\d+)\s*步|步骤\s*(\d+))[:：]?\s*(.*)', inner, re.IGNORECASE)

        if m_en:
            num = m_en.group(1)
            title = m_en.group(2).strip()
            title = re.sub(r'<[^>]+>', '', title).strip()
            anchor = elem_id if elem_id else f"step-{num}"
            return f"## Step {num}: {title} {{: #{anchor} }}"
        elif m_zh:
            num = str(int(m_zh.group(1) or m_zh.group(2)))
            title = m_zh.group(3).strip()
            title = re.sub(r'<[^>]+>', '', title).strip()
            anchor = elem_id if elem_id else f"step-{num}"
            return f"## 第 {num} 步：{title} {{: #{anchor} }}"
        elif elem_id:
            clean_title = re.sub(r'<[^>]+>', '', inner).strip()
            return f"## {clean_title} {{: #{elem_id} }}"
        else:
            return f"## {inner}"

    md_text = re.sub(r'<h2(?:\s+id="([^"]+)")?>(.*?)</h2>', repl_h2, md_text, flags=re.IGNORECASE)
    md_text = re.sub(r'<h3(?:\s+id="([^"]+)")?>(.*?)</h3>', lambda m: f"### {m.group(2).strip().replace('&amp;', '&')}" + (f" {{: #{m.group(1)} }}" if m.group(1) else ""), md_text, flags=re.IGNORECASE)
    md_text = re.sub(r'<h4(?:\s+id="([^"]+)")?>(.*?)</h4>', lambda m: f"#### {m.group(2).strip().replace('&amp;', '&')}", md_text, flags=re.IGNORECASE)

    # Clean legacy fallback anchors like <a id="step-1"></a>
    md_text = re.sub(r'<a\s+(?:id|name)=["\'](step-\d+)["\']>\s*</a>', '', md_text, flags=re.IGNORECASE)

    # Also handle markdown ## Step N or ## 第 N 步 if it doesn't already have {: #...}
    def repl_md_en(m):
        full = m.group(0)
        if '{:' in full:
            return full
        num = m.group(1)
        rest = m.group(2) or ""
        rest = rest.replace('&amp;', '&').strip()
        sep = ": " if rest and not rest.startswith((':', '：')) else ""
        return f"## Step {num}{sep}{rest} {{: #step-{num} }}"

    def repl_md_zh(m):
        full = m.group(0)
        if '{:' in full:
            return full
        num = str(int(m.group(1) or m.group(2)))
        rest = m.group(3) or ""
        rest = rest.replace('&amp;', '&').strip()
        sep = "：" if rest and not rest.startswith((':', '：')) else ""
        return f"## 第 {num} 步{sep}{rest} {{: #step-{num} }}"

    md_text = re.sub(r'^##\s+Step\s*(\d+)[:：]?\s*(.*)$', repl_md_en, md_text, flags=re.MULTILINE | re.IGNORECASE)
    md_text = re.sub(r'^##\s+(?:第\s*(\d+)\s*步|步骤\s*(\d+))[:：]?\s*(.*)$', repl_md_zh, md_text, flags=re.MULTILINE | re.IGNORECASE)

    return md_text


def convert_callouts(md_text: str, lang: str = 'en') -> str:
    """
    Transforms GitHub-style callout blocks (> [!INTUITION] etc.)
    into MkDocs Material admonition blocks (!!! note "Title").
    """
    ad_map = ADMONITION_MAP_ZH if lang == 'zh' else ADMONITION_MAP_EN
    lines = md_text.split('\n')
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = re.match(r'^>\s*\[!(INTUITION|BRIDGING|MATH|ORIGIN|EXAMPLE|TIP|NOTE|WARNING|TAKEAWAY)\]\s*(.*)$', line, re.IGNORECASE)
        if m:
            ctype = m.group(1).upper()
            custom_title = m.group(2).strip()
            ad_type, default_prefix = ad_map.get(ctype, ("note", ctype))
            if custom_title:
                title = f"{default_prefix}: {custom_title}"
            else:
                title = default_prefix

            block_lines = []
            i += 1
            while i < n and (lines[i].startswith('>') or lines[i].strip() == ''):
                if lines[i].strip() == '' and (i + 1 >= n or not lines[i+1].startswith('>')):
                    break
                l = lines[i]
                if l.startswith('> '):
                    block_lines.append(l[2:])
                elif l.startswith('>'):
                    block_lines.append(l[1:])
                else:
                    block_lines.append(l)
                i += 1

            # Format as MkDocs admonition with 4 spaces indent
            admonition = [f'!!! {ad_type} "{title}"']
            for bline in block_lines:
                if bline.strip():
                    admonition.append(f"    {bline}")
                else:
                    admonition.append("")
            out.append('\n'.join(admonition))
        else:
            out.append(line)
            i += 1

    return '\n'.join(out)


def process_math(md_text: str) -> str:
    """
    Safely normalizes and sanitizes LaTeX math equations for MkDocs Material & KaTeX.
    1. Protects code blocks (``` and `) and <pre> tags so their contents are never touched.
    2. Converts single-line display math (^[ \\t]*\\$\\$(.+?)\\$\\$\\s*$) into isolated display blocks with preserved indentation.
    3. Normalizes existing display math blocks ($$\\n...\\n$$) so they are surrounded by blank lines and properly indented.
    4. Sanitizes inequality symbols inside LaTeX formulas (e.g. w_{<t} -> w_{\\lt t}) strictly inside isolated math contents.
    5. Restores protected code blocks.
    """
    code_store = {}
    def save_code(m):
        token = f"XXPROTECTEDCODE{len(code_store)}XX"
        code_store[token] = m.group(0)
        return token

    # Protect code and preformatted blocks
    md_text = re.sub(r'```[\s\S]*?```', save_code, md_text)
    md_text = re.sub(r'`[^`\n]+`', save_code, md_text)
    md_text = re.sub(r'<pre[\s\S]*?</pre>', save_code, md_text)

    # Sanitize helper for math expressions only
    def sanitize_latex(s: str) -> str:
        return re.sub(r'<([a-zA-Z0-9])', r'\\lt \1', s)

    # Convert single-line display math to multi-line blocks with preserved indentation
    def normalize_single_line_display(m):
        indent = m.group(1)
        math = m.group(2).strip()
        return f"\n\n{indent}$$\n{indent}{math}\n{indent}$$\n\n"

    md_text = re.sub(r'^([ \t]*)\$\$([^\$\n]+?)\$\$\s*$', normalize_single_line_display, md_text, flags=re.MULTILINE)

    # Normalize multi-line display math: preserve indentation and sanitize formula
    def repl_display_math(m):
        indent = m.group(1)
        content = m.group(2)
        sanitized = sanitize_latex(content)
        lines = sanitized.strip().split('\n')
        indented_lines = '\n'.join(f"{indent}{l.strip()}" for l in lines)
        return f"\n\n{indent}$$\n{indented_lines}\n{indent}$$\n\n"

    md_text = re.sub(r'^([ \t]*)\$\$\s*\n([\s\S]*?)\n[ \t]*\$\$', repl_display_math, md_text, flags=re.MULTILINE)

    # Sanitize inline math strictly inside single $ delimiters
    def repl_inline_math(m):
        content = m.group(1)
        sanitized = sanitize_latex(content)
        return f"${sanitized}$"

    md_text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', repl_inline_math, md_text)

    # Restore protected code and pre blocks
    for token, orig in code_store.items():
        md_text = md_text.replace(token, orig)

    return md_text


def adapt_links(md_text: str, current_module: str, is_zh: bool = False) -> str:
    """Rewrites relative links between chapters to match mkdocs directory layout."""
    suffix = ".zh.md" if is_zh else ".md"
    html_suffix = "index.zh.html" if is_zh else "index.html"

    # Map each chapter folder to its module path
    folder_to_mod = {}
    for ch_dir, mod_path in CHAPTER_MAP:
        folder_to_mod[ch_dir] = mod_path

    # Replace ../NN-topic/index.html or NN-topic/index.html
    for ch_dir, mod_path in folder_to_mod.items():
        # Relative from another module: ../module-X/filename.md
        target_file = f"../{mod_path}{suffix}"
        pattern_up = rf'(?:\.\./)?{re.escape(ch_dir)}/(?:index\.zh\.html|index\.html)'
        md_text = re.sub(pattern_up, target_file, md_text)

    # In labs or exercises:
    md_text = re.sub(
        r'(?:labs/|04b-lab-micro-brain/)01_micro_brain_exercise\.py',
        r'../labs/01_micro_brain_exercise.py',
        md_text
    )
    md_text = re.sub(
        r'(?:labs/|04b-lab-micro-brain/)01_micro_brain\.py',
        r'../labs/01_micro_brain.py',
        md_text
    )
    md_text = re.sub(
        r'micro_brain_exercise\.py',
        r'../labs/01_micro_brain_exercise.py',
        md_text
    )
    md_text = re.sub(
        r'micro_brain\.py',
        r'../labs/01_micro_brain.py',
        md_text
    )

    return md_text


def process_chapter_file(src_path: Path, dst_path: Path, current_module: str, lang: str):
    """Processes a single chapter markdown file and writes it to destination."""
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()

    is_zh = (lang == 'zh')
    content = strip_manual_navs(content)
    content = adapt_headings(content, lang=lang)
    content = convert_callouts(content, lang=lang)
    content = process_math(content)
    content = adapt_links(content, current_module=current_module, is_zh=is_zh)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_curriculum_pages():
    """Generates curriculum.md and curriculum.zh.md from root curriculum.md."""
    src_curr = REPO_ROOT / "curriculum.md"
    if not src_curr.exists():
        return

    with open(src_curr, 'r', encoding='utf-8') as f:
        raw_en = f.read()

    # Adapt English curriculum
    curr_en = strip_manual_navs(raw_en)
    curr_en = convert_callouts(curr_en, lang='en')
    curr_en = process_math(curr_en)
    # Replace relative links in curriculum:
    for ch_dir, mod_path in CHAPTER_MAP:
        curr_en = re.sub(rf'{re.escape(ch_dir)}/index\.html', f'{mod_path}.md', curr_en)
    curr_en = re.sub(r'08b-lab-attention-brain/index\.html', '#', curr_en)
    curr_en = re.sub(r'13b-lab-transformer-brain/index\.html', '#', curr_en)
    curr_en = re.sub(r'17b-lab-complete-llm/index\.html', '#', curr_en)

    with open(DOCS_DIR / "curriculum.md", 'w', encoding='utf-8') as f:
        f.write(curr_en)

    # Generate Chinese curriculum page
    curr_zh = strip_manual_navs(raw_en)
    curr_zh = convert_callouts(curr_zh, lang='zh')
    curr_zh = process_math(curr_zh)
    for ch_dir, mod_path in CHAPTER_MAP:
        curr_zh = re.sub(rf'{re.escape(ch_dir)}/index\.html', f'{mod_path}.zh.md', curr_zh)
    curr_zh = re.sub(r'08b-lab-attention-brain/index\.html', '#', curr_zh)
    curr_zh = re.sub(r'13b-lab-transformer-brain/index\.html', '#', curr_zh)
    curr_zh = re.sub(r'17b-lab-complete-llm/index\.html', '#', curr_zh)

    with open(DOCS_DIR / "curriculum.zh.md", 'w', encoding='utf-8') as f:
        f.write(curr_zh)


def generate_index_pages():
    """Generates clean, welcoming portal homepages index.md and index.zh.md."""
    index_en = r"""# The Math Behind Large Language Models

<p><em>An intuitive, rigorous, and comprehensive guide to understanding the mathematical foundations of Large Language Models.</em></p>

---

## 1. Core Teaching Philosophy: The 5-Step Learning Ladder

Every single chapter in this course follows an unshakeable 5-step learning ladder, bridging physical intuition with mathematical rigor:

| Step | Section Title | What You Learn | Tactile Metaphor |
| :---: | :--- | :--- | :--- |
| **Step 1** | **3-Year-Old Intuition** | Physical intuition with zero mathematical jargon | Mystery boxes, playground maps, spotlight beams |
| **Step 2** | **The Bridging Question** | Translating physical concepts into computer matrices | Connecting physical games to linear arrays |
| **Step 3** | **Exact Math & Formulas** | Genuine equations used in modern LLMs (LLaMA, Transformers) | Explicit Greek symbols ($\sum, \prod, \exp, \nabla$), shapes & dims |
| **Step 4** | **Where Did It Come From?** | Historical origins, design rationale & failed attempts | Boltzmann (1868), Shannon (1948), Vaswani (2017) |
| **Step 5** | **Concrete Toy Example** | Step-by-step arithmetic with tiny numbers (dim 2 or 3) | Hand-calculated operations verifying every math step |
| **Step 6** | **Core Takeaway** | Punchy 1–2 sentence summary of architectural role | Mental anchor for the overall model brain |

---

## 2. Complete Curriculum Roadmap

The course is structured into **9 progressive modules** spanning from next-word probability to post-training alignment:

```
[00: Next-Word Prediction] ──► [01: Vectors & Embeddings] ──► [02: Dot Product & Similarity]
                                                                        │
                                                                        ▼
[05: Transformer Blueprint] ◄── [Lab 01: Micro-Brain] ◄── [04: Activations] ◄── [03: Matrix Mult]
         │
         ▼
[06: Queries, Keys, Values] ──► [07: Softmax] ──► [08: Attention Formula] ──► [09: Causal Masking]
                                                                                      │
                                                                                      ▼
[12: Residual Connections] ◄── [11: Multi-Head Attention] ◄── [10: Positional Encodings & RoPE]
         │
         ▼
[13: RMSNorm] ──► [14: Feed-Forward Blocks] ──► [15: Cross-Entropy Loss]
                                                        │
                                                        ▼
[18: Sampling (Temp/Top-p)] ◄── [17: Adam Optimizer] ◄── [16: Backpropagation]
         │
         ▼
[19: Alignment (RLHF & DPO)]
```

### Module Quick Navigation

- **Module 0: The Big Picture**
  - [Chapter 00: The Next-Word Guessing Game](module-0/00-next-word-prediction.md)
- **Module 1: Vector Geometry & Embeddings**
  - [Chapter 01: The Word Map (Vectors & Embeddings)](module-1/01-vectors-and-spaces.md)
  - [Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)](module-1/02-dot-product-and-similarity.md)
- **Module 2: Transforming Spaces & Activations**
  - [Chapter 03: The Magic Stretching Box (Matrix Multiplication)](module-2/03-matrix-multiplication.md)
  - [Chapter 04: The One-Way Gate (ReLU, GELU, SwiGLU)](module-2/04-activation-functions.md)
  - [Hands-on Lab 01: The Micro-Brain in 80 Lines of Python](labs/01-micro-brain.md)
- **Module 3: The Attention Mechanism**
  - [Chapter 05: The Transformer Blueprint & 70B Parameter Census](module-3/05-transformer-architecture.md)
  - [Chapter 06: The Library Clue Hunt (Queries, Keys, Values)](module-3/06-queries-keys-values.md)
  - [Chapter 07: The Fair Voting Booth (The Softmax Function)](module-3/07-softmax-function.md)
  - [Chapter 08: The Attention Formula & Division by $\\sqrt{d_k}$](module-3/08-attention-formula.md)
- **Module 4: Positional Encodings & Multi-Head Attention**
  - [Chapter 09: Blindfolds on Future Words (Causal Masking)](module-4/09-causal-masking.md)
  - [Chapter 10: Where in the Sentence Am I? (RoPE)](module-4/10-positional-encodings.md)
  - [Chapter 11: Looking Through Different Glasses (Multi-Head Attention)](module-4/11-multi-head-attention.md)
- **Module 5: Residuals, Normalization & Feed-Forward**
  - [Chapter 12: The Shortcut Bridge (Residual Connections)](module-5/12-residual-connections.md)
  - [Chapter 13: Keeping Everyone Calm (LayerNorm & RMSNorm)](module-5/13-layer-norm-and-rmsnorm.md)
  - [Chapter 14: The Thinking Chamber (Feed-Forward Networks)](module-5/14-feed-forward-networks.md)
- **Module 6: Training, Loss & Optimization**
  - [Chapter 15: How Wrong Was I? (Cross-Entropy Loss & Perplexity)](module-6/15-cross-entropy-loss.md)
  - [Chapter 16: Walking Down the Mountain (Gradients & Backpropagation)](module-6/16-gradients-and-backpropagation.md)
  - [Chapter 17: The Smart Walker (Momentum & AdamW)](module-6/17-adam-optimizer.md)
- **Module 7: Inference & Sampling**
  - [Chapter 18: Turning Up the Heat (Temperature, Top-k, Top-p)](module-7/18-sampling-and-temperature.md)
- **Module 8: Alignment & Post-Training**
  - [Chapter 19: Teaching Good Manners (RLHF & DPO)](module-8/19-rlhf-and-dpo.md)
"""

    index_zh = """# 大模型背后的数学原理与公式推导

<p><em>左手极简物理直觉，右手严谨数学公式：带你真正读懂大语言模型（LLM）的底层数学机制与演化脉络。</em></p>

---

## 1. 核心教学法：五步递进学习梯（The 5-Step Pedagogy）

本课程的每一个章节，都严格按照不跳步、无预设假设的阶梯式教学逻辑展开：

| 教学步骤 | 章节板块 | 核心攻坚目标 | 实体生活隐喻 |
| :---: | :--- | :--- | :--- |
| **第 1 步** | **3岁小孩的直觉** | 彻底剥离数学术语，用生活实体建立物理模型 | 神秘盒子、游乐场地图、聚光灯手电筒、橡皮泥压模机 |
| **第 2 步** | **计算连接问题** | 明确计算机如何将物理直觉转化为数字张量 | 从游戏规则到可存储、可计算的浮点数组 |
| **第 3 步** | **数学公式与符号剖析** | 当代大模型主流公式（LLaMA/Transformer） | 彻底拆解所有希腊字母（$\\sum, \\prod, \\exp, \\nabla$）、下标与张量形状 |
| **第 4 步** | **历史渊源与设计必然** | 追溯公式起源，解答“为什么非得写成这样” | 玻尔兹曼（1868）、香农（1948）、瓦斯瓦尼（2017） |
| **第 5 步** | **超简单玩具算例** | 极小数字（2维/3维向量）手算推演全流程 | 读者可用笔在纸上逐项验算每一步乘法与加法 |
| **第 6 步** | **核心一句话要点** | 1-2 句点睛之笔，凝练其在大模型大脑中的定位 | 形成不可动摇的长期记忆认知锚点 |

---

## 2. 全课程九大模块演进全景图

从词语预测概率，到注意力机制，再到现代参数对齐，全景数据流管道如下：

```
[00: 下一个词预测] ──► [01: 向量与嵌入空间] ──► [02: 点积与相似度]
                                                      │
                                                      ▼
[05: Transformer宏观蓝图] ◄── [实战01: 微脑工坊] ◄── [04: 激活函数] ◄── [03: 矩阵乘法]
         │
         ▼
[06: 查询、键与值 (QKV)] ──► [07: Softmax投票] ──► [08: 注意力公式] ──► [09: 因果掩码]
                                                                                │
                                                                                ▼
[12: 残差连接立交桥] ◄── [11: 多头注意力] ◄── [10: 位置编码与RoPE]
         │
         ▼
[13: RMSNorm归一化] ──► [14: 前馈网络密室] ──► [15: 交叉熵损失函数]
                                                      │
                                                      ▼
[18: 采样策略与温度] ◄── [17: AdamW优化器] ◄── [16: 梯度与反向传播]
         │
         ▼
[19: 模型对齐 (RLHF与DPO)]
```

### 模块极速直达目录

- **模块 0：认知起点与宏观全景**
  - [第 00 章：猜词游戏（自回归下一个词预测）](module-0/00-next-word-prediction.zh.md)
- **模块 1：向量几何与嵌入空间**
  - [第 01 章：单词地图（向量与嵌入空间）](module-1/01-vectors-and-spaces.zh.md)
  - [第 02 章：丈量亲疏（点积与余弦相似度）](module-1/02-dot-product-and-similarity.zh.md)
- **模块 2：空间变换与激活函数**
  - [第 03 章：空间变形盒（矩阵乘法）](module-2/03-matrix-multiplication.zh.md)
  - [第 04 章：单向阀门（激活函数：ReLU, GELU, SwiGLU）](module-2/04-activation-functions.zh.md)
  - [实战工坊 01：80行纯Python手搓微脑 (Bengio 2003)](labs/01-micro-brain.zh.md)
- **模块 3：核心注意力机制**
  - [第 05 章：全景蓝图（Transformer 宏观架构与 70B 参数普查）](module-3/05-transformer-architecture.zh.md)
  - [第 06 章：图书寻踪（查询、键与值：Q, K, V）](module-3/06-queries-keys-values.zh.md)
  - [第 07 章：公平投票箱（Softmax 函数）](module-3/07-softmax-function.zh.md)
  - [第 08 章：注意力公式与为何除以根号dk](module-3/08-attention-formula.zh.md)
- **模块 4：序列顺序与多头注意力**
  - [第 09 章：戴上眼罩（因果掩码）](module-4/09-causal-masking.zh.md)
  - [第 10 章：我在句中何处（位置编码与旋转位置编码 RoPE）](module-4/10-positional-encodings.zh.md)
  - [第 11 章：换副眼镜看世界（多头注意力机制）](module-4/11-multi-head-attention.zh.md)
- **模块 5：残差连接、归一化与前馈网络**
  - [第 12 章：高速立交桥（残差连接）](module-5/12-residual-connections.zh.md)
  - [第 13 章：让大家冷静下来（层归一化与 RMSNorm）](module-5/13-layer-norm-and-rmsnorm.zh.md)
  - [第 14 章：沉思密室（前馈神经网络）](module-5/14-feed-forward-networks.zh.md)
- **模块 6：模型训练、损失度量与参数优化**
  - [第 15 章：我错得有多离谱（交叉熵损失与困惑度）](module-6/15-cross-entropy-loss.zh.md)
  - [第 16 章：摸黑下山（梯度与反向传播）](module-6/16-gradients-and-backpropagation.zh.md)
  - [第 17 章：聪明的向导（动量与 AdamW 优化器）](module-6/17-adam-optimizer.zh.md)
- **模块 7：推理生成与采样策略**
  - [第 18 章：调高温度（温度、Top-k 与 Top-p 采样）](module-7/18-sampling-and-temperature.zh.md)
- **模块 8：模型对齐与人类偏好**
  - [第 19 章：立规矩学礼貌（RLHF、奖励模型与直接偏好优化 DPO）](module-8/19-rlhf-and-dpo.zh.md)
"""

    with open(DOCS_DIR / "index.md", 'w', encoding='utf-8') as f:
        f.write(index_en)

    with open(DOCS_DIR / "index.zh.md", 'w', encoding='utf-8') as f:
        f.write(index_zh)


def main():
    print("Converting 'The Math Behind Large Language Models' to MkDocs Material portal...")

    # Process all chapter files
    for ch_dir, mod_rel in CHAPTER_MAP:
        src_dir = REPO_ROOT / ch_dir
        if not src_dir.exists():
            print(f"  [WARN] Source directory {src_dir} not found, skipping.")
            continue

        module_name = mod_rel.split('/')[0]

        # English chapter
        src_en = src_dir / "index.md"
        if src_en.exists():
            dst_en = DOCS_DIR / f"{mod_rel}.md"
            process_chapter_file(src_en, dst_en, current_module=module_name, lang='en')
            print(f"  [EN] {src_en.relative_to(REPO_ROOT)} -> {dst_en.relative_to(MATH_DIR)}")

        # Chinese chapter
        src_zh = src_dir / "index.zh.md"
        if src_zh.exists():
            dst_zh = DOCS_DIR / f"{mod_rel}.zh.md"
            process_chapter_file(src_zh, dst_zh, current_module=module_name, lang='zh')
            print(f"  [ZH] {src_zh.relative_to(REPO_ROOT)} -> {dst_zh.relative_to(MATH_DIR)}")

    # Special handling for Lab 01 in labs/
    lab_src = REPO_ROOT / "04b-lab-micro-brain"
    if (lab_src / "index.md").exists():
        process_chapter_file(lab_src / "index.md", DOCS_DIR / "labs/01-micro-brain.md", current_module="labs", lang='en')
        print(f"  [LAB-EN] 04b-lab-micro-brain/index.md -> docs/labs/01-micro-brain.md")
    if (lab_src / "index.zh.md").exists():
        process_chapter_file(lab_src / "index.zh.md", DOCS_DIR / "labs/01-micro-brain.zh.md", current_module="labs", lang='zh')
        print(f"  [LAB-ZH] 04b-lab-micro-brain/index.zh.md -> docs/labs/01-micro-brain.zh.md")

    # Generate curriculum and index pages
    generate_curriculum_pages()
    print("  Generated curriculum.md and curriculum.zh.md")

    generate_index_pages()
    print("  Generated index.md and index.zh.md")

    print("\nConversion complete! All docs created in math/docs/")


if __name__ == "__main__":
    main()
