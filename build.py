#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "markdown",
# ]
# ///
"""
Static Site Generator for 'The Math Behind Large Language Models'
Translates Markdown (.md, .zh.md) into high-density, pure semantic HTML with KaTeX.
Zero client-side markdown compilation. Bulletproof static rendering.
"""

import os
import sys
import re
import glob
import shutil
import subprocess
import urllib.parse
from html.parser import HTMLParser

# Self-bootstrap if markdown is missing and uv is available
try:
    import markdown
except ImportError:
    if shutil.which("uv"):
        subprocess.run(["uv", "run", "--with", "markdown", "python3", __file__] + sys.argv[1:])
        sys.exit(0)
    else:
        sys.exit("Error: 'markdown' library is required. Run with: uv run build.py (or pip install markdown)")

CALLOUT_TITLES = {
    'en': {
        'INTUITION': '3-Year-Old Intuition',
        'MATH': 'The Exact Math & Formula',
        'ORIGIN': 'Where Did It Come From?',
        'EXAMPLE': 'Concrete Toy Example',
        'TIP': 'Core Takeaway',
        'NOTE': 'Key Note',
        'WARNING': 'Caution'
    },
    'zh': {
        'INTUITION': '3 岁小孩直觉',
        'MATH': '严谨数学公式与推导',
        'ORIGIN': '历史渊源与技术演进',
        'EXAMPLE': '手算极简数值示例',
        'TIP': '核心精髓总结',
        'NOTE': '重要注解',
        'WARNING': '避坑警示'
    }
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{LANG_CODE}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{PAGE_TITLE} - The Math Behind LLMs</title>
  <!-- KaTeX CSS for equation rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <!-- KaTeX JS -->
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <!-- KaTeX Auto-Render Extension -->
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <style>
    /* Compact High-Density Minimal Style (~38 lines) */
    body {{
      max-width: 800px;
      margin: 0 auto;
      padding: 1rem 0.75rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 13.5px;
      line-height: 1.48;
      color: #222222;
      background-color: #f6f6ef;
    }}
    h1, h2, h3, h4 {{
      color: #000000;
      margin: 1.2rem 0 0.35rem;
      line-height: 1.25;
    }}
    h1 {{ font-size: 1.35rem; }}
    h2 {{ font-size: 1.15rem; border-bottom: 1px solid #dcdcd4; padding-bottom: 2px; }}
    h3 {{ font-size: 1rem; }}
    p, ul, ol {{ margin: 0.45rem 0; }}
    hr {{ border: 0; border-top: 1px solid #dcdcd4; margin: 0.85rem 0; }}
    a {{ color: #000000; text-decoration: underline; }}
    a:hover {{ color: #ff6600; }}
    fieldset {{
      background-color: #ffffff;
      border: 1px solid #dcdcd4;
      padding: 0.5rem 0.85rem;
      margin: 0.75rem 0;
    }}
    legend {{ font-weight: bold; color: #222; padding: 0 4px; }}
    blockquote {{
      border-left: 2px solid #828282;
      margin: 0.6rem 0;
      padding: 0.2rem 0 0.2rem 0.75rem;
      color: #666;
      background: #ffffff;
    }}
    details {{
      border: 1px solid #dcdcd4;
      background: #ffffff;
      padding: 0.4rem 0.75rem;
      margin: 0.5rem 0;
    }}
    pre, code, kbd, samp {{
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      background: #ffffff;
      border: 1px solid #dcdcd4;
      padding: 0.6rem;
      overflow-x: auto;
      line-height: 1.35;
    }}
    kbd {{
      background: #eeeeee;
      border: 1px solid #cccccc;
      padding: 1px 4px;
      border-radius: 2px;
    }}
    samp {{ color: #828282; }}
    mark {{ background: #ffffbb; padding: 1px 3px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.75rem 0;
      font-size: 12.5px;
    }}
    th, td {{ border: 1px solid #dcdcd4; padding: 4px 8px; }}
    th {{ background-color: #eae9e1; text-align: left; }}
    caption {{ caption-side: top; text-align: left; font-weight: bold; margin-bottom: 0.3rem; }}
    .katex-display {{ overflow-x: auto; overflow-y: hidden; margin: 0.5em 0; padding: 4px 0; }}
  </style>
</head>
<body>
  <header>
    <h1><a href="{ROOT_PREFIX}index.html">The Math Behind Large Language Models</a></h1>
    <p><em>Intuitive, Rigorous, Pure Semantic HTML with KaTeX Math</em></p>
    <hr>
  </header>

{NAV_BAR}

  <main id="content">
{MAIN_CONTENT}
  </main>

  <footer>
    <hr>
    <p><strong>The Math Behind LLMs</strong> • Pure Semantic HTML • KaTeX Math • Static Build.</p>
    <p><small>Grounded in physical intuition and mathematical rigor.</small></p>
  </footer>

  <script>
    // Language query-param fallback redirection if visiting index.html?lang=zh
    const urlParams = new URLSearchParams(window.location.search);
    const langParam = urlParams.get('lang');
    if (langParam === 'zh' && !window.location.pathname.endsWith('index.zh.html')) {{
      window.location.replace('index.zh.html');
    }} else if (langParam === 'en' && window.location.pathname.endsWith('index.zh.html')) {{
      window.location.replace('index.html');
    }}

    // KaTeX equation rendering pass with DOMContentLoaded safety and complete delimiters
    function renderAllMath() {{
      if (typeof renderMathInElement !== 'undefined') {{
        renderMathInElement(document.body, {{
          delimiters: [
            {{ left: '$$', right: '$$', display: true }},
            {{ left: '$', right: '$', display: false }},
            {{ left: '\\\\(', right: '\\\\)', display: false }},
            {{ left: '\\\\[', right: '\\\\]', display: true }}
          ],
          throwOnError: false
        }});
      }}
    }}
    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', renderAllMath);
    }} else {{
      renderAllMath();
    }}
  </script>
</body>
</html>
"""

def convert_callouts(text, lang='en'):
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r'^>\s*\[!(INTUITION|MATH|ORIGIN|EXAMPLE|TIP|NOTE|WARNING)\]\s*(.*)$', line, re.IGNORECASE)
        if m:
            ctype = m.group(1).upper()
            custom_title = m.group(2).strip()
            title = custom_title or CALLOUT_TITLES.get(lang, {}).get(ctype, ctype)
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
            inner_text = '\n'.join(block_lines).strip()
            out.append(f'<fieldset markdown="1"><legend><strong>{title}</strong></legend>\n\n{inner_text}\n\n</fieldset>')
        else:
            out.append(line)
            i += 1
    return '\n'.join(out)

def strip_manual_navs(md_text):
    """Strip legacy manual Table of Contents and Chapter Navigation from markdown."""
    # Remove manual Table of Contents navigation and any trailing hr
    md_text = re.sub(
        r'<nav\s+aria-label=[\'"](?:Table of Contents|目录导航|目录)[\'"]>[\s\S]*?</nav>\s*(?:<hr\s*/?>|---)?',
        '',
        md_text,
        flags=re.IGNORECASE
    )
    # Remove manual Chapter Navigation and any preceding hr
    md_text = re.sub(
        r'(?:<hr\s*/?>|---)?\s*<nav\s+aria-label=[\'"](?:Chapter Navigation|章节导航)[\'"]>[\s\S]*?</nav>',
        '',
        md_text,
        flags=re.IGNORECASE
    )
    return md_text

def process_step_headings_and_toc(html_content, lang='en'):
    """
    Scans for Step 1..6 headings (both ## markdown-rendered h2 and raw <h2 id=...>),
    ensures each has an id='step-N', and generates an in-page Table of Contents
    inserted directly beneath the chapter's <h1> heading.
    """
    steps = []

    def repl_h2(m):
        existing_id = m.group(1) or ''
        inner = m.group(2)

        # Match Step N (English) or 第 N 步 / 步骤 N (Chinese)
        m_en = re.search(r'Step\s*(\d+)[:：]?\s*(.*)', inner, re.IGNORECASE)
        m_zh = re.search(r'(?:第\s*(\d+)\s*步|步骤\s*(\d+))[:：]?\s*(.*)', inner, re.IGNORECASE)

        step_id = None
        label = None

        if m_en:
            num = m_en.group(1)
            raw_title = m_en.group(2)
            clean_title = re.sub(r'\s*[\(（][^\)）]+[\)）]', '', raw_title).strip()
            clean_title = re.sub(r'<[^>]+>', '', clean_title).strip()
            step_id = f"step-{num}"
            label = f"{num}. {clean_title}" if clean_title else f"Step {num}"
        elif m_zh:
            num = str(int(m_zh.group(1) or m_zh.group(2)))
            raw_title = m_zh.group(3)
            clean_title = re.sub(r'\s*[\(（][^\)）]+[\)）]', '', raw_title).strip()
            clean_title = re.sub(r'<[^>]+>', '', clean_title).strip()
            step_id = f"step-{num}"
            label = f"{num}. {clean_title}" if clean_title else f"第 {num} 步"
        elif existing_id.startswith('step-'):
            step_id = existing_id
            clean_title = re.sub(r'<[^>]+>', '', inner).strip()
            label = clean_title

        if step_id and label:
            if not any(sid == step_id for sid, _ in steps):
                steps.append((step_id, label))
            return f'<h2 id="{step_id}">{inner}</h2>'
        return m.group(0)

    # Process all <h2> tags
    new_html = re.sub(r'<h2(?:\s+id="([^"]+)")?>(.*?)</h2>', repl_h2, html_content, flags=re.IGNORECASE)

    # Fallback for older Step 1 anchors: <a id="step-1"></a>
    if '<a id="step-1"></a>' in new_html and not any(sid == 'step-1' for sid, _ in steps):
        label = "1. 3-Year-Old Intuition" if lang == 'en' else "1. 3 岁小孩直觉"
        steps.insert(0, ('step-1', label))

    # Sort steps by numerical step index if possible
    def step_key(item):
        m = re.search(r'step-(\d+)', item[0])
        return int(m.group(1)) if m else 999
    steps.sort(key=step_key)

    if steps:
        toc_header = "Table of Contents:" if lang == 'en' else "目录导航："
        links = [f'<a href="#{sid}">{slabel}</a>' for sid, slabel in steps]
        toc_block = f"""<nav aria-label="Table of Contents">
  <p>
    <strong>{toc_header}</strong> 
    {" &bull; \n    ".join(links)}
  </p>
</nav>
<hr>"""

        h1_match = re.search(r'(<h1[^>]*>.*?</h1>)', new_html, flags=re.IGNORECASE | re.DOTALL)
        if h1_match:
            pos = h1_match.end()
            new_html = new_html[:pos] + "\n\n" + toc_block + "\n\n" + new_html[pos:]
        else:
            new_html = toc_block + "\n\n" + new_html

    return new_html

def render_markdown_to_html(md_text, lang='en', is_chapter=False):
    if is_chapter:
        md_text = strip_manual_navs(md_text)

    # 1. Convert GitHub-style callouts to fieldsets (with markdown="1" for nested parsing)
    text = convert_callouts(md_text, lang)

    # Ensure all manual fieldsets also enable nested markdown parsing
    text = re.sub(r'<fieldset(?![^>]*markdown=)', '<fieldset markdown="1"', text)

    # 2. Protect code blocks from math parsing
    code_store = {}
    def save_code(match):
        key = f"XXCODEBLOCK{len(code_store)}XX"
        code_store[key] = match.group(0)
        return key

    text = re.sub(r'```[\s\S]*?```', save_code, text)
    text = re.sub(r'`[^`\n]+`', save_code, text)

    # Helper: sanitize LaTeX to avoid browser HTML entity/tag collisions (e.g. w_{<t} -> w_{\lt t})
    def sanitize_math(latex_str):
        return re.sub(r'<([a-zA-Z0-9])', r'\\lt \1', latex_str)

    # 3. Protect display math $$ ... $$ and \[ ... \]
    math_store = {}
    def save_display_dollar(match):
        key = f"XXMATHBLOCK{len(math_store)}XX"
        content = sanitize_math(match.group(1).strip())
        math_store[key] = f"\n\n$$\n{content}\n$$\n\n"
        return f"\n\n{key}\n\n"

    def save_display_bracket(match):
        key = f"XXMATHBLOCK{len(math_store)}XX"
        content = sanitize_math(match.group(1).strip())
        math_store[key] = f"\n\n\\[\n{content}\n\\]\n\n"
        return f"\n\n{key}\n\n"

    text = re.sub(r'\$\$([\s\S]*?)\$\$', save_display_dollar, text)
    text = re.sub(r'\\\[([\s\S]*?)\\\]', save_display_bracket, text)

    # 4. Protect inline math $ ... $ and \( ... \)
    def save_inline_dollar(match):
        key = f"XXMATHINLINE{len(math_store)}XX"
        content = sanitize_math(match.group(0))
        math_store[key] = content
        return key

    def save_inline_paren(match):
        key = f"XXMATHINLINE{len(math_store)}XX"
        content = sanitize_math(match.group(0))
        math_store[key] = content
        return key

    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline_dollar, text)
    text = re.sub(r'\\\((.*?)\\\)', save_inline_paren, text)

    # 5. Ensure lists preceded by regular text have a blank line for proper Markdown parsing
    lines = text.split('\n')
    spaced_lines = []
    for idx, line in enumerate(lines):
        if idx > 0 and re.match(r'^\s*[-*]\s+', line):
            prev = lines[idx-1].strip()
            if prev and not re.match(r'^\s*[-*]\s+', prev) and not prev.startswith(('#', '<')):
                spaced_lines.append('')
        spaced_lines.append(line)
    text = '\n'.join(spaced_lines)

    # 6. Restore code blocks before markdown parsing
    for k, v in code_store.items():
        text = text.replace(k, v)

    # 7. Parse Markdown to HTML
    html = markdown.markdown(text, extensions=[
        'tables',
        'fenced_code',
        'def_list',
        'attr_list',
        'sane_lists',
        'md_in_html'
    ])

    # 7. Restore protected math blocks
    for k, v in math_store.items():
        html = html.replace(f"<p>{k}</p>", v)
        html = html.replace(k, v)

    # 8. Post-process tables with semantic HTML styling attributes
    html = re.sub(r'<table>', '<table border="1" cellpadding="8" cellspacing="0">', html)

    # 9. Auto-inject Step IDs and Table of Contents if this is a chapter
    if is_chapter:
        html = process_step_headings_and_toc(html, lang)

    return html

def extract_first_heading(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# '):
                return line[2:].strip()
    return None

class AnchorCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if 'id' in d and d['id']:
            self.ids.add(d['id'])
        if tag == 'a' and 'href' in d:
            self.links.append((self.getpos(), d['href']))
        elif tag == 'img' and 'src' in d:
            self.links.append((self.getpos(), d['src']))

def verify_site_integrity(html_files):
    file_data = {}
    for hf in html_files:
        if not os.path.exists(hf):
            continue
        with open(hf, 'r', encoding='utf-8') as f:
            parser = AnchorCollector()
            parser.feed(f.read())
            file_data[hf] = {'ids': parser.ids, 'links': parser.links}

    broken_links = 0
    total_links = 0
    total_anchors = 0
    roadmap_links = 0

    for hf, data in file_data.items():
        base_dir = os.path.dirname(hf)
        for (line, col), href in data['links']:
            total_links += 1
            if href.startswith(('http://', 'https://', 'mailto:', 'javascript:')):
                continue
            parsed = urllib.parse.urlparse(href)
            target_path = parsed.path
            target_frag = parsed.fragment

            if not target_path:
                if target_frag:
                    total_anchors += 1
                    if target_frag not in data['ids']:
                        print(f"  [ERROR] Broken in-page anchor in {hf}:{line}:{col} -> #{target_frag}")
                        broken_links += 1
            else:
                resolved = os.path.normpath(os.path.join(base_dir, target_path))
                if target_frag:
                    total_anchors += 1

                if not os.path.exists(resolved):
                    if re.search(r'(\d{2}[a-z]?-[\w-]+)', resolved):
                        roadmap_links += 1
                    else:
                        print(f"  [ERROR] Broken relative link in {hf}:{line}:{col} -> {href} (resolved: {resolved})")
                        broken_links += 1
                else:
                    if target_frag and resolved in file_data:
                        if target_frag not in file_data[resolved]['ids']:
                            print(f"  [ERROR] Broken cross-page anchor in {hf}:{line}:{col} -> {href} (#{target_frag} not in {resolved})")
                            broken_links += 1

    return broken_links, total_links, total_anchors, roadmap_links

def build_site():
    print("Building static site for 'The Math Behind Large Language Models'...")

    # Discover all chapter directories in alphabetical/numerical order
    chapter_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^[0-9]+', d)])
    
    chapters = []
    for cd in chapter_dirs:
        en_md = os.path.join(cd, 'index.md')
        zh_md = os.path.join(cd, 'index.zh.md')
        en_title = extract_first_heading(en_md) or cd
        zh_title = extract_first_heading(zh_md) or en_title
        chapters.append({
            'dir': cd,
            'en_md': en_md,
            'zh_md': zh_md if os.path.exists(zh_md) else None,
            'en_title': en_title,
            'zh_title': zh_title
        })

    print(f"Discovered {len(chapters)} chapters:")
    for idx, c in enumerate(chapters):
        print(f"  [{idx}] {c['dir']}: {c['en_title']}")

    # 1. Build root index.html (curriculum overview)
    if os.path.exists('curriculum.md'):
        with open('curriculum.md', 'r', encoding='utf-8') as f:
            curr_md = f.read()
        rendered_curr = render_markdown_to_html(curr_md, 'en', is_chapter=False)
        nav_bar = """  <nav aria-label="Curriculum Navigation">
    <p><strong>Curriculum Roadmap:</strong> Foundations to Frontier Alignment</p>
  </nav>
  <hr>"""
        root_html = HTML_TEMPLATE.format(
            LANG_CODE="en",
            PAGE_TITLE="Curriculum Overview",
            ROOT_PREFIX="",
            NAV_BAR=nav_bar,
            MAIN_CONTENT=rendered_curr
        )
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(root_html)
        print("Built root index.html")

    # 2. Build each chapter's index.html and index.zh.html
    for idx, c in enumerate(chapters):
        cd = c['dir']
        prev_c = chapters[idx - 1] if idx > 0 else None
        next_c = chapters[idx + 1] if idx < len(chapters) - 1 else None

        # Build English page
        if os.path.exists(c['en_md']):
            with open(c['en_md'], 'r', encoding='utf-8') as f:
                en_content = f.read()
            rendered_en = render_markdown_to_html(en_content, 'en', is_chapter=True)

            # Bottom chapter navigation (EN)
            prev_link_bottom = f'<a href="../{prev_c["dir"]}/index.html">&larr; {prev_c["en_title"]}</a> &bull; ' if prev_c else ''
            next_link_bottom = f' &bull; <a href="../{next_c["dir"]}/index.html">{next_c["en_title"]} &rarr;</a>' if next_c else ''
            bottom_nav_en = f"""<hr>
<nav aria-label="Chapter Navigation">
  <p>
    {prev_link_bottom}<a href="../index.html">Course Overview</a>{next_link_bottom}
  </p>
</nav>"""
            rendered_en = rendered_en.rstrip() + "\n\n" + bottom_nav_en

            # Nav links (EN)
            prev_link = f'<a href="../{prev_c["dir"]}/index.html">&larr; {prev_c["en_title"]}</a> &nbsp;|&nbsp; ' if prev_c else ''
            next_link = f' &nbsp;|&nbsp; <strong>Next:</strong> <a href="../{next_c["dir"]}/index.html">{next_c["en_title"]} &rarr;</a>' if next_c else ''
            
            zh_toggle = f'<a href="index.zh.html">中文</a>' if c['zh_md'] else '<span style="color:#888;">中文 (暂无)</span>'
            nav_bar_en = f"""  <nav aria-label="Site and Language Navigation">
    <p>
      {prev_link}<a href="../index.html">Home / Curriculum Overview</a>{next_link}
      &nbsp;|&nbsp;
      <strong>Language:</strong> 
      <a href="index.html"><strong>English (Active)</strong></a> &bull; 
      {zh_toggle}
    </p>
  </nav>
  <hr>"""

            en_html = HTML_TEMPLATE.format(
                LANG_CODE="en",
                PAGE_TITLE=c['en_title'],
                ROOT_PREFIX="../",
                NAV_BAR=nav_bar_en,
                MAIN_CONTENT=rendered_en
            )
            en_out = os.path.join(cd, 'index.html')
            with open(en_out, 'w', encoding='utf-8') as f:
                f.write(en_html)
            print(f"Built {en_out} ({len(en_html)} bytes)")

        # Build Chinese page
        if c['zh_md'] and os.path.exists(c['zh_md']):
            with open(c['zh_md'], 'r', encoding='utf-8') as f:
                zh_content = f.read()
            rendered_zh = render_markdown_to_html(zh_content, 'zh', is_chapter=True)

            # Bottom chapter navigation (ZH)
            prev_zh_bottom = f'<a href="../{prev_c["dir"]}/index.zh.html">&larr; {prev_c["zh_title"]}</a> &bull; ' if prev_c and prev_c['zh_md'] else (f'<a href="../{prev_c["dir"]}/index.html">&larr; {prev_c["en_title"]}</a> &bull; ' if prev_c else '')
            next_zh_bottom = f' &bull; <a href="../{next_c["dir"]}/index.zh.html">{next_c["zh_title"]} &rarr;</a>' if next_c and next_c['zh_md'] else (f' &bull; <a href="../{next_c["dir"]}/index.html">{next_c["en_title"]} &rarr;</a>' if next_c else '')
            bottom_nav_zh = f"""<hr>
<nav aria-label="Chapter Navigation">
  <p>
    {prev_zh_bottom}<a href="../index.html">目录概览</a>{next_zh_bottom}
  </p>
</nav>"""
            rendered_zh = rendered_zh.rstrip() + "\n\n" + bottom_nav_zh

            # Nav links (ZH)
            prev_zh_link = f'<a href="../{prev_c["dir"]}/index.zh.html">&larr; {prev_c["zh_title"]}</a> &nbsp;|&nbsp; ' if prev_c and prev_c['zh_md'] else (f'<a href="../{prev_c["dir"]}/index.html">&larr; {prev_c["en_title"]}</a> &nbsp;|&nbsp; ' if prev_c else '')
            next_zh_link = f' &nbsp;|&nbsp; <strong>下一章：</strong> <a href="../{next_c["dir"]}/index.zh.html">{next_c["zh_title"]} &rarr;</a>' if next_c and next_c['zh_md'] else (f' &nbsp;|&nbsp; <strong>下一章：</strong> <a href="../{next_c["dir"]}/index.html">{next_c["en_title"]} &rarr;</a>' if next_c else '')

            nav_bar_zh = f"""  <nav aria-label="Site and Language Navigation">
    <p>
      {prev_zh_link}<a href="../index.html">目录导航</a>{next_zh_link}
      &nbsp;|&nbsp;
      <strong>语言 / Language:</strong> 
      <a href="index.html">English</a> &bull; 
      <a href="index.zh.html"><strong>中文 (当前)</strong></a>
    </p>
  </nav>
  <hr>"""

            zh_html = HTML_TEMPLATE.format(
                LANG_CODE="zh-CN",
                PAGE_TITLE=c['zh_title'],
                ROOT_PREFIX="../",
                NAV_BAR=nav_bar_zh,
                MAIN_CONTENT=rendered_zh
            )
            zh_out = os.path.join(cd, 'index.zh.html')
            with open(zh_out, 'w', encoding='utf-8') as f:
                f.write(zh_html)
            print(f"Built {zh_out} ({len(zh_html)} bytes)")

    # 3. Quality Gate Assertions
    print("\nRunning automated Quality Gate checks...")
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    
    # Check content files only (curriculum.md and chapter folders)
    content_files = ['curriculum.md', 'index.html']
    for cd in chapter_dirs:
        content_files.extend([
            os.path.join(cd, 'index.md'),
            os.path.join(cd, 'index.html')
        ])
        if os.path.exists(os.path.join(cd, 'index.zh.md')):
            content_files.extend([
                os.path.join(cd, 'index.zh.md'),
                os.path.join(cd, 'index.zh.html')
            ])

    emoji_violations = 0
    leak_violations = 0

    for fpath in content_files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check emojis
        emojis = emoji_pattern.findall(content)
        if emojis:
            print(f"  [ERROR] Emoji detected in {fpath}: {emojis}")
            emoji_violations += 1

        # Check unrendered HTML tag leaks inside code blocks in .html (Bug A / Bug B)
        if fpath.endswith('.html'):
            m = re.findall(r'<pre><code>&lt;(strong|samp|kbd|fieldset|figure)&gt;', content)
            if m:
                print(f"  [ERROR] Unintended indented code block containing escaped HTML tags in {fpath}: {m}")
                leak_violations += 1

            # Check for un-restored placeholder leaks
            placeholders = re.findall(r'XX(?:MATHBLOCK|MATHINLINE|CODEBLOCK)\d+XX', content)
            if placeholders:
                print(f"  [ERROR] Unrestored compilation placeholder leak in {fpath}: {placeholders}")
                leak_violations += 1

            # Check for math tag collision (e.g. unescaped < followed by letter inside math)
            for im in re.finditer(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', content):
                math_inner = im.group(1)
                if re.search(r'<[a-zA-Z0-9]', math_inner):
                    print(f"  [ERROR] Unescaped HTML tag collision in math formula in {fpath}: ${math_inner}$")
                    leak_violations += 1

    # Verify site link & anchor integrity
    html_files = [f for f in content_files if f.endswith('.html')]
    broken_links, total_links, total_anchors, roadmap_links = verify_site_integrity(html_files)
    print(f"  [AUDIT] Verified {total_links} links, {total_anchors} anchors across {len(html_files)} HTML pages.")
    if roadmap_links > 0:
        print(f"  [AUDIT] {roadmap_links} roadmap curriculum links noted.")

    if emoji_violations > 0 or leak_violations > 0 or broken_links > 0:
        print(f"\nBUILD FAILED: Quality gate assertions not met (emojis: {emoji_violations}, leaks: {leak_violations}, broken links: {broken_links}).")
        sys.exit(1)

    print("\nAll quality gates PASSED! Site built successfully.")

if __name__ == '__main__':
    build_site()
