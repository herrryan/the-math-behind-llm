#!/usr/bin/env python3
"""
Static site generator for 'The Math Behind LLMs'
Converts Markdown content into clean, semantic HTML with KaTeX for mathematical formulas.
Zero custom CSS — pure browser-native semantic styling.
Zero external build dependencies (uses standard Python library).
"""

import os
import re
import html
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

def parse_markdown(text):
    """
    Converts markdown to pure semantic HTML.
    Supports:
    - Headers (# through ######)
    - Bold (**text**), Italic (*text*)
    - Block math ($$ math $$) and inline math ($ math $) via KaTeX
    - Callouts ([!INTUITION], [!MATH], etc.) rendered as native <fieldset><legend>
    - Unordered & ordered lists
    - Code blocks (```lang ... ```)
    - Inline code (`code`)
    - Tables (| col1 | col2 |) with native border attributes
    - Horizontal rules (---)
    - Links [text](url)
    - Paragraphs
    """
    lines = text.split("\n")
    html_out = []
    i = 0
    in_code = False
    code_lang = ""
    code_lines = []
    in_list = False
    list_stack = []
    in_table = False
    table_rows = []
    in_blockquote = False
    blockquote_lines = []

    def close_blocks():
        nonlocal in_list, list_stack, in_table, table_rows, in_blockquote, blockquote_lines
        res = []
        if in_list:
            while list_stack:
                top_type, _ = list_stack.pop()
                res.append(f"</{top_type}>")
            in_list = False
        if in_table:
            res.append(render_table(table_rows))
            table_rows = []
            in_table = False
        if in_blockquote:
            res.append(render_blockquote(blockquote_lines))
            blockquote_lines = []
            in_blockquote = False
        return res

    def format_inline(s):
        # 1. Extract display math $$ ... $$
        disp_math = []
        def save_disp(m):
            disp_math.append(m.group(1))
            return f"___MATH_DISP_{len(disp_math)-1}___"

        # 2. Extract inline math $ ... $
        inl_math = []
        def save_inl(m):
            inl_math.append(m.group(1))
            return f"___MATH_INL_{len(inl_math)-1}___"

        s = re.sub(r'\$\$(.*?)\$\$', save_disp, s, flags=re.DOTALL)
        s = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inl, s)

        # 3. Escape HTML entities on remaining text
        s = html.escape(s)

        # 4. Standard markdown formatting
        s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'\*(.*?)\*', r'<em>\1</em>', s)
        s = re.sub(r'`(.*?)`', r'<code>\1</code>', s)
        s = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', s)

        # 5. Restore display math intact for KaTeX
        for idx, raw in enumerate(disp_math):
            s = s.replace(f"___MATH_DISP_{idx}___", f"$${raw}$$")

        # 6. Restore inline math intact for KaTeX
        for idx, raw in enumerate(inl_math):
            s = s.replace(f"___MATH_INL_{idx}___", f"${raw}$")

        return s

    def render_table(rows):
        if not rows:
            return ""
        out = ['<table border="1" cellpadding="6" cellspacing="0">']
        header = rows[0]
        out.append("<thead><tr>")
        for cell in header:
            out.append(f"<th>{format_inline(cell.strip())}</th>")
        out.append("</tr></thead>")
        
        out.append("<tbody>")
        for row in rows[1:]:
            if all(re.match(r'^:?-+:?$', c.strip()) for c in row):
                continue
            out.append("<tr>")
            for cell in row:
                out.append(f"<td>{format_inline(cell.strip())}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)

    def render_blockquote(lines):
        full_text = "\n".join(lines).strip()
        callout_match = re.match(r'^\[!(INTUITION|MATH|ORIGIN|EXAMPLE|TIP|NOTE|WARNING)\](?:\s+(.*))?', full_text, re.IGNORECASE)
        if callout_match:
            ctype = callout_match.group(1).upper()
            custom_title = callout_match.group(2)
            content = full_text[callout_match.end():].strip()
            
            titles = {
                "INTUITION": "🧸 3-Year-Old Intuition",
                "MATH": "📐 The Exact Math",
                "ORIGIN": "🔍 Where Does This Formula Come From?",
                "EXAMPLE": "🔢 Concrete Toy Example",
                "TIP": "💡 Core Takeaway",
                "NOTE": "📝 Note",
                "WARNING": "⚠️ Common Pitfall"
            }
            default_title = titles.get(ctype, "Note")
            box_title = format_inline(custom_title) if custom_title else default_title
            
            inner_paras = [format_inline(p.strip()) for p in content.split("\n\n") if p.strip()]
            inner_html = "".join(f"<p>{p}</p>" for p in inner_paras)
            return f'<fieldset><legend><strong>{box_title}</strong></legend>{inner_html}</fieldset>'
        else:
            paras = [format_inline(p.strip()) for p in full_text.split("\n\n") if p.strip()]
            inner_html = "".join(f"<p>{p}</p>" for p in paras)
            return f'<blockquote>{inner_html}</blockquote>'

    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith("```"):
            if not in_code:
                html_out.extend(close_blocks())
                in_code = True
                code_lang = line.strip()[3:].strip()
                code_lines = []
            else:
                in_code = False
                escaped_code = html.escape("\n".join(code_lines))
                lang_attr = f' class="language-{code_lang}"' if code_lang else ""
                html_out.append(f'<pre><code{lang_attr}>{escaped_code}</code></pre>')
                code_lines = []
            i += 1
            continue
            
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$', line.strip()):
            html_out.extend(close_blocks())
            html_out.append("<hr>")
            i += 1
            continue

        # Standalone multiline block math $$ ... $$
        if line.strip() == "$$":
            html_out.extend(close_blocks())
            math_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "$$":
                i += 1
            raw_math = "\n".join(math_lines).strip()
            html_out.append(f"<p>$${raw_math}$$</p>")
            continue

        # Standalone single-line block math $$ ... $$
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 2:
            html_out.extend(close_blocks())
            raw_math = line.strip()[2:-2].strip()
            html_out.append(f"<p>$${raw_math}$$</p>")
            i += 1
            continue

        # Headers
        header_match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if header_match:
            html_out.extend(close_blocks())
            level = len(header_match.group(1))
            htext = format_inline(header_match.group(2))
            slug = re.sub(r'[^a-z0-9]+', '-', header_match.group(2).lower()).strip('-')
            html_out.append(f'<h{level} id="{slug}">{htext}</h{level}>')
            i += 1
            continue

        # Blockquote line
        if line.startswith(">"):
            if in_list or in_table:
                html_out.extend(close_blocks())
            in_blockquote = True
            bline = line[1:].strip() if line.startswith("> ") else line[1:]
            blockquote_lines.append(bline)
            i += 1
            continue
        elif in_blockquote:
            html_out.append(render_blockquote(blockquote_lines))
            blockquote_lines = []
            in_blockquote = False

        # Table row
        if "|" in line and line.strip().startswith("|") and line.strip().endswith("|"):
            if in_list or in_blockquote:
                html_out.extend(close_blocks())
            in_table = True
            cells = [c for c in line.strip().split("|")[1:-1]]
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            html_out.append(render_table(table_rows))
            table_rows = []
            in_table = False

        # Lists
        ul_match = re.match(r'^(\s*)([-*+])\s+(.*)$', line)
        ol_match = re.match(r'^(\s*)(\d+)\.\s+(.*)$', line)
        list_match = ul_match or ol_match
        
        if list_match:
            if in_table or in_blockquote:
                html_out.extend(close_blocks())
            
            indent = len(list_match.group(1))
            l_type = "ul" if ul_match else "ol"
            item_content = list_match.group(3)

            if not in_list:
                in_list = True
                list_stack = [(l_type, indent)]
                html_out.append(f"<{l_type}>")
            else:
                curr_type, curr_indent = list_stack[-1]
                if indent > curr_indent:
                    list_stack.append((l_type, indent))
                    html_out.append(f"<{l_type}>")
                elif indent < curr_indent:
                    while len(list_stack) > 1 and list_stack[-1][1] > indent:
                        top_type, _ = list_stack.pop()
                        html_out.append(f"</{top_type}>")
                elif l_type != curr_type:
                    top_type, _ = list_stack.pop()
                    html_out.append(f"</{top_type}>")
                    list_stack.append((l_type, indent))
                    html_out.append(f"<{l_type}>")

            html_out.append(f"<li>{format_inline(item_content)}</li>")
            i += 1
            continue
        elif in_list and (line.startswith("  ") or line.startswith("\t")):
            stripped = line.strip()
            if stripped.startswith("$$") and stripped.endswith("$$"):
                math_inner = format_inline(stripped)
                html_out.append(f"<p>{math_inner}</p>")
            else:
                html_out.append(f"<p>{format_inline(stripped)}</p>")
            i += 1
            continue
        elif in_list:
            while list_stack:
                top_type, _ = list_stack.pop()
                html_out.append(f"</{top_type}>")
            in_list = False

        # Blank line
        if not line.strip():
            html_out.extend(close_blocks())
            i += 1
            continue

        # Normal Paragraph
        html_out.extend(close_blocks())
        para_lines = [line]
        while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#', '>', '```', '|', '*', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '$$')):
            i += 1
            para_lines.append(lines[i])
        
        p_text = format_inline(" ".join(para_lines))
        html_out.append(f"<p>{p_text}</p>")
        i += 1

    html_out.extend(close_blocks())
    return "\n".join(html_out)


def build_page_template(title, content_html, rel_root=".", prev_chap=None, next_chap=None, chapter_num=None):
    """
    Renders pure semantic HTML with KaTeX for mathematical notation.
    Zero custom CSS files.
    """
    nav_links = [f'<a href="{rel_root}/index.html">🏠 Home / Curriculum</a>']
    if prev_chap:
        nav_links.append(f'<a href="{rel_root}/{prev_chap["dir"]}/index.html">← Previous: {prev_chap["title"]}</a>')
    if next_chap:
        nav_links.append(f'<a href="{rel_root}/{next_chap["dir"]}/index.html">Next: {next_chap["title"]} →</a>')
    
    top_nav = f'<nav><p>{" &nbsp;|&nbsp; ".join(nav_links)}</p></nav>'
    bottom_nav = f'<nav><p>{" &nbsp;|&nbsp; ".join(nav_links)}</p></nav>'
    chap_badge = f'<p><strong>Chapter {chapter_num}</strong></p>' if chapter_num is not None else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)} - The Math Behind LLMs</title>
  <!-- KaTeX for math formulas -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {{
            delimiters: [
              {{left: '$$', right: '$$', display: true}},
              {{left: '$', right: '$', display: false}}
            ],
            throwOnError: false
          }});"></script>
</head>
<body>
  <header>
    <h1><a href="{rel_root}/index.html">The Math Behind Large Language Models</a></h1>
    <p><em>Intuitive, Rigorous, Pure Semantic HTML with KaTeX Math</em></p>
    <hr>
  </header>

  {top_nav}
  <hr>

  <main>
    {chap_badge}
    {content_html}
  </main>

  <hr>
  {bottom_nav}

  <footer>
    <hr>
    <p><strong>The Math Behind LLMs</strong> • Built with pure semantic HTML and KaTeX.</p>
    <p><small>Grounded in physical intuition and mathematical rigor.</small></p>
  </footer>
</body>
</html>
"""

def scan_and_build():
    """
    Scans for chapter directories and generates HTML files.
    """
    chapter_dirs = []
    for item in sorted(ROOT_DIR.iterdir()):
        if item.is_dir() and (re.match(r'^\d{2}-', item.name) or item.name.startswith("chapter-")):
            md_file = item / "index.md"
            if md_file.exists():
                chapter_dirs.append(item)

    chapters_meta = []
    for cdir in chapter_dirs:
        md_text = (cdir / "index.md").read_text(encoding="utf-8")
        first_h1 = re.search(r'^#\s+(.*)$', md_text, re.MULTILINE)
        title = first_h1.group(1).strip() if first_h1 else cdir.name
        
        m = re.match(r'^(\d+)', cdir.name)
        cnum = int(m.group(1)) if m else None
        
        chapters_meta.append({
            "dir": cdir.name,
            "path": cdir,
            "title": title,
            "num": cnum,
            "md_text": md_text
        })

    # Render each chapter
    for idx, chap in enumerate(chapters_meta):
        prev_c = chapters_meta[idx - 1] if idx > 0 else None
        next_c = chapters_meta[idx + 1] if idx + 1 < len(chapters_meta) else None
        
        body_html = parse_markdown(chap["md_text"])
        full_html = build_page_template(
            title=chap["title"],
            content_html=body_html,
            rel_root="..",
            prev_chap=prev_c,
            next_chap=next_c,
            chapter_num=chap["num"]
        )
        out_file = chap["path"] / "index.html"
        out_file.write_text(full_html, encoding="utf-8")
        print(f"Generated: {out_file.relative_to(ROOT_DIR)}")

    # Render root index.html
    root_md = ROOT_DIR / "curriculum.md"
    if not root_md.exists():
        root_md = ROOT_DIR / "README.md"
    
    if root_md.exists():
        md_text = root_md.read_text(encoding="utf-8")
        first_h1 = re.search(r'^#\s+(.*)$', md_text, re.MULTILINE)
        title = first_h1.group(1).strip() if first_h1 else "The Math Behind LLMs - Curriculum"
        body_html = parse_markdown(md_text)
        
        first_c = chapters_meta[0] if chapters_meta else None
        full_html = build_page_template(
            title=title,
            content_html=body_html,
            rel_root=".",
            prev_chap=None,
            next_chap=first_c,
            chapter_num=None
        )
        root_html_file = ROOT_DIR / "index.html"
        root_html_file.write_text(full_html, encoding="utf-8")
        print(f"Generated root homepage: {root_html_file.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    scan_and_build()
