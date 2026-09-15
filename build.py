#!/usr/bin/env python3
"""
Simple static site generator for 'The Math Behind LLMs'
Converts Markdown content in chapter folders to pure HTML without any JavaScript.
Zero external dependencies (uses standard Python library).
"""

import os
import re
import html
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

def parse_markdown(text):
    """
    Converts simple markdown to semantic HTML.
    Supports:
    - Headers (# through ####)
    - Bold (**text**), Italic (*text*)
    - Block math ($$ math $$) and inline math ($ math $)
    - Blockquotes (including special callout tags like [!INTUITION], [!MATH], [!ORIGIN], [!EXAMPLE])
    - Unordered & ordered lists
    - Code blocks (```lang ... ```)
    - Inline code (`code`)
    - Tables (| col1 | col2 |)
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
        # Escape HTML entities first, except already formatted spans/tags
        s = html.escape(s)
        
        # Display math $$ ... $$
        s = re.sub(r'\$\$(.*?)\$\$', r'<div class="math-display">\1</div>', s)
        # Inline math $ ... $
        s = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'<span class="math-inline">\1</span>', s)
        
        # Bold
        s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
        # Italic
        s = re.sub(r'\*(.*?)\*', r'<em>\1</em>', s)
        # Inline code
        s = re.sub(r'`(.*?)`', r'<code>\1</code>', s)
        # Links
        s = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', s)
        return s

    def render_table(rows):
        if not rows:
            return ""
        out = ['<div class="table-container"><table>']
        # First row is header
        header = rows[0]
        out.append("<thead><tr>")
        for cell in header:
            out.append(f"<th>{format_inline(cell.strip())}</th>")
        out.append("</tr></thead>")
        
        out.append("<tbody>")
        for row in rows[1:]:
            # Skip separator row (e.g. ---|---)
            if all(re.match(r'^:?-+:?$', c.strip()) for c in row):
                continue
            out.append("<tr>")
            for cell in row:
                out.append(f"<td>{format_inline(cell.strip())}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")
        return "\n".join(out)

    def render_blockquote(lines):
        full_text = "\n".join(lines).strip()
        callout_type = "quote"
        title = ""
        
        # Check for custom callout markers
        callout_match = re.match(r'^\[!(INTUITION|MATH|ORIGIN|EXAMPLE|TIP|NOTE|WARNING)\](?:\s+(.*))?', full_text, re.IGNORECASE)
        if callout_match:
            ctype = callout_match.group(1).upper()
            custom_title = callout_match.group(2)
            content = full_text[callout_match.end():].strip()
            
            titles = {
                "INTUITION": ("🧸 3-Year-Old Intuition", "callout-intuition"),
                "MATH": ("📐 The Exact Math", "callout-math"),
                "ORIGIN": ("🔍 Where Does This Formula Come From?", "callout-origin"),
                "EXAMPLE": ("🔢 Concrete Toy Example", "callout-example"),
                "TIP": ("💡 Core Takeaway", "callout-tip"),
                "NOTE": ("📝 Note", "callout-note"),
                "WARNING": ("⚠️ Common Pitfall", "callout-warning")
            }
            default_title, css_class = titles.get(ctype, ("Note", "callout-note"))
            box_title = custom_title if custom_title else default_title
            
            # Format lines inside callout
            inner_paras = [format_inline(p.strip()) for p in content.split("\n\n") if p.strip()]
            inner_html = "".join(f"<p>{p}</p>" for p in inner_paras)
            return f'<div class="callout {css_class}"><div class="callout-header">{box_title}</div><div class="callout-body">{inner_html}</div></div>'
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

        # List handling with indentation stack
        ul_match = re.match(r'^(\s*)[\*\-]\s+(.*)$', line)
        ol_match = re.match(r'^(\s*)(\d+)\.\s+(.*)$', line)

        if ul_match or ol_match:
            if in_blockquote or in_table:
                html_out.extend(close_blocks())

            indent = len(ul_match.group(1)) if ul_match else len(ol_match.group(1))
            l_type = "ul" if ul_match else "ol"
            item_content = ul_match.group(2) if ul_match else ol_match.group(3)

            # If we're not currently in a list, start one
            if not in_list:
                in_list = True
                list_stack = [(l_type, indent)]
                html_out.append(f"<{l_type}>")
            else:
                # Check indentation level relative to stack
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
            # Continuation of previous list item or block math inside list
            stripped = line.strip()
            if stripped.startswith("$$") and stripped.endswith("$$"):
                math_inner = format_inline(stripped)
                html_out.append(f"<div class=\"math-display-list\">{math_inner}</div>")
            else:
                html_out.append(f"<div class=\"list-detail\">{format_inline(stripped)}</div>")
            i += 1
            continue
        elif in_list:
            while list_stack:
                top_type, _ = list_stack.pop()
                html_out.append(f"</{top_type}>")
            in_list = False

        # Regular line or blank line
        if not line.strip():
            html_out.extend(close_blocks())
            i += 1
            continue

        # Normal Paragraph
        html_out.extend(close_blocks())
        para_lines = [line]
        while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#', '>', '```', '|', '*', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            i += 1
            para_lines.append(lines[i])
        
        p_text = format_inline(" ".join(para_lines))
        html_out.append(f"<p>{p_text}</p>")
        i += 1

    html_out.extend(close_blocks())
    return "\n".join(html_out)


def build_page_template(title, content_html, rel_root=".", prev_chap=None, next_chap=None, chapter_num=None):
    """
    Renders pure HTML without any JavaScript.
    """
    nav_html = []
    nav_html.append(f'<a href="{rel_root}/index.html" class="nav-btn">🏠 Home / Curriculum</a>')
    if prev_chap:
        nav_html.append(f'<a href="{rel_root}/{prev_chap["dir"]}/index.html" class="nav-btn">← {prev_chap["title"]}</a>')
    if next_chap:
        nav_html.append(f'<a href="{rel_root}/{next_chap["dir"]}/index.html" class="nav-btn nav-btn-primary">{next_chap["title"]} →</a>')
    
    top_nav = f'<nav class="top-nav">{"".join(nav_html)}</nav>'
    bottom_nav = f'<nav class="bottom-nav">{"".join(nav_html)}</nav>'
    
    chap_tag = f'<div class="chapter-badge">Chapter {chapter_num}</div>' if chapter_num is not None else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)} - The Math Behind LLMs</title>
  <link rel="stylesheet" href="{rel_root}/style.css">
</head>
<body>
  <div class="page-container">
    <header class="site-header">
      <div class="site-brand">
        <a href="{rel_root}/index.html">The Math Behind Large Language Models</a>
      </div>
      <p class="site-tagline">Intuitive, Rigorous, Pure HTML — Zero JavaScript</p>
    </header>

    {top_nav}

    <main class="content">
      {chap_tag}
      {content_html}
    </main>

    {bottom_nav}

    <footer class="site-footer">
      <p><strong>The Math Behind LLMs</strong> • Built with pure semantic HTML & CSS.</p>
      <p class="footer-subtext">No JavaScript • No external tracking • Grounded in physical intuition and rigorous math.</p>
    </footer>
  </div>
</body>
</html>
"""

def scan_and_build():
    """
    Scans for chapter directories and generates HTML files.
    """
    # Look for chapter folders: e.g. 00-xxx, 01-xxx, or chapter-xx
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
        
        # Check chapter number from folder name
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

    # Render root index.html if curriculum.md or index.md exists at root
    root_md = ROOT_DIR / "curriculum.md"
    if not root_md.exists():
        root_md = ROOT_DIR / "README.md"
    
    if root_md.exists():
        md_text = root_md.read_text(encoding="utf-8")
        first_h1 = re.search(r'^#\s+(.*)$', md_text, re.MULTILINE)
        title = first_h1.group(1).strip() if first_h1 else "The Math Behind LLMs - Curriculum"
        body_html = parse_markdown(md_text)
        
        # Link to first chapter if available
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
