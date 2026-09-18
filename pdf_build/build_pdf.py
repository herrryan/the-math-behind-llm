#!/usr/bin/env python3
"""
PDF Book Generator for 'The Math Behind Large Language Models'
Consolidates all chapters into a high-fidelity, publication-grade PDF book
with KaTeX math, semantic HTML styling, and running headers/footers.

Usage:
    uv run --with pymupdf --with markdown python3 pdf_build/build_pdf.py
"""

import os
import re
import sys
import subprocess
import importlib.util
from pathlib import Path

# Add project root to sys.path to reuse build.py functions
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load local build.py explicitly
spec = importlib.util.spec_from_file_location("local_build", ROOT_DIR / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

import pymupdf

OUTPUT_DIR = ROOT_DIR / "pdf_build"
OUTPUT_DIR.mkdir(exist_ok=True)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Print and Book-Optimized High-Density Stylesheet
BOOK_CSS = """
  @page {
    size: A4 portrait;
    margin: 20mm 16mm 22mm 16mm;
  }

  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    background: #ffffff;
    margin: 0;
    padding: 0;
  }

  /* Page Break Controls */
  .page-break-before {
    break-before: page !important;
    page-break-before: always !important;
  }

  .page-break-after {
    break-after: page !important;
    page-break-after: always !important;
  }

  .avoid-break, figure, table, fieldset, .katex-display, pre {
    break-inside: avoid !important;
    page-break-inside: avoid !important;
  }

  h1, h2, h3, h4 {
    break-after: avoid !important;
    page-break-after: avoid !important;
    color: #111111;
  }

  /* Cover Page */
  .cover-page {
    min-height: 85vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 40mm 15mm;
    break-after: page;
  }

  .cover-badge {
    display: inline-block;
    padding: 4px 12px;
    border: 1px solid #999;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 25px;
    color: #555;
  }

  .cover-title {
    font-size: 32pt;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 15px 0;
    color: #111;
  }

  .cover-subtitle {
    font-size: 15pt;
    font-weight: 400;
    color: #555;
    margin: 0 0 40px 0;
    line-height: 1.4;
  }

  .cover-divider {
    width: 80px;
    height: 3px;
    background: #111;
    margin: 20px auto 40px auto;
  }

  .cover-meta {
    font-size: 10.5pt;
    color: #444;
    line-height: 1.8;
  }

  /* Master Table of Contents */
  .toc-page {
    padding: 20px 0;
    break-after: page;
  }

  .toc-title {
    font-size: 20pt;
    font-weight: 700;
    border-bottom: 2px solid #111;
    padding-bottom: 8px;
    margin-bottom: 25px;
  }

  .toc-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .toc-item {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 8px 0;
    border-bottom: 1px dotted #ccc;
    font-size: 11pt;
  }

  .toc-item-title {
    font-weight: 600;
    color: #111;
    text-decoration: none;
  }

  .toc-item-subtitle {
    font-size: 9.5pt;
    color: #666;
    margin-left: 8px;
  }

  /* Chapters */
  .chapter-section {
    break-before: page;
    margin-top: 10px;
  }

  .chapter-header {
    border-bottom: 2px solid #222;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }

  .chapter-number-tag {
    font-size: 10pt;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #777;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .chapter-title {
    font-size: 22pt;
    font-weight: 700;
    margin: 0;
    line-height: 1.25;
  }

  h2 {
    font-size: 14pt;
    border-bottom: 1px solid #dcdcd4;
    padding-bottom: 4px;
    margin-top: 25px;
    margin-bottom: 12px;
  }

  h3 {
    font-size: 12pt;
    margin-top: 20px;
    margin-bottom: 8px;
  }

  p, ul, ol {
    margin: 0.5rem 0;
    text-align: justify;
  }

  /* Callout Containers */
  fieldset {
    background-color: #fbfbf9;
    border: 1px solid #dcdcd4;
    padding: 0.6rem 0.9rem;
    margin: 0.85rem 0;
    border-radius: 2px;
  }

  legend {
    font-weight: bold;
    color: #222;
    padding: 0 6px;
    font-size: 10.5pt;
  }

  blockquote {
    border-left: 3px solid #888;
    margin: 0.6rem 0;
    padding: 0.3rem 0.8rem;
    color: #444;
    background: #fdfdfd;
  }

  /* Code, Badges, Tables */
  pre, code, kbd, samp {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 9.5pt;
  }

  pre {
    background: #fdfdfc;
    border: 1px solid #dcdcd4;
    padding: 8px 12px;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.35;
  }

  kbd {
    background: #f0f0f0;
    border: 1px solid #cccccc;
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 9pt;
  }

  samp { color: #555555; }
  mark { background: #ffffaa; padding: 1px 3px; }

  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.85rem 0;
    font-size: 9.5pt;
  }

  th, td {
    border: 1px solid #dcdcd4;
    padding: 5px 8px;
  }

  th {
    background-color: #f0eee6;
    text-align: left;
    font-weight: bold;
  }

  caption {
    caption-side: top;
    text-align: left;
    font-weight: bold;
    margin-bottom: 0.4rem;
    font-size: 10pt;
  }

  meter {
    width: 100%;
    height: 12px;
  }

  dl dt {
    font-weight: bold;
    margin-top: 10px;
  }

  dl dd {
    margin-left: 15px;
    margin-bottom: 10px;
  }

  /* KaTeX Adjustments */
  .katex-display {
    margin: 0.6em 0 !important;
    padding: 2px 0 !important;
    overflow-x: hidden;
  }

  .katex {
    font-size: 1.05em;
  }
"""

KATEX_SCRIPTS = """
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
  <script>
    function renderMath() {
      if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\\\(', right: '\\\\)', display: false },
            { left: '\\\\[', right: '\\\\]', display: true }
          ],
          throwOnError: false
        });
        document.body.setAttribute('data-katex-rendered', 'true');
      }
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', renderMath);
    } else {
      renderMath();
    }
  </script>
"""

def extract_chapter_body(md_content, lang='en', dir_name=''):
    """
    Renders chapter markdown to HTML and extracts the chapter body,
    stripping website navigation headers, web TOCs, and footers.
    Re-maps Step anchors to include dir_name to prevent ID collisions.
    """
    html = build.render_markdown_to_html(md_content, lang=lang, is_chapter=True)

    # Strip web in-page TOC
    html = re.sub(r'<nav aria-label=[\'"]Table of Contents[\'"]>[\s\S]*?</nav>', '', html, flags=re.IGNORECASE)
    # Strip web bottom Chapter Navigation
    html = re.sub(r'<nav aria-label=[\'"]Chapter Navigation[\'"]>[\s\S]*?</nav>', '', html, flags=re.IGNORECASE)

    # Extract <h1> title
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.IGNORECASE | re.DOTALL)
    title = h1_match.group(1).strip() if h1_match else dir_name
    title_clean = re.sub(r'<[^>]+>', '', title).strip()

    # Remove <h1> from body since we render a dedicated chapter header
    body = re.sub(r'<h1[^>]*>.*?</h1>', '', html, count=1, flags=re.IGNORECASE | re.DOTALL)

    # Re-scope step IDs to avoid document-wide collisions (e.g. id="step-1" -> id="step-01-vectors-1")
    def scope_step_ids(m):
        prefix = m.group(1)
        step_num = m.group(2)
        rest = m.group(3)
        return f'{prefix}id="step-{dir_name}-{step_num}"{rest}'

    body = re.sub(r'(<h2[^>]*\s+)id="step-(\d+)"([^>]*>)', scope_step_ids, body, flags=re.IGNORECASE)

    # Rewrite internal relative chapter links (e.g. ../01-vectors-and-spaces/index.html -> #chapter-01-vectors-and-spaces)
    def rewrite_chapter_links(m):
        target_dir = m.group(1)
        return f'href="#chapter-{target_dir}"'

    body = re.sub(r'href=[\'"](?:\.\./)?(\d{2}[a-z]?-[\w-]+)/(?:index(?:\.zh)?\.html)?[\'"]', rewrite_chapter_links, body)

    return title_clean, body.strip()

def discover_chapters():
    """
    Discovers all chapter directories in natural sorted order.
    """
    chapter_dirs = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(ROOT_DIR / d) and re.match(r'^[0-9]+', d)])
    chapters = []
    for cd in chapter_dirs:
        en_md = ROOT_DIR / cd / 'index.md'
        zh_md = ROOT_DIR / cd / 'index.zh.md'
        en_title = build.extract_first_heading(str(en_md)) or cd
        zh_title = build.extract_first_heading(str(zh_md)) or en_title
        chapters.append((cd, en_title, zh_title))
    return chapters

def assemble_book_html(lang='en'):
    """
    Assembles all chapters into a single consolidated HTML document for the specified language.
    """
    chapters = discover_chapters()
    is_en = (lang == 'en')

    book_title = "The Math Behind Large Language Models" if is_en else "大语言模型背后的数学原理"
    book_subtitle = "From 3-Year-Old Intuition to Production Rigor" if is_en else "从 3 岁直觉到工程严谨性"
    toc_heading = "Table of Contents" if is_en else "全书目录"
    author_text = "Mathematical Foundations Course" if is_en else "大语言模型数学基础课程"
    edition_text = "Static Pre-Compiled KaTeX Edition" if is_en else "静态编译 KaTeX 典藏版"

    chapter_entries = []
    chapter_contents = []

    for dir_name, en_heading, zh_heading in chapters:
        ch_dir = ROOT_DIR / dir_name
        md_file = ch_dir / ("index.md" if is_en else "index.zh.md")
        if not md_file.exists():
            md_file = ch_dir / "index.md"

        content = md_file.read_text(encoding='utf-8')
        title, body = extract_chapter_body(content, lang=lang, dir_name=dir_name)

        chapter_id = f"chapter-{dir_name}"
        chapter_entries.append((chapter_id, title))

        chapter_html = f"""
        <section class="chapter-section" id="{chapter_id}">
          <header class="chapter-header">
            <div class="chapter-number-tag">{dir_name.upper()}</div>
            <h1 class="chapter-title">{title}</h1>
          </header>
          <div class="chapter-content">
            {body}
          </div>
        </section>
        """
        chapter_contents.append(chapter_html)

    # Build TOC HTML
    toc_items_html = "\n".join([
        f'<li class="toc-item"><a class="toc-item-title" href="#{cid}">{title}</a></li>'
        for cid, title in chapter_entries
    ])

    full_html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>{book_title}</title>
  <style>
{BOOK_CSS}
  </style>
{KATEX_SCRIPTS}
</head>
<body>

  <!-- Cover Page -->
  <div class="cover-page">
    <div class="cover-badge">{edition_text}</div>
    <h1 class="cover-title">{book_title}</h1>
    <div class="cover-subtitle">{book_subtitle}</div>
    <div class="cover-divider"></div>
    <div class="cover-meta">
      <p><strong>{author_text}</strong></p>
      <p>Covering Modules 0 through 3 &bull; Chapters 00 to 07</p>
      <p><small>Pure Semantic HTML &bull; KaTeX Vectors &amp; Matrices</small></p>
    </div>
  </div>

  <!-- Master Table of Contents -->
  <div class="toc-page">
    <h2 class="toc-title">{toc_heading}</h2>
    <ul class="toc-list">
      {toc_items_html}
    </ul>
  </div>

  <!-- Chapters Content -->
  {"".join(chapter_contents)}

</body>
</html>
"""
    output_html_path = OUTPUT_DIR / f"book-{lang}.html"
    output_html_path.write_text(full_html, encoding='utf-8')
    print(f"Generated consolidated HTML: {output_html_path} ({len(full_html)} bytes)")
    return output_html_path

def render_pdf_with_chrome(html_path, output_pdf_path):
    """
    Invokes Google Chrome in headless mode with KaTeX rendering time budget.
    """
    print(f"Printing PDF via Headless Chrome: {html_path.name} -> {output_pdf_path.name}...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=12000",
        f"--print-to-pdf={output_pdf_path}",
        str(html_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not output_pdf_path.exists():
        print("Chrome error output:", res.stderr)
        raise RuntimeError(f"Chrome failed to generate PDF: {output_pdf_path}")
    print(f"Raw PDF generated: {output_pdf_path} ({output_pdf_path.stat().st_size} bytes)")

def stamp_headers_and_footers(pdf_path, lang='en'):
    """
    Uses PyMuPDF to insert vector running headers and page numbers.
    Skips page 1 (cover) and page 2 (table of contents).
    """
    print(f"Stamping headers and page numbers onto {pdf_path.name}...")
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    is_en = (lang == 'en')
    book_title = "The Math Behind Large Language Models" if is_en else "大语言模型背后的数学原理"
    header_font = "helv" if is_en else "china-ss"
    footer_font = "helv"

    # A4 standard coordinates: 595.3 x 841.9 pt
    header_y = 35.0
    footer_y = 815.0

    for idx, page in enumerate(doc):
        page_num = idx + 1
        # Skip cover page (1) and TOC page (2)
        if page_num <= 2:
            continue

        rect = page.rect
        width = rect.width

        # Running Header
        header_text = book_title
        page.insert_text(
            pymupdf.Point(45, header_y),
            header_text,
            fontsize=8.5,
            fontname=header_font,
            color=(0.4, 0.4, 0.4)
        )

        # Draw light rule beneath header
        page.draw_line(
            pymupdf.Point(45, header_y + 6),
            pymupdf.Point(width - 45, header_y + 6),
            color=(0.85, 0.85, 0.85),
            width=0.5
        )

        # Running Footer (Page Number)
        footer_text = f"{page_num}"
        text_len = pymupdf.get_text_length(footer_text, fontname=footer_font, fontsize=9)
        footer_x = (width - text_len) / 2.0

        page.insert_text(
            pymupdf.Point(footer_x, footer_y),
            footer_text,
            fontsize=9.0,
            fontname=footer_font,
            color=(0.35, 0.35, 0.35)
        )

    # Save stamped PDF
    stamped_pdf_path = pdf_path.parent / pdf_path.name.replace(".raw.pdf", ".pdf")
    doc.save(str(stamped_pdf_path))
    doc.close()

    # Clean up raw un-stamped PDF
    if pdf_path != stamped_pdf_path:
        pdf_path.unlink()

    print(f"Final publication PDF complete: {stamped_pdf_path} ({total_pages} pages, {stamped_pdf_path.stat().st_size} bytes)")
    return stamped_pdf_path, total_pages

def build_language_edition(lang='en'):
    html_path = assemble_book_html(lang=lang)
    raw_pdf_path = OUTPUT_DIR / f"the-math-behind-llm-{lang}.raw.pdf"
    render_pdf_with_chrome(html_path, raw_pdf_path)
    final_pdf_path, total_pages = stamp_headers_and_footers(raw_pdf_path, lang=lang)
    return final_pdf_path, total_pages

def main():
    print("=" * 70)
    print("PDF Book Builder for The Math Behind Large Language Models")
    print("=" * 70)

    en_pdf, en_pages = build_language_edition(lang='en')
    zh_pdf, zh_pages = build_language_edition(lang='zh')

    print("\n" + "=" * 70)
    print("PDF BUILD SUMMARY:")
    print(f"  - English Edition: {en_pdf} ({en_pages} pages)")
    print(f"  - Chinese Edition: {zh_pdf} ({zh_pages} pages)")
    print("=" * 70)

if __name__ == "__main__":
    main()
