# PDF Book Generator

This directory contains the self-contained, publication-grade PDF book generator for **The Math Behind Large Language Models** (covering both English and Chinese editions).

---

## How It Works

1. **Chapter Assembly**:
   - Discovers all numbered chapter directories in curriculum order (`00-next-word-prediction` through the latest chapter).
   - Reuses the core Markdown parsing and KaTeX math isolation pipeline from [`build.py`](../build.py).
   - Combines the chapters into consolidated static documents (`book-en.html` and `book-zh.html`) with a front cover, metadata, and master Table of Contents.

2. **Vector Math Rendering & Printing**:
   - Uses headless Google Chrome to render KaTeX equations, tables, figures, callouts, and typography into high-resolution vector PDF pages (`the-math-behind-llm-en.raw.pdf` and `the-math-behind-llm-zh.raw.pdf`).

3. **Running Headers & Pagination**:
   - Uses `pymupdf` to stamp running headers with a divider rule and centered page numbers (`Helvetica` for English, `china-ss` for Simplified Chinese).
   - Automatically skips the front cover and Table of Contents pages.

---

## Requirements

- **Google Chrome** installed at `/Applications/Google Chrome.app` (macOS default).
- Python package runner **`uv`** (or Python with `pymupdf` and `markdown`).

---

## Usage

To generate both the English and Chinese PDF books:

```bash
uv run --with pymupdf --with markdown python3 pdf_build/build_pdf.py
```

### Outputs

- [`the-math-behind-llm-en.pdf`](./the-math-behind-llm-en.pdf): Full English book (~85 pages).
- [`the-math-behind-llm-zh.pdf`](./the-math-behind-llm-zh.pdf): Full Chinese book (~91 pages).

---

## Clean-up

Because all PDF build scripts, temporary files, and output books reside exclusively inside `pdf_build/`, the entire experiment can be removed cleanly at any time by deleting this folder:

```bash
rm -rf pdf_build
```
