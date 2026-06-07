# Croatian News Dataset Index 📰

A high-performance toolkit for crawling, indexing, and analyzing over 20 years of Croatian news from Index.hr. This repository supports everything from raw sitemap scraping to advanced NLP Topic Modeling using BERTić.

## 🚀 Quick Start (Recommended)

This project uses `uv` for extremely fast, reliable dependency management and to ensure correct GPU acceleration setup.

1. **Install `uv` (if you haven't already):**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Create Environment & Install Dependencies:**
   ```bash
   uv venv --python 3.12
   # On Windows: .\ .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
   > Note: The `requirements.txt` is configured for NVIDIA GPU (CUDA 12.4) support. This is required for the BERTić modeling scripts to run efficiently.

## 📊 Dataset Overview

The archive contains a comprehensive record of Index.hr metadata spanning from 2002 to February 2026.

- **Total Articles:** ~91 MB of cleaned metadata.
- **Content:** URL, Date (YYYY-MM-DD), Headline, and Lead Summary.
- **Category Focus:** Specialized subsets for "Vijesti" (News) and "Politics."

## 🛠️ Repository Contents
| File | Purpose |
# Croatian News Dataset Index

A toolkit for crawling, indexing, and analyzing Croatian news from Index.hr (2002–2026). This repository includes scrapers, data cleaning utilities, comment extraction, and topic-modeling analysis with the Croatian BERTić model.

**Status:** research / analysis code. Use for research and educational purposes; respect Index.hr terms of service and robots.txt.

**This README covers:** repository layout, environment setup (Windows/Linux), key scripts, how to run them, and troubleshooting notes.

## Quick setup (recommended)

You can use the included `dataset_venv` (already provisioned) or create a fresh venv. These instructions use a new venv for clarity.

Windows (PowerShell):
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU notes:
- The `requirements.txt` contains a PyTorch wheel index URL for CUDA 12.4 (`--index-url`). If you do not have an NVIDIA GPU, install CPU-only PyTorch or remove the index line and install a CPU wheel.
- If using the included `dataset_venv`, activate it with `dataset_venv\\Scripts\\activate` (Windows) or `source dataset_venv/bin/activate` (Unix).

## Repository overview

- **Data files**: `index_hr_clean_2002_2026.csv`, `index_hr_clean_vijesti_only.csv`, `index_politics_final.csv` — cleaned metadata CSVs.
- **Scrapers**:
  - `index-scraper.py`: crawl sitemaps and build raw CSV archives.
  - `comment_scraper.py`: extract user comments (can output Parquet).
- **Processing / utilities**:
  - `data_sorter.py`: filter/clean datasets by category/date range.
  - `comment_news_merger.py`: merge article metadata with comment datasets.
  - `news_classifier.py`: simple heuristics classifier for quick tag/category filtering.
- **Modeling & analysis**:
  - `bertic_model_quality_control.py`: runs BERTopic-style topic modeling with Croatian BERTić embeddings and evaluates topic quality.
  - `inspect_topics.py`, `temporal evaluation.py`, `visualizations_robustness.py`, `toxicity_analysis.py`, `toxicity_visualiser.py`: analysis & viz helpers.
- **Outputs**:
  - `extracted_topics_report.csv`, `model_evaluation_report.txt`, various `.html` visualizations produced by the modeling scripts.

## Common workflows

1) Scrape article metadata
```bash
python index-scraper.py --output index_hr_clean_2002_2026.csv
```

2) Filter for politics
```bash
python data_sorter.py --input index_hr_clean_2002_2026.csv --category politics --output index_politics_final.csv
```

3) Extract comments for a set of articles
```bash
python comment_scraper.py --input index_politics_final.csv --output politics_comments_joined.parquet
```

4) Topic modeling and evaluation (BERTić)
```bash
python bertic_model_quality_control.py --input index_politics_final.csv --outdir outputs/bertic_quality
```
The script will create HTML visualizations and CSV reports in the output directory.

## How to run safely on Windows

- Activate a fresh venv before importing heavy packages like `torch` or `pandas` to avoid DLL conflicts.
- If you see import-time crashes, try importing `torch` after `pandas` or use the provided `dataset_venv`.

## Troubleshooting

- Missing CUDA / GPU issues: install matching `torch` wheel for your CUDA version, or use CPU-only wheels.
- Memory errors during modeling: reduce batch sizes or run on smaller subsets (`--sample N`).
- If a script is silent on failure, run it with `python -u script.py` to see unbuffered logs.

## Development notes

- The codebase is research-focused and may include quick utility scripts (not production hardened).
- Tests are not provided; validate runs on a small sample before large-scale processing.

## Files changed
- Updated `README.md` and `requirements.txt` to improve setup guidance and reproducibility.

## License

MIT. Article content and metadata are the property of Index.hr; use responsibly and for research/education.

If you'd like, I can run a local dependency check or pin exact versions from the `dataset_venv` to `requirements.txt` next.
