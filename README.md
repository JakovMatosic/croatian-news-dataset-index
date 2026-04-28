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
|-------------|---------|
| `index-scraper.py` | Crawl sitemaps and build raw CSV archives. |
| `comment_scraper.py` | High-speed extraction of user comments for sentiment/topic analysis. |
| `bertic_model_quality_control.py` | New: Topic modeling using BERTopic + classla/bcms-bertic. |
| `data_sorter.py` | Filter and clean datasets by category or date range. |
| `news_classifier.py` | Categorizes articles using keyword-based heuristics. |

## 🧠 Advanced Usage: Topic Modeling

To analyze the "Quality" of topics within the politics dataset using the Croatian BERTić model:
```bash
python bertic_model_quality_control.py
```

### Why this script is unique:
- **Hardware Accelerated:** Automatically detects and utilizes NVIDIA GPUs (e.g., RTX 4050).
- **Security Hardened:** Requires PyTorch 2.6.0+ to mitigate CVE-2025-32434.
- **Evaluation:** Directly compares BERTić results against a traditional LDA baseline using Coherence (cv​) and Diversity metrics.

## 🔧 Workflow Examples
1. **Scrape Article Metadata**
   ```bash
   python index-scraper.py --output index_hr_clean_2002_2026.csv
   ```
2. **Filter for Politics**
   ```bash
   python data_sorter.py --input index_hr_clean_2002_2026.csv --category politics --output index_politics_final.csv
   ```
3. **Extract Comments**
   ```bash
   python comment_scraper.py --input index_politics_final.csv --output politics_comments_joined.parquet
   ```

## ⚠️ Important Considerations
- **Environment Stability:** If the scripts crash silently on Windows, ensure you are using the `uv` environment. Some scripts use "Late Imports" to prevent DLL conflicts between torch and pandas.
- **Ethics:** Respect robots.txt and the terms of service of Index.hr. This dataset is intended for research and educational purposes only.

## License

Scripts are released under the MIT License. Article content and metadata are copyright of Index.hr.

Happy dataset exploring! If you find this useful for Croatian NLP research, feel free to contribute! 😊
