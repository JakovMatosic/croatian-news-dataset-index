# Croatian News Dataset Index 📰

A collection of scripts and data for crawling, indexing, and analyzing articles from
Index.hr and related Croatian news sources. The repository includes tools for scraping
sitemaps, sorting datasets, classifying articles, and extracting comments.

## Overview

This dataset contains a comprehensive archive of article metadata from the Croatian news
portal Index.hr, spanning from its early years in 2002 to February 2026.

## Data Source
Data was aggregated from public XML sitemaps hosted at `https://www.index.hr/sitemap.xml`.

## Dataset Structure
The CSV contains the following columns:
- **url**: The direct link to the article.
- **date**: Publication or last modified date (YYYY-MM-DD).
- **title**: The headline of the article.
- **description**: A short summary or lead paragraph.

### Included Data Files
- `index_hr_clean_2002_2026.csv` – full cleaned archive of Index.hr articles.
- `index_hr_clean_vijesti_only.csv` – subset containing only "Vijesti" (news) category.
- `index_comments_results.csv` – results from comment scraping routines.
- `index_politics_final.csv` – filtered set of politics-related articles.

## Technical Details
- **Format:** CSV (UTF-8 with BOM)
- **Size:** ~91 MB (final archive)
- **Row Count:** _(see dataset or run `wc -l index_hr_clean_2002_2026.csv`)_
- **Cleaning:** HTML tags and `&nbsp;` entities removed; whitespace normalized.

## Repository Contents
| File/Script | Purpose |
|-------------|---------|
| `index-scraper.py` | Crawl Index.hr sitemap and build raw CSV. |
| `data_sorter.py` | Filter and sort CSVs by category or date. |
| `news_classifier.py` | Simple classifier for article categories using keywords. |
| `comment_scraper.py` | Extract article comments for specified URLs. |
| `logs.txt` | Log output from scraping runs. |
| `README.md` | Project documentation. |

## Usage
1. **Scrape articles**
   ```sh
   python index-scraper.py --output index_hr_clean_2002_2026.csv
   ```
2. **Sort or filter**
   ```sh
   python data_sorter.py --input index_hr_clean_2002_2026.csv \ 
                         --category politics --output index_politics_final.csv
   ```
3. **Classify articles**
   ```sh
   python news_classifier.py --input index_hr_clean_2002_2026.csv --model keyword
   ```
4. **Scrape comments**
   ```sh
   python comment_scraper.py --url https://www.index.hr/... --output index_comments_results.csv
   ```

Adjust flags and file names as needed. Some scripts may require dependencies listed in
`requirements.txt` (create if necessary).

> ⚠️ Ensure you respect Index.hr's robots.txt and terms of service when scraping.

## License & Attribution
This compilation is provided for research and educational purposes. Content copyright
belongs to Index.hr. Scripts are released under the MIT License – see `LICENSE`.

---

Happy dataset exploring! 😊
