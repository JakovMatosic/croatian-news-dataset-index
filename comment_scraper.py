import pandas as pd
import requests
import re
import csv
import time
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Files
INPUT_CSV = "index_hr_clean_vijesti_only.csv"
OUTPUT_CSV = "index_comments_results.csv"

# Universal Thread ID Regex
THREAD_RE = re.compile(r'commentThreadId["\']?\s*[:=]\s*["\']?(\d+)')

def process_article(url):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.index.hr/'
    })
    
    try:
        resp = session.get(url, timeout=12)
        if resp.status_code != 200: return []
        
        thread_match = THREAD_RE.search(resp.text)
        if not thread_match: return []
        thread_id = thread_match.group(1)

        article_comments = []
        skip = 0
        
        while True:
            api_url = "https://www.index.hr/api/comments"
            params = {"sortBy": "2", "commentThreadId": thread_id, "skip": str(skip), "take": "20"}
            
            api_res = session.get(api_url, params=params, timeout=10)
            data = api_res.json()
            
            if not isinstance(data, dict): break

            # Modern vs Legacy logic
            if "comments" in data and isinstance(data["comments"], list):
                items = data["comments"]
            else:
                items = [v for k, v in data.items() if isinstance(v, dict)]

            if not items: break

            for c in items:
                article_comments.append([
                    url, thread_id, 
                    c.get('posterFullName', 'Anonymous'), 
                    c.get('content', ''), 
                    c.get('numberOfLikes', 0), 
                    c.get('numberOfDislikes', 0), 
                    c.get('createdDateUtc', '')
                ])
            
            if len(items) < 20: break
            skip += 20
            
        return article_comments
    except:
        return []

def run_full_scrape():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    urls = df['url'].tolist()

    # RESUME LOGIC
    done_urls = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            # We only read the URL column to save memory
            temp_df = pd.read_csv(OUTPUT_CSV, usecols=['url'], encoding='utf-8-sig')
            done_urls = set(temp_df['url'].unique())
        except:
            pass

    urls_to_scrape = [u for u in urls if u not in done_urls]
    
    print("-" * 50)
    print(f"Total articles in CSV:  {len(urls)}")
    print(f"Already completed:      {len(done_urls)}")
    print(f"Remaining to scrape:    {len(urls_to_scrape)}")
    print("-" * 50)

    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['url', 'thread_id', 'user', 'text', 'likes', 'dislikes', 'date'])

    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # Custom progress bar with clear time metrics
        progress_bar = tqdm(
            total=len(urls_to_scrape),
            unit="art",
            desc="Scraping Index.hr",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        with ThreadPoolExecutor(max_workers=12) as executor:
            for result in executor.map(process_article, urls_to_scrape):
                if result:
                    writer.writerows(result)
                    f.flush()
                progress_bar.update(1)
        
        progress_bar.close()

if __name__ == "__main__":
    run_full_scrape()