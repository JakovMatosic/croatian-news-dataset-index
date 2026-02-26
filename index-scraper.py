import requests
import csv
import xml.etree.ElementTree as ET
import re
from datetime import datetime

def clean_text(text):
    if not text:
        return ""
    # Remove HTML tags if any remain
    text = re.sub('<[^<]+?>', '', text)
    # Replace non-breaking spaces and fix whitespace
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    return " ".join(text.split())

def scrape_index_archives(start_year, end_year):
    filename = f"index_hr_clean_{start_year}_{end_year}.csv"
    # Frequency is removed; focusing on the core content
    headers = ['url', 'date', 'title', 'description']
    
    # We use a session for better performance and to avoid getting blocked
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Stop at the current date (February 2026)
                if year == 2026 and month > 2:
                    break
                
                print(f"Scraping {month:02d}/{year}...")
                url = f"https://www.index.hr/sitemap.xml?month={month}&year={year}"
                
                try:
                    response = session.get(url, timeout=20)
                    if response.status_code != 200:
                        continue
                    
                    root = ET.fromstring(response.content)
                    nmsp = {
                        'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                        'image': 'http://www.google.com/schemas/sitemap-image/1.1'
                    }

                    for url_node in root.findall('sm:url', nmsp):
                        loc = url_node.find('sm:loc', nmsp).text if url_node.find('sm:loc', nmsp) is not None else ""
                        
                        # Index uses YYYY-MM-DD format in lastmod
                        lastmod = url_node.find('sm:lastmod', nmsp).text if url_node.find('sm:lastmod', nmsp) is not None else ""
                        date_only = lastmod.split('T')[0] if 'T' in lastmod else lastmod

                        img_node = url_node.find('image:image', nmsp)
                        title = ""
                        desc = ""
                        
                        if img_node is not None:
                            t_node = img_node.find('image:title', nmsp)
                            c_node = img_node.find('image:caption', nmsp)
                            
                            title = clean_text(t_node.text) if t_node is not None else ""
                            desc = clean_text(c_node.text) if c_node is not None else ""

                        # Only write rows that actually have content
                        if title or desc:
                            writer.writerow([loc, date_only, title, desc])
                        
                except Exception as e:
                    print(f"Error on {month}/{year}: {e}")

    print(f"\nFinished! Your data is in: {filename}")

# Run the script
scrape_index_archives(2002, 2026)