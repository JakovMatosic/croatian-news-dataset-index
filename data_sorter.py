import pandas as pd

# Load the CSV
INPUT_FILE = "index_hr_clean_2002_2026.csv"
OUTPUT_FILE = "index_hr_clean_vijesti_only.csv"

df = pd.read_csv(INPUT_FILE)

# Filter for news articles: keep only rows where URL contains '/vijesti/'
news_only = df[df['url'].str.contains('/vijesti/', case=False, na=False)]

# Save to new file
news_only.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"Filtered {len(df)} rows down to {len(news_only)} news articles.")
print(f"Saved to {OUTPUT_FILE}")
