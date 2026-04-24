import pandas as pd

# 1. Load the files
news_df = pd.read_csv('index_politics_final.csv')
comments_df = pd.read_csv('index_comments_results.csv')

# 2. Group comments by URL
# Changed 'comment_text' to 'text' to match your file
comments_grouped = comments_df.groupby('url')['text'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

# 3. Merge tables (Left Join)
# This keeps every news article and attaches the grouped comments
df = pd.merge(news_df, comments_grouped, on='url', how='left')

# Fill empty comments with an empty string so the formatting doesn't fail
df['text'] = df['text'].fillna('')

# 4. Create the 'document' column
# Using the specific columns you listed: url, date, title, description, text
df['document'] = df.apply(lambda x: (
    f"Naslov:\n{x['title']}\n"
    f"Sažetak:\n{x['description']}\n"
    f"Datum:\n{x['date']}\n"
    f"Komentari:\n{x['text']}"
), axis=1)

# 5. Add time columns
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.to_period('M')

# 6. Filtriraj na političke članke koji imaju barem nekoliko komentara
# (Filter out articles where the combined comment text is very short/empty)
df_filtered = df[df['text'].str.len() > 10].copy()

# 7. Spremi rezultat kao politics_comments_joined.parquet
df_filtered.to_parquet('politics_comments_joined.parquet', index=False)

print("Success! The merged dataset is ready.")
print(f"Total articles processed: {len(df_filtered)}")