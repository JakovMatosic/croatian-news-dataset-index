import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the enriched data
print("📊 Loading enriched data...")
df = pd.read_parquet('politics_final_enriched.parquet')

# Convert month period to string if it isn't already for plotting
df['month_str'] = df['month'].astype(str)

# Set the visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# --- PLOT 1: MONTHLY TOXICITY TREND ---
plt.figure()
monthly_tox = df.groupby('month_str')['is_toxic'].mean() * 100 # Convert to %

ax1 = monthly_tox.plot(kind='line', marker='o', color='#e74c3c', linewidth=2)
plt.title('Average Toxicity (%) in Croatian Political Comments', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Percentage of Toxic Comments', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('toxicity_timeline.png')
print("✅ Timeline saved as 'toxicity_timeline.png'")

# --- PLOT 2: TOPIC TOXICITY LEADERBOARD ---
plt.figure(figsize=(10, 8))

# Get top words/names for the topics
topic_stats = df.groupby('topic_id')['is_toxic'].mean().reset_index()
# Filter out Topic -1 (Outliers) for a cleaner chart
topic_stats = topic_stats[topic_stats['topic_id'] != -1]

# Sort by toxicity
topic_stats = topic_stats.sort_values(by='is_toxic', ascending=False).head(15)

sns.barplot(
    data=topic_stats, 
    x='is_toxic', 
    y='topic_id', 
    palette='flare',
    orient='h'
)

plt.title('Top 15 Most Toxic Topics (Average %)', fontsize=14, fontweight='bold')
plt.xlabel('Toxicity Score (0.0 - 1.0)', fontsize=12)
plt.ylabel('Topic ID', fontsize=12)
plt.tight_layout()
plt.savefig('topic_toxicity_rank.png')
print("✅ Leaderboard saved as 'topic_toxicity_rank.png'")

print("\n🚀 Visualization complete! Check your folder for the PNG files.")