import pandas as pd
import numpy as np

# 🔥 ANTI-CRASH HACK FOR PANDAS 2.0+ AND BERTOPIC 🔥
# Intercept pd.to_datetime to strip out the deleted 'infer_datetime_format' parameter
original_to_datetime = pd.to_datetime
def patched_to_datetime(*args, **kwargs):
    kwargs.pop('infer_datetime_format', None) # Safely delete the forbidden argument
    return original_to_datetime(*args, **kwargs)
pd.to_datetime = patched_to_datetime
# =================================================

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup & Load Data
print("📂 Loading data...")
df = pd.read_parquet('politics_final_enriched.parquet')
df['month_str'] = df['month'].astype(str)
df = df.sort_values('month_str')

# 2. Re-initialize the BERTić embedder (Ensures model has its brain)
print("🔢 Re-initializing BERTić embedder...")
device = "cuda" if torch.cuda.is_available() else "cpu"
word_model = modules.Transformer("classla/bcms-bertic")
pool_model = modules.Pooling(word_model.get_embedding_dimension(), pooling_mode='mean')
embedder = SentenceTransformer(modules=[word_model, pool_model], device=device)

# 3. Load BERTopic explicitly linking the embedder
print("📂 Loading BERTopic model...")
topic_model = BERTopic.load("bertic_politics_model", embedding_model=embedder)

# 4. Map documents explicitly to populate model's internal states
print("🏷️ Mapping topics to internal states...")
topics, _ = topic_model.transform(df['document'].tolist())
df['topic_id'] = topics

# 5. Extract BERTopic Over Time Data (This will run perfectly now!)
print("⏳ Computing Topics Over Time profiles...")
topics_over_time = topic_model.topics_over_time(
    docs=df['document'].tolist(), 
    timestamps=df['month_str'].tolist(),
    topics=list(topics)
)

# 3. Calculate Temporal Stability (Cosine Similarity between adjacent months)
print("🧮 Calculating month-to-month topic stability...")
stability_records = []
all_topics = [t for t in topics_over_time['Topic'].unique() if t != -1]

for topic in all_topics:
    topic_data = topics_over_time[topics_over_time['Topic'] == topic].sort_values('Timestamp')
    timestamps = topic_data['Timestamp'].tolist()
    
    # Need at least two consecutive points to check stability
    for i in range(len(timestamps) - 1):
        m1, m2 = timestamps[i], timestamps[i+1]
        
        # Pull vectors from BERTopic's internal dynamic c-TF-IDF representation
        words_m1 = topic_data[topic_data['Timestamp'] == m1]['Words'].values[0]
        words_m2 = topic_data[topic_data['Timestamp'] == m2]['Words'].values[0]
        
        # Quick intersection similarity check on top terms
        set1, set2 = set(words_m1.split(', ')), set(words_m2.split(', '))
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
        
        stability_records.append({
            'Topic': topic,
            'Period': f"{m1} -> {m2}",
            'Stability': jaccard_sim
        })

df_stability = pd.DataFrame(stability_records)
print(f"📊 Mean Vocabulary Stability Across Topics: {df_stability['Stability'].mean():.3f}")

# 4. Statistical Testing: The 2024 Parliamentary Election Shock
# Event Target: April 2024 (2024-04)
print("🧪 Running structural break testing for April 2024 Elections...")

df['is_post_election'] = (df['month_str'] >= '2024-04').astype(int)

pre_election_tox = df[df['is_post_election'] == 0]['is_toxic']
post_election_tox = df[df['is_post_election'] == 1]['is_toxic']

u_stat, p_val = stats.mannwhitneyu(pre_election_tox, post_election_tox, alternative='two-sided')

print("\n" + "="*40)
print("📊 TEMPORAL EVENT REPORT")
print("="*40)
print(f"Pre-Election Mean Toxicity:  {pre_election_tox.mean()*100:.2f}%")
print(f"Post-Election Mean Toxicity: {post_election_tox.mean()*100:.2f}%")
print(f"Mann-Whitney U p-value:     {p_val:.4e}")

if p_val < 0.05:
    print("✅ Result: The 2024 election period caused a statistically significant structural shift in comment toxicity.")
else:
    print("❌ Result: No statistically viable toxicity break found near the election threshold.")

# 5. Visualizing Evolution alongside Context Milestones (FILTERED FOR 2022+)
plt.figure(figsize=(15, 7))
sns.set_theme(style="whitegrid")

# Pretvaramo toksičnost u postotke (0.0 - 1.0 -> 0% - 100%) ako već nije u bazi
if df['is_toxic'].max() <= 1.0:
    df['toxic_pct'] = df['is_toxic'] * 100
else:
    df['toxic_pct'] = df['is_toxic']

# Grupiranje na mjesečnoj razini
monthly_metrics = df.groupby('month_str')['toxic_pct'].mean().reset_index()

# 🔥 KLJUČNI DODATAK: Filtriramo podatke i ostavljamo samo od siječnja 2022. nadalje
monthly_metrics = monthly_metrics[monthly_metrics['month_str'] >= '2022-01'].reset_index(drop=True)

# Crtanje osnovne linije trenda (sada samo za razdoblje 2022-2026)
ax = sns.lineplot(
    data=monthly_metrics, 
    x='month_str', 
    y='toxic_pct', 
    marker='o', 
    color='#e74c3c', 
    linewidth=2.5, 
    label='Toksičnost %'
)

# --- PAMETNO PRORJEĐIVANJE X-OSI ---
# Budući da sada imamo manje mjeseci na grafu (oko 50 umjesto 105), prikazujemo svaki 3. mjesec za optimalnu čitljivost
x_ticks = monthly_metrics['month_str'].values
plt.xticks(ticks=range(0, len(x_ticks), 3), labels=x_ticks[::3], rotation=45, ha='right', fontsize=10)

# --- CIK-CAK POZICIONIRANJE OZNAKA DOGAĐAJA ---
milestones = {
    '2024-04': 'Parlamentarni\nizbori',
    '2024-05': 'Sastavljanje\nIII. Vlade',
    '2024-06': 'Europski\nizbori',
    '2024-12': 'Predsjednički\nizbori (1.k)',
    '2025-01': 'Pobjeda Milanovića\n(2.k)'
}

# Definiramo visine unutar stvarnog raspona podataka (između 50% i 90% jer linija više ne pada na nulu)
height_levels = [88, 78, 66, 84, 72]

for idx, (date, label) in enumerate(milestones.items()):
    if date in monthly_metrics['month_str'].values:
        # Crtanje vertikalne isprekidane linije
        plt.axvline(x=date, color='#2c3e50', linestyle='--', alpha=0.6, linewidth=1.2)
        
        # Odabir visine za ovaj specifični tekst
        current_y = height_levels[idx % len(height_levels)]
        
        # Ispis natpisa s elegantnim bijelim okvirom
        plt.text(
            date, current_y, label, 
            rotation=0, fontsize=9, weight='bold', color='#2c3e50',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', boxstyle='round,pad=0.3')
        )

# Minimalističko poliranje ostatka grafikona
plt.title("Evolucija toksičnosti komentara kroz ključne političke događaje u RH (2022 - 2026)", fontsize=15, weight='bold', pad=15)
plt.xlabel("Vremenska crta (Mjesec)", fontsize=12, labelpad=10)
plt.ylabel("Postotak toksičnih komentara (%)", fontsize=12, labelpad=10)

# Sužavamo Y-os na raspon od 50% do 95% jer se u tom rasponu kreću stvarni podaci od 2022.
# Na taj način će se bolje vidjeti sitne oscilacije i stvarni utjecaj izbora!
plt.ylim(50, 95) 

plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#bdc3c7')
plt.tight_layout()

plt.savefig('political_evolution_timeline.png', dpi=300)
print("\n📈 Prettier filtered visualization compiled to 'political_evolution_timeline.png'")