import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# --- HACK ZA PANDAS 2.0+ I BERTOPIC ---
original_to_datetime = pd.to_datetime
def patched_to_datetime(*args, **kwargs):
    kwargs.pop('infer_datetime_format', None)
    return original_to_datetime(*args, **kwargs)
pd.to_datetime = patched_to_datetime
# =====================================

print("📂 Loading final data and model...")
df = pd.read_parquet('politics_final_enriched.parquet')
df['month_str'] = df['month'].astype(str)

# Ponovno podizanje embeddera radi BERTopic kompatibilnosti
device = "cuda" if torch.cuda.is_available() else "cpu"
word_model = modules.Transformer("classla/bcms-bertic")
pool_model = modules.Pooling(word_model.get_embedding_dimension(), pooling_mode='mean')
embedder = SentenceTransformer(modules=[word_model, pool_model], device=device)

topic_model = BERTopic.load("bertic_politics_model", embedding_model=embedder)

# Osiguravamo svježe mapiranje tema (duljina 16768)
print("🏷️ Re-mapping documents to ensure perfect alignment...")
topics, _ = topic_model.transform(df['document'].tolist())
df['topic_id'] = topics

# Filtriramo DataFrame za čisti dio podataka (2022+) za line-plotove
df_filtered = df[df['month_str'] >= '2022-01'].copy()

print("⏳ Computing Topics Over Time profiles (2022+)...")
topics_over_time = topic_model.topics_over_time(
    docs=df_filtered['document'].tolist(), 
    timestamps=df_filtered['month_str'].tolist(),
    topics=[t for t, df_t in zip(topics, df['month_str']) if df_t >= '2022-01']
)

# ==========================================
# 📊 FAZA 1: BERTopic INTERAKTIVNE VIZUALIZACIJE (HTML)
# ==========================================
print("🌐 Generating BERTopic interactive HTML visualizations...")

fig_topics = topic_model.visualize_topics()
fig_topics.write_html("bertopic_2d_space.html")

fig_barchart = topic_model.visualize_barchart(top_n_topics=12)
fig_barchart.write_html("bertopic_top_words_barchart.html")

fig_tot = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=8)
fig_tot.write_html("bertopic_topics_over_time.html")

print("✅ HTML vizualizacije spremljene (otvori ih dvoklikom u pregledniku).")

# ==========================================
# 📈 FAZA 2: LINE PLOTOVI (Sveobuhvatni trendovi od 2022.)
# ==========================================
print("📉 Plotting monthly volume, toxicity, and dominant topics...")

# Priprema podataka na mjesečnoj razini
df_filtered['is_toxic_num'] = df_filtered['is_toxic'].astype(float)
if df_filtered['is_toxic_num'].max() <= 1.0:
    df_filtered['is_toxic_num'] = df_filtered['is_toxic_num'] * 100

monthly_agg = df_filtered.groupby('month_str').agg(
    broj_komentara=('document', 'count'),
    prosjecna_toksicnost=('is_toxic_num', 'mean')
).reset_index()

# Traženje dominantne teme za svaki mjesec
dominant_topics = df_filtered.groupby(['month_str', 'topic_id']).size().reset_index(name='count')
dominant_topics = dominant_topics.sort_values(['month_str', 'count'], ascending=[True, False])
dominant_monthly_topic = dominant_topics.drop_duplicates(subset=['month_str']).copy()

# Mapiramo ID teme u njezine ključne riječi radi lakšeg čitanja na grafu
topic_labels = {t: "_".join([w[0] for w in words[:2]]) for t, words in topic_model.get_topics().items()}
dominant_monthly_topic['labels'] = dominant_monthly_topic['topic_id'].map(topic_labels)

# Spajanje svega u jedan pregledni graf s dvije Y-osi
fig, ax1 = plt.subplots(figsize=(15, 7))
sns.set_theme(style="whitegrid")

# Lijeva Y-os: Broj komentara (Stupci)
ax1.bar(monthly_agg['month_str'], monthly_agg['broj_komentara'], color='#34495e', alpha=0.15, label='Broj komentara')
ax1.set_ylabel('Mjesečni volumen komentara', color='#34495e', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#34495e')

# Desna Y-os: Toksičnost (Linija)
ax2 = ax1.twinx()
sns.lineplot(data=monthly_agg, x='month_str', y='prosjecna_toksicnost', marker='o', color='#e74c3c', linewidth=2.5, ax=ax2, label='Toksičnost %')
ax2.set_ylabel('Prosječna toksičnost (%)', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')

# Dodavanje natpisa dominantnih tema direktno na graf (svaki drugi mjesec radi čitljivosti)
for idx, row in dominant_monthly_topic.iterrows():
    m_idx = monthly_agg[monthly_agg['month_str'] == row['month_str']].index
    if len(m_idx) > 0 and m_idx[0] % 2 == 0:
        y_pos = monthly_agg.loc[m_idx[0], 'broj_komentara']
        ax1.text(row['month_str'], y_pos * 1.05, f"T{row['topic_id']}\n({row['labels']})", 
                 fontsize=8, ha='center', color='#2c3e50', rotation=30,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))

ax1.set_xticklabels(monthly_agg['month_str'].values[::2], rotation=45, ha='right')
ax1.set_xticks(range(0, len(monthly_agg), 2))
plt.title("Analiza volumena, toksičnosti i dominantnih tema po mjesecima (2022 - 2026)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('monthly_comprehensive_trends.png', dpi=300)

# ==========================================
# 🔍 FAZA 3: PREGLED INTERPRETABILNOSTI TOP TEMA
# ==========================================
print("\n🔍 Evaluating topic interpretability and sample comments...")
top_topics = df_filtered['topic_id'].value_counts().head(12).index.tolist()
if -1 in top_topics:
    top_topics.remove(-1) # Preskačemo šum za kvalitativnu analizu

print("\n=======================================================")
print("📋 KVALITATIVNI PREGLED INTERPRETABILNOSTI")
print("=======================================================")

for t_id in top_topics[:10]:
    words = ", ".join([w[0] for w in topic_model.get_topic(t_id)[:7]])
    print(f"\n📌 TEMA {t_id} | Ključne riječi: {words}")
    
    # Izvlačimo 3 reprezentativna komentara za ovu temu
    samples = df_filtered[df_filtered['topic_id'] == t_id]['document'].head(3).tolist()
    for i, sample in enumerate(samples):
        clean_sample = sample.replace('\n', ' ')[:120] + "..." if len(sample) > 120 else sample
        print(f"  [{i+1}] {clean_sample}")

# ==========================================
# 🛡️ FAZA 4: ROBUSTNESS CHECKS (Testiranje stabilnosti hiperparametara)
# ==========================================
print("\n🛡️ Running Robustness Checks (Topic Reduction Stability Analysis)...")

# Simuliramo smanjenje broja tema (min_topic_size stabilnost kroz reduce_topics)
# Kopiramo model kako ne bismo uništili originalni učitani model na disku
model_reduced_15 = deepcopy(topic_model)
# POPRAVAK: Eksplicitno sinkroniziramo interno stanje modela s ispravnom duljinom (16768)
model_reduced_15.topics_ = list(topics)
model_reduced_15.reduce_topics(df['document'].tolist(), nr_topics=15)
topics_15 = model_reduced_15.topics_

model_reduced_10 = deepcopy(topic_model)
model_reduced_10.topics_ = list(topics)
model_reduced_10.reduce_topics(df['document'].tolist(), nr_topics=10)
topics_10 = model_reduced_10.topics_

# Računamo korelaciju raspodjele dokumenata
# Ako dokumenti koji su bili zajedno u velikom modelu ostanu grupirani i u manjem, model je stabilan
df['topics_r15'] = topics_15
df['topics_r10'] = topics_10

corr_15 = df['topic_id'].corr(df['topics_r15'], method='spearman')
corr_10 = df['topic_id'].corr(df['topics_r10'], method='spearman')

# Priprema teksta izvještaja za konzolu i datoteku
robustness_report = f"""
=======================================================
🛡️ REPORT PROVJERE ROBUSNOSTI MODELA
=======================================================
Spearmanova korelacija stabilnosti (Original -> 15 Tema):  {corr_15:.4f}
Spearmanova korelacija stabilnosti (Original -> 10 Tema):  {corr_10:.4f}
"""

if corr_15 > 0.70:
    robustness_report += "\n✅ REZULTAT: Visoka korelacija dokazuje da je struktura tema stabilna i robusna.\n"
    robustness_report += "   Smanjenjem granularnosti (broja tema) dokumenti se logički spajaju u veće cjeline,\n"
    robustness_report += "   što znači da model nije preosjetljiv na sitne promjene u hiperparametrima.\n"
else:
    robustness_report += "\n⚠️ REZULTAT: Umjerena stabilnost. Promjena broja tema značajnije preslaguje strukturu.\n"

print(robustness_report)

# ==========================================
# 💾 SPREMANJE SVIH REZULTATA KVALITATIVNE ANALIZE U DATOTEKU
# ==========================================
print("💾 Saving interpretability and robustness reports to disk...")
with open("model_evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("=======================================================\n")
    f.write("📋 KVALITATIVNI PREGLED INTERPRETABILNOSTI I STRUKTURE TEMA\n")
    f.write("=======================================================\n")
    
    for t_id in top_topics[:10]:
        words = ", ".join([w[0] for w in topic_model.get_topic(t_id)[:7]])
        f.write(f"\n📌 TEMA {t_id} | Ključne riječi: {words}\n")
        
        samples = df_filtered[df_filtered['topic_id'] == t_id]['document'].head(3).tolist()
        for i, sample in enumerate(samples):
            clean_sample = sample.replace('\n', ' ')[:120] + "..." if len(sample) > 120 else sample
            f.write(f"  [{i+1}] {clean_sample}\n")
            
    f.write("\n" + robustness_report)

print("📝 Izvještaj uspješno spremljen u datoteku 'model_evaluation_report.txt'")
print("\n🚀 Sve faze izvršene uspješno! Provjeri HTML datoteke i PNG grafikon u direktoriju.")