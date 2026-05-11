import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
from transformers import pipeline
import torch
from scipy import stats

# 1. Redefine the exact same embedder you used to train
print("🔢 Re-initializing BERTić embedder...")
device = "cuda" if torch.cuda.is_available() else "cpu"
word_model = modules.Transformer("classla/bcms-bertic")
pool_model = modules.Pooling(word_model.get_embedding_dimension(), pooling_mode='mean')
embedder = SentenceTransformer(modules=[word_model, pool_model], device=device)

# 2. Load the model and EXPLICITLY link the embedder
print("📂 Loading BERTopic model...")
model_path = "bertic_politics_model"
topic_model = BERTopic.load(model_path, embedding_model=embedder)

# 3. Now transform will work!
print("🏷️ Mapping documents to topics...")
df = pd.read_parquet('politics_comments_joined.parquet')
topics, _ = topic_model.transform(df['document'].tolist())
df['topic_id'] = topics

# 2. Toxicity Pipeline (GPU Accelerated)
print("☢️ Initializing Toxicity Classifier on GPU...")
# device=0 specifically tells Transformers to use your first CUDA device
tox_pipe = pipeline("text-classification", 
                    model="classla/bcms-bertic-frenk-hate", 
                    device=0)

print("🧪 Analyzing toxicity (GPU Power Active)...")
# batch_size=64 or 128 is usually the sweet spot for a modern GPU
results = tox_pipe(df['document'].str[:512].tolist(), batch_size=64)

# Convert labels: 'Acceptable' -> 0 (Clean), 'Other' -> 1 (Toxic/Hate)
df['is_toxic'] = [1 if r['label'] != 'Acceptable' else 0 for r in results]

# 3. Monthly Trends
print("📈 Calculating monthly trends...")
# Since you have the 'month' column, we'll use it for the timeline
monthly_tox = df.groupby('month')['is_toxic'].mean().reset_index()

# 4. Topic Analysis
topic_info = topic_model.get_topic_info()
topic_tox = df.groupby('topic_id')['is_toxic'].mean().reset_index()
topic_tox = topic_tox.merge(topic_info[['Topic', 'Name']], left_on='topic_id', right_on='Topic')

# 5. Statistical Evaluation (ANOVA)
groups = [group['is_toxic'].values for name, group in df.groupby('topic_id') if name != -1]
f_stat, p_val = stats.f_oneway(*groups)

# --- FINAL OUTPUT ---
print("\n" + "="*40)
print("📊 FINAL TOXICITY REPORT")
print("="*40)
print(f"ANOVA P-Value: {p_val:.4e}")

print("\n🔥 TOP 3 MOST TOXIC TOPICS (Hate Speech Focus):")
# Sorting to find the 'hottest' topics
toxic_leaderboard = topic_tox.sort_values(by='is_toxic', ascending=False)
print(toxic_leaderboard[['Name', 'is_toxic']].head(3))

# Save for your final index/database
df.to_parquet('politics_final_enriched.parquet')
print("\n✅ Analysis complete. Results saved to 'politics_final_enriched.parquet'")