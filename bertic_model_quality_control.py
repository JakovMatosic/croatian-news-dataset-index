import torch
import pandas as pd
import sys
import os
import time

# Move imports inside to prevent circular import issues on some Windows setups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def main():
    print("--- STARTING SECURE LOAD SEQUENCE ---")
    sys.stdout.flush()

    # STEP 1: LOAD DATA FIRST (Before touching GPU)
    print("📂 Opening Parquet file...")
    sys.stdout.flush()
    
    try:
        if not os.path.exists('politics_comments_joined.parquet'):
            print("❌ File not found!")
            return

        # Using fastparquet + only 1 column + head(500) to be safe
        df = pd.read_parquet(
            'politics_comments_joined.parquet', 
            columns=['document'], 
            engine='fastparquet'
        )
        documents = df['document'].head(500).astype(str).tolist()
        print(f"✅ Data loaded. Found {len(documents)} documents.")
        sys.stdout.flush()
        del df # Free up RAM
    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        return

    # STEP 2: INITIALIZE GPU
    print("🚀 Initializing GPU...")
    sys.stdout.flush()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("📥 Loading BERTić...")
    sys.stdout.flush()
    try:
        word_model = Transformer("classla/bcms-bertic")
        pool_model = Pooling(word_model.get_embedding_dimension(), pooling_mode='mean')
        embedder = SentenceTransformer(modules=[word_model, pool_model], device=device)
        print("✅ BERTić Ready.")
    except Exception as e:
        print(f"❌ MODEL ERROR: {e}")
        return

    # STEP 3: ENCODING
    print("🔢 Encoding documents...")
    sys.stdout.flush()
    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=16)

    # STEP 4: BERTopic
    print("🧠 Training BERTopic...")
    sys.stdout.flush()
    topic_model = BERTopic(embedding_model=embedder, nr_topics=20, verbose=True)
    topics, _ = topic_model.fit_transform(documents, embeddings)

    print(f"✅ SUCCESS! Topics found: {len(set(topics))}")
    
    # Simple Metrics Print
    print("\nTop Topics:")
    print(topic_model.get_topic_info().head(5))

if __name__ == "__main__":
    main()