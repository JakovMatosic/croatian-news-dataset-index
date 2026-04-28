import pandas as pd
import sys
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"



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

        df = pd.read_parquet(
            'politics_comments_joined.parquet', 
            columns=['document'], 
            engine='pyarrow'
        )
        documents = df['document'].dropna().astype(str).tolist()
        print(f"✅ Data loaded. Found {len(documents)} documents.")
        sys.stdout.flush()
        del df # Free up RAM
    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        return
    
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules
    Transformer = modules.Transformer
    Pooling = modules.Pooling
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.ldamodel import LdaModel
    import torch

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
    topic_model = BERTopic(
        nr_topics="auto",   # better than fixed 20
        min_topic_size=15   # avoids tiny clusters
    )
    topics, _ = topic_model.fit_transform(documents, embeddings)

    print(f"✅ SUCCESS! Topics found: {len(set(topics))}")
    
    # Simple Metrics Print
    print("\nTop Topics:")
    print(topic_model.get_topic_info().head(5))

if __name__ == "__main__":
    main()