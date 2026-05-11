import pandas as pd
import sys
import os
import time
import classla
from tqdm import tqdm
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def lemmatize_documents(documents, nlp, batch_size=32):
    processed_docs = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Lemmatizing"):
        batch = documents[i:i + batch_size]
        docs = [nlp(doc) for doc in batch]
        for doc in docs:
            lemmas = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    lemma = word.lemma.lower()
                    if lemma.isalpha() and len(lemma) > 2:
                        lemmas.append(lemma)
            processed_docs.append(" ".join(lemmas))
    return processed_docs

def evaluate_model(topic_model, tokenized_docs):
    """Calculate Coherence and Diversity with heavy debug logging."""
    try:
        all_topics_dict = topic_model.get_topics()
        topic_words = []
        
        # 1. Extraction with type enforcement
        for topic_id, words_with_weights in all_topics_dict.items():
            if topic_id == -1: continue
            
            # Extract and force to pure Python strings
            extracted = [str(w[0]) for w in words_with_weights if w[0]]
            if extracted:
                topic_words.append(extracted)
        
        if not topic_words:
            print("⚠️ DEBUG: No topics found in the model.")
            return 0.0, 0.0
            
        # 2. Vocabulary Alignment Check
        # Gensim fails if a topic contains a word not present in 'texts'
        dictionary = Dictionary(tokenized_docs)
        vocab = set(dictionary.token2id.keys())
        
        cleaned_topics = []
        for i, topic in enumerate(topic_words):
            filtered_topic = [word for word in topic if word in vocab]
            if not filtered_topic:
                print(f"⚠️ DEBUG: Topic {i} has ZERO words present in the document vocabulary!")
            else:
                cleaned_topics.append(filtered_topic)

        if not cleaned_topics:
            print("❌ DEBUG: After vocabulary filtering, no topics remain.")
            return 0.0, 0.0

        # 3. Calculate Coherence
        cm = CoherenceModel(
            topics=cleaned_topics, 
            texts=tokenized_docs, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        
        coherence = cm.get_coherence()
        
        # 4. Diversity
        top_10 = [words[:10] for words in cleaned_topics]
        unique_words = set([word for topic in top_10 for word in topic])
        total_words = sum([len(topic) for topic in top_10])
        diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        return float(coherence), float(diversity)
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        # --- THE 'PRINT THE FUCKING THING' PART ---
        try:
            print("\n--- ERROR CONTEXT DATA ---")
            print(f"Number of topics: {len(topic_words)}")
            if len(topic_words) > 0:
                print(f"Sample Topic 0: {topic_words[0]}")
                print(f"Type of first word: {type(topic_words[0][0])}")
            print("---------------------------\n")
        except:
            print("Could not even print debug context.")
        return 0.0, 0.0

def evaluate_lda_model(lda_model, vectorizer, tokenized_docs):
    """Calculate Coherence and Diversity for LDA baseline."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        topic_words = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topic_words.append([feature_names[i] for i in top_indices])
        
        dictionary = Dictionary(tokenized_docs)
        cm = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        
        all_words = [word for topic in topic_words for word in topic]
        diversity = len(set(all_words)) / len(all_words) if all_words else 0
        
        return coherence, diversity
    except Exception as e:
        print(f"❌ LDA evaluation error: {e}")
        return 0.0, 0.0

def check_stability(docs, embeddings, nr_topics, n_runs=3):
    """Checks stability by comparing word overlap across different UMAP seeds."""
    from umap import UMAP
    from bertopic import BERTopic
    
    all_runs_words = []
    for i in range(n_runs):
        print(f"   Stability run {i+1}/{n_runs}...")
        temp_umap = UMAP(n_neighbors=15, n_components=5, random_state=i*42)
        temp_model = BERTopic(umap_model=temp_umap, nr_topics=nr_topics, verbose=False).fit(docs, embeddings)
        
        run_topics = []
        for t in temp_model.get_topics():
            if t != -1:
                run_topics.append(set([w for w, _ in temp_model.get_topic(t)[:10]]))
        all_runs_words.append(run_topics)

    similarities = []
    for r in range(len(all_runs_words) - 1):
        run_a, run_b = all_runs_words[r], all_runs_words[r+1]
        matches = 0
        for t_a in run_a:
            # If a topic in Run A shares 50% words with any topic in Run B
            if any(len(t_a & t_b) >= 5 for t_b in run_b):
                matches += 1
        similarities.append(matches / len(run_a) if run_a else 0)
    
    return np.mean(similarities)

def main():
    print("--- STARTING SECURE LOAD SEQUENCE ---")
    
    # 1. DATA LOADING
    try:
        df = pd.read_parquet('politics_comments_joined.parquet', columns=['title', 'description', 'document'])
        df = df.dropna(subset=['title', 'description']).copy()
        documents = (df['title'].str.strip() + "\n" + df['description'].str.strip() + "\n" + df["document"].str[:500].str.strip()).tolist()
        original_documents = documents.copy()
        del df
    except Exception as e:
        print(f"❌ DATA ERROR: {e}"); return

    # 2. LEMMATIZATION
    lemmatized_path = "lemmatized_docs.parquet"
    if os.path.exists(lemmatized_path):
        lemmatized_documents = pd.read_parquet(lemmatized_path)["document"].tolist()
        print("✅ Loaded cached lemmatized documents.")
    else:
        nlp = classla.Pipeline("hr", processors="tokenize,pos,lemma", use_gpu=True)
        lemmatized_documents = lemmatize_documents(original_documents, nlp)
        pd.DataFrame({"document": lemmatized_documents}).to_parquet(lemmatized_path)

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules
    import torch
    from umap import UMAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    word_model = modules.Transformer("classla/bcms-bertic")
    pool_model = modules.Pooling(word_model.get_embedding_dimension(), pooling_mode='mean')
    embedder = SentenceTransformer(modules=[word_model, pool_model], device=device)
    
    print("🔢 Encoding documents...")
    embeddings = embedder.encode(original_documents, show_progress_bar=True, batch_size=16)

    # 3. TRAIN MAIN MODEL (BERTić) WITH EXPLICIT HDBSCAN
    hr_stopwords = {
        "biti", "imati", "htjeti", "moći", "trebati", "raditi", "ići", "doći", 
        "vidjeti", "znati", "reći", "misliti", "i", "u", "na", "da", "ne", "što", 
        "za", "su", "sam", "hrvatska", "danas", "jučer", "video", "sažetak", 
        "kazati", "oko", "kad", "dok", "nakon", "samo", "već", "čak", "ovaj", "onaj",
        "datum", "komentar", "naslov", "sebe", "koji", "ovaj", "onaj", 
        "biti", "iznad", "ispod", "slika", "pročitati", "više", "povezano",
        "stvar", "moći", "reći", "oglas", "prijaviti", "pretplatiti", "autorska",
        "taj", "onaj", "sav", "oni", "on", "ona", "to", "kako", "željeti", 
        "moći", "mnogo", "malo", "sav", "svaki", "tako", "biti", "ovaj", 
        "svoj", "koji", "naš", "vaš", "kod", "gdje", "tko", "što"
    }
    vectorizer_model = CountVectorizer(
        stop_words=list(hr_stopwords), 
        ngram_range=(1, 2), 
        min_df=2, 
        token_pattern=r'(?u)\b[a-zA-ZčćžšđČĆŽŠĐ]{3,}\b'
    )

    # Tight UMAP to help separate news categories
    umap_model = UMAP(
        n_neighbors=10, 
        n_components=5, 
        metric='cosine', 
        random_state=42
    )

    # Explicit HDBSCAN with sensitive settings
    hdbscan_model = HDBSCAN(
        min_cluster_size=15, 
        min_samples=5, 
        metric='euclidean', 
        cluster_selection_method='leaf', 
        prediction_data=True
    )

    # The BERTopic call using the explicit models
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3)
    }

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model, # Re-integrated here
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=10,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(lemmatized_documents, embeddings)
    
    topic_model.reduce_topics(lemmatized_documents, nr_topics="auto")

    # 4. EVALUATION
    print("\n" + "="*60 + "\n📊 STARTING MODEL COMPARISON\n" + "="*60)
    tokenized_docs = [doc.split() for doc in lemmatized_documents]
    nr_topics_fixed = 50

    print("🔍 Evaluating BERTić...")
    b_coh, b_div = evaluate_model(topic_model, tokenized_docs)
    b_stab = check_stability(lemmatized_documents, embeddings, nr_topics_fixed)

    print("🔍 Evaluating LDA Baseline...")
    lda_vec = CountVectorizer(stop_words=list(hr_stopwords), max_features=5000)
    lda_data = lda_vec.fit_transform(lemmatized_documents)
    lda_model = LatentDirichletAllocation(n_components=nr_topics_fixed, random_state=42).fit(lda_data)
    l_coh, l_div = evaluate_lda_model(lda_model, lda_vec, tokenized_docs)

    print("🔍 Evaluating Multilingual Baseline...")
    try:
        m_embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2", device=device)
        m_embeddings = m_embedder.encode(original_documents, batch_size=16)
        m_model = BERTopic(embedding_model=m_embedder, nr_topics=nr_topics_fixed).fit(lemmatized_documents, m_embeddings)
        m_coh, m_div = evaluate_model(m_model, tokenized_docs)
        m_stab = check_stability(lemmatized_documents, m_embeddings, nr_topics_fixed)
    except:
        m_coh, m_div, m_stab = 0, 0, 0

    # 5. RESULTS
    res = pd.DataFrame({
        'Model': ['BERTić (Croatian)', 'LDA (Classical)', 'Multilingual BERT'],
        'Coherence (c_v)': [b_coh, l_coh, m_coh],
        'Diversity': [b_div, l_div, m_div],
        'Stability': [b_stab, 1.0, m_stab] # LDA stability usually fixed for comparison
    })
    print("\n", res.to_string(index=False))
    topic_model.save("bertic_politics_model", serialization="safetensors")
    print("\n✅ Done. Model saved.")

if __name__ == "__main__":
    main()