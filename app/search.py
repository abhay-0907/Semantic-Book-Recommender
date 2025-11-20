import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CLEAN_PATH = os.path.join("data", "processed", "books_clean.parquet")
EMB_PATH = os.path.join("data", "processed", "book_embeddings.npy")
INDEX_PATH = os.path.join("data", "processed", "faiss_index.bin")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_resources():
    df = pd.read_parquet(CLEAN_PATH)
    embeddings = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return df, embeddings, index, model

def search_books(query_text: str, top_k: int = 5):
    df, emb, index, model = load_resources()

    q_vec = model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, top_k)

    print(f"\nQuery: {query_text}\n")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        row = df.iloc[idx]
        title = row.get("Book", "Unknown title")
        author = row.get("Author", "Unknown author")
        desc = str(row.get("Description", ""))[:250].replace("\n", " ")
        print(f"{rank}. {title} by {author}  (score: {score:.3f})")
        print(f"   {desc}...\n")

if __name__ == "__main__":
    q = input("Describe what you feel like reading: ")
    search_books(q, top_k=5)
