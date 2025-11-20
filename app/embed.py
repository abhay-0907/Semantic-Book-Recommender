import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

CLEAN_PATH = os.path.join("data", "processed", "books_clean.parquet")
EMB_PATH = os.path.join("data", "processed", "book_embeddings.npy")
INDEX_PATH = os.path.join("data", "processed", "faiss_index.bin")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_clean_books(path: str = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)

def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    texts = df["semantic_text"].tolist()
    print(f"Encoding {len(texts)} books...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")  # FAISS likes float32

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatIP, path: str = INDEX_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"Saved FAISS index to {path}")

if __name__ == "__main__":
    df = load_clean_books()
    embeddings = build_embeddings(df)

    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, embeddings)
    print(f"Saved embeddings to {EMB_PATH}")

    index = build_faiss_index(embeddings)
    save_index(index)