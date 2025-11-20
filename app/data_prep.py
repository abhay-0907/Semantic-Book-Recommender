import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "books.csv")
PROCESSED_PATH = os.path.join("data", "processed", "books_clean.parquet")

def load_raw_books(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Raw shape:", df.shape)
    print(df.head(3))
    return df

def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    # Column names from Best_Books_10k_Multi_Genre
    title_col = "Book"
    author_col = "Author"
    desc_col = "Description"
    rating_col = "Average Rating"

    # keep extra useful cols if present
    keep_cols = [
        c for c in [
            title_col,
            author_col,
            desc_col,
            rating_col,
            "Genres",
            "Number of Ratings",
            "URL",
        ] if c in df.columns
    ]
    df = df[keep_cols].copy()

    # basic cleaning
    df = df.dropna(subset=[title_col, desc_col])
    df = df.drop_duplicates(subset=[title_col, author_col, desc_col])
    df = df.reset_index(drop=True)
    df["book_id"] = df.index

    # build semantic_text for embeddings
    def make_semantic_text(row):
        title = str(row.get(title_col, "")).strip()
        author = str(row.get(author_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()
        rating = row.get(rating_col, None)
        genres = str(row.get("Genres", "")).strip()

        parts = [f"{title} by {author}".strip()]
        if genres:
            parts.append(f"Genres: {genres}.")
        if rating is not None and rating == rating:  # not NaN
            parts.append(f"Average rating: {rating}/5.")
        parts.append(f"Description: {desc}")
        return " ".join(parts)

    df["semantic_text"] = df.apply(make_semantic_text, axis=1)

    print("Cleaned shape:", df.shape)
    return df

def save_clean_books(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved cleaned books to {path}")

if __name__ == "__main__":
    df_raw = load_raw_books()
    df_clean = clean_books(df_raw)
    save_clean_books(df_clean)