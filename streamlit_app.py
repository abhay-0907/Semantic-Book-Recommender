import os
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# Paths
CLEAN_PATH = os.path.join("data", "processed", "books_clean.parquet")
EMB_PATH = os.path.join("data", "processed", "book_embeddings.npy")
INDEX_PATH = os.path.join("data", "processed", "faiss_index.bin")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_resources():
    df = pd.read_parquet(CLEAN_PATH)
    embeddings = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return df, embeddings, index, model


def semantic_search(
    query_text: str,
    top_k: int = 5,
    min_rating: float = 0.0,
    focus_mode: str = "Anything",
):
    df, emb, index, model = load_resources()

    if not query_text.strip():
        return []

    # Encode query
    q_vec = model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    # Get more candidates than we finally show, so filters have room
    scores, idxs = index.search(q_vec, top_k * 3)

    raw_results = []
    for score, idx in zip(scores[0], idxs[0]):
        row = df.iloc[idx]
        raw_results.append(
            {
                "score": float(score),
                "title": row.get("Book", "Unknown title"),
                "author": row.get("Author", "Unknown author"),
                "description": str(row.get("Description", "")),
                "genres": str(row.get("Genres", "")),
                "avg_rating": row.get("Average Rating", None),
                "num_ratings": row.get("Number of Ratings", None),
                "url": row.get("URL", None),
            }
        )

    # Filter by minimum rating
    filtered = []
    for r in raw_results:
        rating = r["avg_rating"]
        try:
            rating_val = float(rating) if rating is not None else 0.0
        except ValueError:
            rating_val = 0.0

        if rating_val < min_rating:
            continue
        filtered.append(r)

    # Focus by rough genre buckets using the "Genres" text
    def matches_focus(r):
        g = r["genres"].lower()
        if focus_mode == "Anything":
            return True

        if focus_mode == "Self-help / Non-fiction":
            keywords = [
                "self help",
                "self-help",
                "nonfiction",
                "non-fiction",
                "psychology",
                "personal development",
                "religion",
                "spirituality",
                "mental health",
                "philosophy",
                "business",
                "biography",
                "memoir",
            ]
        else:  # "Fiction / Story"
            keywords = [
                "fiction",
                "novel",
                "fantasy",
                "romance",
                "thriller",
                "mystery",
                "young adult",
                "ya",
                "contemporary",
                "historical",
                "science fiction",
                "sci-fi",
                "horror",
            ]

        return any(k in g for k in keywords)

    focused = [r for r in filtered if matches_focus(r)]

    # If filters are too strict, fall back to less strict sets
    if len(focused) < 3:
        focused = filtered
    if len(focused) < 3:
        focused = raw_results

    # Sort by combination of similarity, rating, and popularity
    def score_key(r):
        rating = r["avg_rating"] if r["avg_rating"] is not None else 0.0
        num = r["num_ratings"] if r["num_ratings"] is not None else 0
        return (r["score"], float(rating), float(num))

    focused.sort(key=score_key, reverse=True)

    return focused[:top_k]


def generate_humanized_response(user_text: str, books: list[dict]) -> str:
    """
    Simple, rule-based explanation to keep things working without an LLM.
    Later you can replace this with a real LLM call (RAG).
    """
    if not user_text.strip() or not books:
        return ""

    lines = []
    lines.append(
        "Based on what you shared, it sounds like you are going through a lot emotionally. "
        "Here are some books that connect closely with what you described and may be meaningful for you:"
    )

    for i, b in enumerate(books, start=1):
        title = b["title"]
        author = b["author"]
        genres = b.get("genres", "")
        extra_bits = []

        if genres:
            extra_bits.append(f"Genres: {genres}")
        if b.get("avg_rating") is not None:
            extra_bits.append(f"Average rating: {b['avg_rating']}/5")
        if b.get("num_ratings") is not None:
            try:
                extra_bits.append(f"Based on {int(b['num_ratings'])} ratings")
            except Exception:
                pass

        extra_str = " | ".join(extra_bits)
        desc = b["description"]
        if len(desc) > 300:
            desc = desc[:300].rstrip() + "..."

        lines.append(f"\n{i}. {title} by {author}")
        if extra_str:
            lines.append(f"   {extra_str}")
        lines.append(f"   Why it might help you: {desc}")

    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="Semantic Book Recommender",
        page_icon="📚",
        layout="wide",
    )

    st.title("Semantic Book Recommender")
    st.write(
        "Tell me what you are going through or what kind of book you feel like reading. "
        "I will search for books whose themes, moods and descriptions match your feelings."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Search settings")

        top_k = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
        )

        min_rating = st.slider(
            "Minimum average rating",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
        )

        focus_mode = st.selectbox(
            "Recommendation focus",
            [
                "Anything",
                "Self-help / Non-fiction",
                "Fiction / Story",
            ],
        )

    # Main input
    user_text = st.text_area(
        "Describe your feelings, situation, or what you are in the mood to read:",
        height=180,
        placeholder=(
            "Examples:\n"
            "- I feel angry all the time and want a book that helps me manage it.\n"
            "- I feel lonely and want a warm, hopeful story about friendship.\n"
            "- I am burnt out and want something gentle and comforting to read."
        ),
    )

    search_clicked = st.button("Find books")

    if search_clicked:
        if not user_text.strip():
            st.warning("Please describe what you are going through or what you feel like reading.")
            return

        with st.spinner("Searching for books that match your feelings..."):
            results = semantic_search(
                user_text,
                top_k=top_k,
                min_rating=min_rating,
                focus_mode=focus_mode,
            )

        if not results:
            st.info("No matching books were found. Try rephrasing your description or relaxing the filters.")
            return

        explanation = generate_humanized_response(user_text, results)
        if explanation:
            st.subheader("Why these books might help")
            st.write(explanation)

        st.subheader("Recommended books")

        for b in results:
            with st.container():
                st.markdown(f"### {b['title']}  \n*by {b['author']}*")

                meta_parts = []
                if b.get("genres"):
                    meta_parts.append(f"**Genres:** {b['genres']}")
                if b.get("avg_rating") is not None:
                    meta_parts.append(f"**Average rating:** {b['avg_rating']}/5")
                if b.get("num_ratings") is not None:
                    try:
                        meta_parts.append(f"**Ratings:** {int(b['num_ratings'])}")
                    except Exception:
                        pass

                if meta_parts:
                    st.markdown(" · ".join(meta_parts))

                desc = b["description"]
                if len(desc) > 600:
                    desc_short = desc[:600].rstrip() + "..."
                else:
                    desc_short = desc
                st.write(desc_short)

                if b.get("url"):
                    st.markdown(f"[View more details]({b['url']})")

                st.divider()


if __name__ == "__main__":
    main()
