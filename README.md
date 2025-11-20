# 📚 Semantic Book Recommender (Emotion-Aware Recommendation System)

A powerful **AI-driven Semantic Book Recommendation System** that suggests books based on a user’s **emotional state, feelings, or reading intentions**.  
The system uses **Transformer embeddings**, **FAISS vector search**, and **LLM-based Retrieval-Augmented Generation (RAG)** to provide **accurate, empathetic, and context-aware** book suggestions.

Built using **Python**, **Streamlit**, **SentenceTransformers**, **FAISS**, and **OpenAI GPT-4o-mini**.

---

## 🚀 Features

### 🔍 Emotion-Aware Recommendations  
Users describe what they feel (e.g., *“I feel lonely and overwhelmed”*), and the system recommends semantically related books.

### 🧠 Semantic Embeddings  
Book descriptions and user text are converted to vector embeddings using SentenceTransformer MiniLM.

### ⚡ Fast Vector Search (FAISS)  
FAISS enables instant retrieval of the most meaningful recommendations.

### 🤖 LLM-Based Humanized Explanations  
A Large Language Model generates a warm, empathetic explanation of why each book matches the user’s emotional state.

### 🎛 Filters & Personalization  
Includes:
- Minimum rating filter  
- Focus Mode (Self-help / Non-fiction / Fiction / Anything)  
- Number of recommendations  

### 🌐 Streamlit Interface  
Clean, interactive UI for simple deployment and use.

---

## 🏗️ System Architecture

# User Input → Embedding Generation → FAISS Search → Filtering & Ranking → RAG (LLM) → Final Recommendations


### Components:
- **Data Preprocessing**
- **Semantic Embedding Generation**
- **FAISS Vector Index**
- **Semantic Search**
- **Retrieval-Augmented Generation (LLM)**
- **Streamlit Web App**

---

## 📦 Tech Stack

| Technology | Purpose |
|-----------|----------|
| Python | Core Language |
| Pandas / NumPy | Data Cleaning & Processing |
| SentenceTransformers | Text Embeddings |
| FAISS | Vector Similarity Search |
| OpenAI GPT-4o | Humanized Explanations |
| Streamlit | UI/Frontend |
| Dotenv | API Key Management |

---

## 📁 Project Structure

semantic-book-recommender/
│
├── app/
│ ├── data_prep.py # Preprocess dataset & create semantic_text
│ ├── embed.py # Build embeddings & FAISS index
│ ├── search.py # CLI search test tool
│
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Cleaned data, embeddings, FAISS index
│
├── streamlit_app.py # Main application file
├── requirements.txt
├── README.md
└── .env (ignored) # Stores OpenAI API key



