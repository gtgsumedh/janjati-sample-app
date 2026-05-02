import logging
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(SAVE_DIR, "heroes.index")
DATA_MAP_PATH = os.path.join(SAVE_DIR, "data_map.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

logger.info("Loading SentenceTransformer model (this may take a few seconds)...")
embedding_model = SentenceTransformer(MODEL_NAME)
logger.info("SentenceTransformer model loaded successfully.")

fallback_documents = [
    "Python is a programming language used for AI and web development.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "JavaScript is mainly used for building interactive web applications.",
    "Birsa Munda was an Indian tribal independence activist and folk hero."
]


def load_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DATA_MAP_PATH):
        logger.warning("Vector store not found. Falling back to default knowledge snippets.")
        return None, fallback_documents, []

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(DATA_MAP_PATH, "rb") as f:
            data_map = pickle.load(f)
        chunks = data_map.get("chunks", [])
        metadatas = data_map.get("metadatas", [])
        return index, chunks, metadatas
    except Exception as exc:
        logger.error(f"Unable to load vector store: {exc}")
        return None, fallback_documents, []


index, chunks, metadatas = load_vector_store()


def find_best_context(query: str, top_k: int = 3):
    logger.info("Vector DB: retrieving best context for query...")

    if index is None or not chunks:
        logger.info("Using fallback documents for best context.")
        return fallback_documents[0]

    query_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        name = metadata.get("name", "Unknown")
        section = metadata.get("section", "General")
        results.append(f"{section} ({name}): {chunks[idx]}")

    if not results:
        logger.warning("No search results found; using fallback documents.")
        return fallback_documents[0]

    return "\n\n".join(results)
