import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load the free local embedding model
logger.info("Loading SentenceTransformer model (this may take a few seconds)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer model loaded successfully.")

# --- KNOWLEDGE BASE ---
documents = [
    "Python is a programming language used for AI and web development.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "JavaScript is mainly used for building interactive web applications.",
    "Birsa Munda was an Indian tribal independence activist and folk hero."
]
doc_embeddings = [embedding_model.encode(doc) for doc in documents]

def find_best_context(query: str):
    logger.info(f"Team 1 (Vector DB): Embedding and searching context for query...")
    query_emb = embedding_model.encode(query)
    similarities = cosine_similarity([query_emb], doc_embeddings)[0]
    best_index = np.argmax(similarities)
    return documents[best_index]
