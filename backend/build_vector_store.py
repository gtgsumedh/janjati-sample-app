import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "knowledge_data")
SAVE_DIR = os.path.join(BASE_DIR, "vector_store")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge_data():
    chunks = []
    metadatas = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: knowledge directory not found: {DATA_DIR}")
        return chunks, metadatas

    for hero_folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, hero_folder)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            hero_entries = json.load(f)

        for entry in hero_entries:
            text = entry.get("text", "").strip()
            if not text:
                continue

            chunks.append(text)
            metadata = entry.get("metadata", {}).copy()
            metadata["section"] = entry.get("section", "General")
            metadatas.append(metadata)

    return chunks, metadatas


def build_vector_store():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    chunks, metadatas = load_knowledge_data()
    if not chunks:
        print("No knowledge data found. Add JSON files under backend/knowledge_data.")
        return

    print(f"Embedding {len(chunks)} knowledge chunks...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(SAVE_DIR, "heroes.index"))
    with open(os.path.join(SAVE_DIR, "data_map.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)

    print("Vector store built successfully in backend/vector_store.")


if __name__ == "__main__":
    build_vector_store()
