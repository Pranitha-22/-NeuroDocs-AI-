# app/rag/vector_store.py

import faiss
import numpy as np

from app.rag.embeddings import encode_texts


def _chunk_texts(chunks):
    return [item.get("text", "") if isinstance(item, dict) else str(item) for item in chunks]


def create_faiss_index(chunks):
    if isinstance(chunks, np.ndarray):
        embeddings = np.array(chunks, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embedding array must be a non-empty 2D matrix.")
    else:
        texts = _chunk_texts(chunks)
        if not texts:
            raise ValueError("Cannot create a FAISS index without document chunks.")
        embeddings = np.array(encode_texts(texts), dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index, path):
    faiss.write_index(index, str(path))


def load_faiss_index(path):
    return faiss.read_index(str(path))


def search_similar_chunks(query, index, chunks=None, top_k=3):
    if isinstance(query, np.ndarray):
        query_embedding = np.array([query], dtype="float32") if query.ndim == 1 else np.array(query, dtype="float32")
    elif isinstance(query, list) and query and isinstance(query[0], (int, float, np.floating)):
        query_embedding = np.array([query], dtype="float32")
    else:
        query_embedding = np.array(encode_texts([query]), dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    if chunks is None:
        return [int(i) for i in indices[0] if i >= 0]

    retrieved_chunks = []
    for distance, chunk_index in zip(distances[0], indices[0]):
        if 0 <= chunk_index < len(chunks):
            item = chunks[chunk_index]
            if isinstance(item, dict):
                result = item.copy()
                result["distance"] = float(distance)
                retrieved_chunks.append(result)
            else:
                retrieved_chunks.append({
                    "id": f"chunk-{chunk_index}",
                    "page": None,
                    "chunk_index": int(chunk_index),
                    "text": str(item),
                    "distance": float(distance),
                })

    return retrieved_chunks
