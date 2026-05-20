import hashlib
import os

import numpy as np

_MODEL = None
_MODEL_FAILED = False
EMBEDDING_DIMENSION = 384


def _hash_embedding(text):
    vector = np.zeros(EMBEDDING_DIMENSION, dtype="float32")
    words = [word for word in str(text).lower().split() if word]

    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _load_model():
    global _MODEL, _MODEL_FAILED

    if _MODEL is not None:
        return _MODEL
    if _MODEL_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("NEURODOCS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(model_name, local_files_only=os.getenv("NEURODOCS_OFFLINE", "0") == "1")
        return _MODEL
    except Exception:
        _MODEL_FAILED = True
        return None


def encode_texts(texts):
    values = [str(text or "") for text in texts]
    model = _load_model()

    if model is not None:
        return np.array(model.encode(values), dtype="float32")

    return np.array([_hash_embedding(text) for text in values], dtype="float32")


def create_embeddings(chunks):
    texts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in chunks]
    return encode_texts(texts)


class LazyEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        embeddings = encode_texts(texts)
        return embeddings if convert_to_numpy else embeddings.tolist()


model = LazyEmbeddingModel()
