\
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def build_index(chunks):
    if not chunks:
        raise ValueError("No chunks available to index.")

    model = get_embedding_model()
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, k=4):
    if not query.strip():
        return []

    model = get_embedding_model()
    q = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(k, len(chunks))
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({
                "text": chunks[idx],
                "score": float(score),
                "chunk_id": int(idx),
            })
    return results
