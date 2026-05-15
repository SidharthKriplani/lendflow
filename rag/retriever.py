"""
LendFlow RAG Retriever
Hybrid search: BM25 (sparse) + ChromaDB dense embeddings → cross-encoder rerank.
Falls back gracefully if ChromaDB is not indexed yet.
"""
from __future__ import annotations
import os
from typing import Optional

import config

# ── Lazy imports — only load heavy models when first used ─────────────────────
_chroma_client  = None
_collection     = None
_embed_model    = None
_reranker       = None
_bm25_index     = None
_bm25_corpus    = None   # list of {"text": ..., "source": ...}


def _get_chroma():
    global _chroma_client, _collection
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        try:
            _collection = _chroma_client.get_collection(config.CHROMA_COLLECTION)
        except Exception:
            _collection = None
    return _collection


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker = None
    return _reranker


def _load_bm25_corpus() -> None:
    """Load BM25 corpus from disk (written by rag/indexer.py)."""
    global _bm25_corpus
    if _bm25_corpus:
        return
    import json
    corpus_path = os.path.join(config.CHROMA_PERSIST_DIR, "bm25_corpus.json")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            _bm25_corpus = json.load(f)


def _get_bm25():
    global _bm25_index, _bm25_corpus
    _load_bm25_corpus()
    if _bm25_index is None and _bm25_corpus:
        from rank_bm25 import BM25Okapi
        tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index


def dense_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Retrieve top_k chunks using ChromaDB dense search."""
    collection = _get_chroma()
    if collection is None:
        return []
    try:
        embed_model = _get_embed_model()
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            chunks.append({
                "text":   doc,
                "source": meta.get("source", "policy"),
                "score":  round(1.0 - dist, 4),  # cosine distance → similarity
                "method": "dense",
            })
        return chunks
    except Exception as e:
        return []


def bm25_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Retrieve top_k chunks using BM25 sparse search."""
    bm25 = _get_bm25()
    if bm25 is None or not _bm25_corpus:
        return []
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**_bm25_corpus[i], "score": round(float(scores[i]), 4), "method": "bm25"}
        for i in top_indices if scores[i] > 0
    ]


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidate chunks using a cross-encoder."""
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_k]:
        chunk = dict(chunk)
        chunk["rerank_score"] = round(float(score), 4)
        result.append(chunk)
    return result


def retrieve_policy_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: dense + BM25 → deduplicate → rerank → top_k.
    Returns list of {text, source, score, method}.
    """
    dense_results = dense_retrieve(query, top_k=top_k * 2)
    bm25_results  = bm25_retrieve(query,  top_k=top_k * 2)

    # Deduplicate by text
    seen_texts: set[str] = set()
    merged: list[dict] = []
    for chunk in dense_results + bm25_results:
        key = chunk["text"][:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(chunk)

    if not merged:
        return []

    return rerank(query, merged, top_k=top_k)
