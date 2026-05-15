"""
LendFlow RAG Indexer
Chunks policy documents → embeds → stores in ChromaDB + saves BM25 corpus JSON.
Run once before starting the pipeline: python rag/indexer.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

POLICY_DOCS_DIR = Path(__file__).parent / "policy_docs"
CHUNK_SIZE      = 300   # ~words per chunk
CHUNK_OVERLAP   = 50    # ~words overlap between chunks


def load_policy_docs() -> list[dict]:
    """Load all .txt files from policy_docs/."""
    docs = []
    for fpath in sorted(POLICY_DOCS_DIR.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        docs.append({"source": fpath.stem, "text": text})
        print(f"  Loaded: {fpath.name} ({len(text):,} chars)")
    return docs


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Sliding-window word-level chunking."""
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap
    i      = 0
    idx    = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text  = " ".join(chunk_words)
        chunks.append({
            "id":     f"{source}_chunk_{idx:03d}",
            "text":   chunk_text,
            "source": source,
        })
        i   += step
        idx += 1
        if i + chunk_size > len(words) and i < len(words):
            # Final partial chunk
            chunk_words = words[i:]
            if len(chunk_words) > 20:   # skip tiny trailing scraps
                chunks.append({
                    "id":     f"{source}_chunk_{idx:03d}",
                    "text":   " ".join(chunk_words),
                    "source": source,
                })
            break
    return chunks


def build_chroma_index(chunks: list[dict]) -> None:
    """Embed chunks and store in ChromaDB."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    client     = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # Drop existing collection to allow re-indexing
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
        print("  Dropped existing ChromaDB collection.")
    except Exception:
        pass

    collection = client.create_collection(config.CHROMA_COLLECTION)
    model      = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"  Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}...")
    texts      = [c["text"]   for c in chunks]
    ids        = [c["id"]     for c in chunks]
    metadatas  = [{"source": c["source"]} for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"  ChromaDB: indexed {collection.count()} chunks.")


def save_bm25_corpus(chunks: list[dict]) -> None:
    """Save chunks as JSON for BM25 index (loaded by retriever at startup)."""
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    corpus_path = os.path.join(config.CHROMA_PERSIST_DIR, "bm25_corpus.json")
    corpus = [{"text": c["text"], "source": c["source"]} for c in chunks]
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  BM25 corpus saved: {corpus_path} ({len(corpus)} entries)")


def main() -> None:
    print("\n=== LendFlow RAG Indexer ===\n")

    print("[1/4] Loading policy documents...")
    docs = load_policy_docs()
    if not docs:
        print("ERROR: No policy docs found in", POLICY_DOCS_DIR)
        sys.exit(1)

    print(f"\n[2/4] Chunking {len(docs)} documents...")
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["source"])
        print(f"  {doc['source']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"  Total chunks: {len(all_chunks)}")

    print("\n[3/4] Building ChromaDB dense index...")
    build_chroma_index(all_chunks)

    print("\n[4/4] Saving BM25 corpus...")
    save_bm25_corpus(all_chunks)

    print("\n✅ Indexing complete.")
    print(f"   ChromaDB: {config.CHROMA_PERSIST_DIR}")
    print(f"   Chunks indexed: {len(all_chunks)}")
    print("\nNow start the pipeline — RAG retrieval is ready.\n")


if __name__ == "__main__":
    main()
