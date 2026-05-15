from __future__ import annotations
"""
LendFlow — Extraction Result Cache
SHA-256 hash of redacted document text → cached extracted_fields + field_confidences.
Falls back to in-memory dict when Redis is unavailable (dev/local mode).

Redis is optional: if REDIS_URL is not set or Redis is unreachable,
the cache silently degrades to a process-local dict with no TTL.
"""
import hashlib
import json
import os
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_URL   = os.environ.get("REDIS_URL", "")          # e.g. redis://localhost:6379/0
CACHE_TTL   = int(os.environ.get("CACHE_TTL_SECONDS", 86400))  # 24 hours default
CACHE_PREFIX = "lendflow:extract:"

# ── Backend ───────────────────────────────────────────────────────────────────
_redis_client = None
_memory_cache: dict[str, str] = {}   # fallback: key → json string


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=0.5)
        _redis_client.ping()   # fail fast if unreachable
        return _redis_client
    except Exception:
        _redis_client = None
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def make_cache_key(redacted_text: str, doc_type: str) -> str:
    """SHA-256 of (doc_type + redacted_text) → deterministic cache key."""
    payload = f"{doc_type}::{redacted_text}".encode("utf-8")
    digest  = hashlib.sha256(payload).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


def get_cached(redacted_text: str, doc_type: str) -> Optional[dict]:
    """
    Return cached extraction result dict, or None on miss.
    Result dict has keys: extracted_fields, field_confidences.
    """
    key = make_cache_key(redacted_text, doc_type)
    r   = _get_redis()

    try:
        raw = r.get(key) if r else _memory_cache.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def set_cached(redacted_text: str, doc_type: str,
               extracted_fields: dict, field_confidences: dict) -> None:
    """Store extraction result in cache."""
    key     = make_cache_key(redacted_text, doc_type)
    payload = json.dumps({
        "extracted_fields":  extracted_fields,
        "field_confidences": field_confidences,
        "_cache_hit": True,
    })
    r = _get_redis()
    try:
        if r:
            r.setex(key, CACHE_TTL, payload)
        else:
            _memory_cache[key] = payload   # in-memory fallback, no TTL
    except Exception:
        pass


def cache_stats() -> dict:
    """Return cache backend info — useful for /health endpoint."""
    r = _get_redis()
    backend = "redis" if r else "memory"
    size    = len(_memory_cache) if not r else None
    return {"backend": backend, "memory_entries": size, "ttl_seconds": CACHE_TTL}
