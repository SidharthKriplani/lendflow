from __future__ import annotations
"""
LendFlow Configuration
Loads from .env — copy .env.example to .env and fill in values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY:  str = os.getenv("LLM_API_KEY",  "lm-studio")
LLM_MODEL:    str = os.getenv("LLM_MODEL",     "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF")

# Detect whether we're using LM Studio or OpenAI
IS_LOCAL_LLM: bool = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR:    str = os.getenv("CHROMA_PERSIST_DIR",    "./data/chroma_db")
CHROMA_COLLECTION:     str = os.getenv("CHROMA_COLLECTION_NAME","lendflow_policy")

# ── Pipeline thresholds ───────────────────────────────────────────────────────
CONFIDENCE_HIGH: float = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.85"))
CONFIDENCE_LOW:  float = float(os.getenv("CONFIDENCE_LOW_THRESHOLD",  "0.65"))
MAX_POLICY_CHUNKS: int = int(os.getenv("MAX_POLICY_CHUNKS", "5"))
MAX_RETRIES:       int = int(os.getenv("MAX_RETRIES", "2"))

# ── Audit ─────────────────────────────────────────────────────────────────────
AUDIT_LOG_DIR: Path = Path(os.getenv("AUDIT_LOG_DIR", "./data/audit_logs"))
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Routing labels ────────────────────────────────────────────────────────────
ROUTING_APPROVE  = "APPROVE"
ROUTING_ESCALATE = "ESCALATE"
ROUTING_REJECT   = "REJECT"

# ── Document types ────────────────────────────────────────────────────────────
DOC_BANK_STATEMENT   = "bank_statement"
DOC_SALARY_SLIP      = "salary_slip"
DOC_KYC              = "kyc"
DOC_VEHICLE_REPORT   = "vehicle_report"
DOC_UNKNOWN          = "unknown"

ALL_DOC_TYPES = [DOC_BANK_STATEMENT, DOC_SALARY_SLIP, DOC_KYC, DOC_VEHICLE_REPORT]
