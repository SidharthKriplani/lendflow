from __future__ import annotations
"""
Node 1: Intake
Assigns application_id, classifies document type from raw text.
Runs before any LLM call — purely deterministic.
"""
import re
import uuid
from pipeline.state import LendFlowState
import config


# ── Keyword signatures for document type classification ───────────────────────
_SIGNATURES: dict[str, list[str]] = {
    config.DOC_BANK_STATEMENT: [
        "account statement", "bank statement", "opening balance", "closing balance",
        "transaction date", "debit", "credit", "ifsc", "branch",
    ],
    config.DOC_SALARY_SLIP: [
        "salary slip", "pay slip", "payslip", "gross salary", "net salary",
        "basic pay", "hra", "provident fund", "tds deducted", "employee id",
        "designation", "department",
    ],
    config.DOC_KYC: [
        "aadhaar", "pan card", "date of birth", "father's name", "address proof",
        "kyc", "know your customer", "identity proof", "passport", "driving licence",
        "[aadhaar_", "[pan_",  # placeholder patterns after PII redaction
    ],
    config.DOC_VEHICLE_REPORT: [
        "vehicle inspection", "inspection report", "profecto", "odometer",
        "chassis number", "engine number", "registration certificate",
        "rc number", "vehicle condition", "assessed value", "hypothecation",
        "make and model",
    ],
}


def classify_doc_type(text: str) -> str:
    """Rule-based classification. Returns the doc type with highest keyword hits."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for doc_type, keywords in _SIGNATURES.items():
        scores[doc_type] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else config.DOC_UNKNOWN


def intake_node(state: LendFlowState) -> dict:
    """
    Intake node: assign application_id, classify document type.
    Input:  state['raw_text']
    Output: application_id, doc_type, pipeline_version
    """
    raw_text = state.get("raw_text", "")
    if not raw_text.strip():
        return {
            "error": "Empty document received",
            "error_node": "intake",
            "application_id": str(uuid.uuid4()),
            "doc_type": config.DOC_UNKNOWN,
            "pipeline_version": "1.0.0",
        }

    doc_type = classify_doc_type(raw_text)
    application_id = state.get("application_id") or f"APP-{uuid.uuid4().hex[:8].upper()}"

    return {
        "application_id": application_id,
        "doc_type": doc_type,
        "pipeline_version": "1.0.0",
        "error": None,
        "error_node": None,
    }
