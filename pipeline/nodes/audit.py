from __future__ import annotations
"""
Node 7: Audit Log
Writes a complete, immutable JSON audit record for every processed application.
The audit log is the regulatory paper trail — it must capture everything
needed to reproduce and explain the routing decision.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
from pipeline.state import LendFlowState


def _sanitize_for_json(obj):
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def audit_node(state: LendFlowState) -> dict:
    """
    Audit node: write complete JSON audit log.
    Input:  full pipeline state
    Output: audit_id
    """
    audit_id = f"AUDIT-{uuid.uuid4().hex[:12].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        # ── Identity ───────────────────────────────────────────────────────
        "audit_id":          audit_id,
        "application_id":    state.get("application_id", "UNKNOWN"),
        "timestamp_utc":     timestamp,
        "pipeline_version":  state.get("pipeline_version", "1.0.0"),

        # ── Document ───────────────────────────────────────────────────────
        "doc_type":          state.get("doc_type"),
        "pii_map_id":        state.get("pii_map_id"),        # pointer only — not the map
        "pii_entity_count":  state.get("pii_entity_count", 0),

        # ── Extraction ─────────────────────────────────────────────────────
        "extracted_fields":  _sanitize_for_json(state.get("extracted_fields", {})),
        "field_confidences": _sanitize_for_json(state.get("field_confidences", {})),

        # ── Policy ─────────────────────────────────────────────────────────
        "policy_chunks_count": len(state.get("policy_chunks", [])),
        "policy_citations":  [
            c.get("source") for c in state.get("policy_chunks", [])
        ],
        "policy_flags":      _sanitize_for_json(state.get("policy_flags", [])),

        # ── Routing ────────────────────────────────────────────────────────
        "uncertainty_score": state.get("uncertainty_score"),
        "uncertainty_band":  state.get("uncertainty_band"),
        "routing_decision":  state.get("routing_decision"),
        "reason_codes":      state.get("reason_codes", []),
        "human_review_required": state.get("human_review_required", False),

        # ── Human override (if any) ────────────────────────────────────────
        "human_override":    state.get("human_override"),

        # ── Token economics ────────────────────────────────────────────────
        "token_log":         _sanitize_for_json(state.get("token_log", {})),

        # ── Error ──────────────────────────────────────────────────────────
        "pipeline_error":    state.get("error"),
        "error_node":        state.get("error_node"),
    }

    # Write to disk
    log_path = Path(config.AUDIT_LOG_DIR) / f"{audit_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)

    return {"audit_id": audit_id}
