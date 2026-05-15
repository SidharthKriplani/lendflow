from __future__ import annotations
"""
Node 6: Routing
Maps uncertainty_band + policy flags → APPROVE / ESCALATE / REJECT.
Generates human-readable reason codes.
"""
import config
from pipeline.state import LendFlowState

# ── Reason code templates ──────────────────────────────────────────────────────
_REASON_TEMPLATES = {
    "FOIR_LIMIT":       "Fixed obligations exceed the maximum FOIR threshold ({foir:.0%} vs policy limit)",
    "KYC_COMPLETE":     "KYC documentation is incomplete or unverified",
    "RC_ENCUMBRANCE":   "Vehicle has existing hypothecation — clear title required before disbursement",
    "INCOME_VERIFIABLE":"Monthly income could not be verified from the provided documents",
    "RED_FLAGS":        "Suspicious transaction patterns detected in bank statement",
    "LOW_CONFIDENCE":   "Insufficient confidence in extracted fields — manual verification required",
    "MISSING_CRITICAL": "One or more critical fields could not be extracted: {fields}",
}


def _format_reason(template_key: str, fields: dict) -> str:
    template = _REASON_TEMPLATES.get(template_key, template_key)
    try:
        return template.format(**fields)
    except (KeyError, ValueError):
        return template


def route_node(state: LendFlowState) -> dict:
    """
    Routing node.
    Input:  uncertainty_band, policy_flags, field_confidences, extracted_fields
    Output: routing_decision, reason_codes, human_review_required
    """
    if state.get("error"):
        return {
            "routing_decision":    config.ROUTING_ESCALATE,
            "reason_codes":        ["Pipeline error — routing to human review"],
            "human_review_required": True,
        }

    band           = state.get("uncertainty_band", "LOW")
    policy_flags   = state.get("policy_flags", [])
    fields         = state.get("extracted_fields", {})
    fc_obj         = state.get("field_confidences", {})
    doc_type       = state.get("doc_type", config.DOC_UNKNOWN)

    # Handle FieldConfidences Pydantic model or plain dict
    if hasattr(fc_obj, "model_dump"):
        fc_dict = fc_obj.model_dump()
    elif hasattr(fc_obj, "scores"):
        fc_dict = {"scores": fc_obj.scores}
    else:
        fc_dict = fc_obj or {}
    scores = fc_dict.get("scores", {})

    # Handle PolicyFlag Pydantic models or plain dicts
    def _flag_triggered(f) -> bool:
        return f.triggered if hasattr(f, "triggered") else bool(f.get("triggered"))
    def _flag_rule(f) -> str:
        return f.rule_name if hasattr(f, "rule_name") else f["rule_name"]

    triggered_flags = [f for f in policy_flags if _flag_triggered(f)]
    triggered_rules = {_flag_rule(f) for f in triggered_flags}

    # Normalise extracted_fields to dict for attribute access
    if hasattr(fields, "model_dump"):
        fields_dict = fields.model_dump()
    elif isinstance(fields, dict):
        fields_dict = fields
    else:
        fields_dict = {}

    # reason_codes uses rule names (machine-readable); reason_messages has human text
    reason_codes: list[str] = []

    # ── Hard-reject conditions ─────────────────────────────────────────────
    hard_rejects = {"RC_ENCUMBRANCE", "RED_FLAGS", "INSPECTION_FAILED"}
    if triggered_rules & hard_rejects:
        for rule in sorted(triggered_rules & hard_rejects):
            reason_codes.append(rule)
        return {
            "routing_decision":    config.ROUTING_REJECT,
            "reason_codes":        reason_codes[:3],
            "human_review_required": False,
        }

    # ── FOIR violation → REJECT ────────────────────────────────────────────
    if "FOIR_LIMIT" in triggered_rules:
        foir  = fields_dict.get("foir", 0)
        emp   = fields_dict.get("employment_type", "SALARIED")
        limit = 0.50 if emp == "SELF_EMPLOYED" else 0.55
        if foir and foir > limit:
            reason_codes.append("FOIR_EXCEEDED")
            return {
                "routing_decision":    config.ROUTING_REJECT,
                "reason_codes":        reason_codes[:3],
                "human_review_required": False,
            }

    # ── Escalation conditions ──────────────────────────────────────────────
    # LOW band → always escalate (truly uncertain extraction)
    # MEDIUM band → escalate only if soft flags triggered or critical fields missing
    # HIGH band → proceed to APPROVE below
    # New soft flags: FOIR_BORDERLINE, VEHICLE_CONDITION also force escalation
    soft_flags = {"KYC_COMPLETE", "INCOME_VERIFIABLE", "FOIR_BORDERLINE", "VEHICLE_CONDITION"}
    soft_triggered = triggered_rules & soft_flags

    from pipeline.nodes.confidence import _CRITICAL_FIELDS
    critical = _CRITICAL_FIELDS.get(doc_type, [])
    missing_critical = [f for f in critical if scores.get(f, 0) < 0.5]

    should_escalate = (
        band == "LOW"
        or (band == "MEDIUM" and (soft_triggered or missing_critical))
    )

    if should_escalate:
        if band == "LOW":
            reason_codes.append("LOW_CONFIDENCE")
        elif band == "MEDIUM":
            reason_codes.append("MEDIUM_CONFIDENCE")
        for rule in sorted(soft_triggered):
            reason_codes.append(rule)
        if missing_critical:
            reason_codes.append("MISSING_CRITICAL_FIELDS")

        return {
            "routing_decision":    config.ROUTING_ESCALATE,
            "reason_codes":        reason_codes[:3] or ["Further verification required"],
            "human_review_required": True,
        }

    # ── Approve ────────────────────────────────────────────────────────────
    return {
        "routing_decision":    config.ROUTING_APPROVE,
        "reason_codes":        ["All required fields verified with high confidence"],
        "human_review_required": False,
    }
