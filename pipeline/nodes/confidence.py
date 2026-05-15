from __future__ import annotations
"""
Node 5: Confidence Evaluation
Aggregates field-level confidence + policy flag results into an
application-level uncertainty score and band (HIGH / MEDIUM / LOW confidence).
"""
import config
from pipeline.state import LendFlowState

# Critical fields that must be high-confidence for a clean APPROVE
_CRITICAL_FIELDS = {
    config.DOC_BANK_STATEMENT: ["estimated_monthly_income", "foir", "employment_type"],
    config.DOC_SALARY_SLIP:    ["net_salary", "employment_type"],
    config.DOC_KYC:            ["kyc_complete"],
    config.DOC_VEHICLE_REPORT: ["assessed_value", "rc_encumbrance", "inspection_passed"],
}

# Policy rules whose trigger → automatic uncertainty escalation
_HARD_BLOCK_RULES = {"RC_ENCUMBRANCE", "RED_FLAGS"}
_SOFT_ESCALATE_RULES = {"FOIR_LIMIT", "KYC_COMPLETE", "INCOME_VERIFIABLE"}


def confidence_node(state: LendFlowState) -> dict:
    """
    Confidence evaluation node.
    Input:  field_confidences, policy_flags, doc_type
    Output: uncertainty_score, uncertainty_band
    """
    if state.get("error"):
        return {}

    doc_type       = state.get("doc_type", config.DOC_UNKNOWN)
    fc_dict        = state.get("field_confidences", {})
    policy_flags   = state.get("policy_flags", [])

    # Normalize field_confidences (may be Pydantic model or plain dict)
    if hasattr(fc_dict, "model_dump"):
        fc_dict = fc_dict.model_dump()
    scores: dict[str, float] = fc_dict.get("scores", {}) if isinstance(fc_dict, dict) else {}
    overall: float = fc_dict.get("overall", 0.5) if isinstance(fc_dict, dict) else 0.5

    # ── Critical field penalty ─────────────────────────────────────────────
    critical = _CRITICAL_FIELDS.get(doc_type, [])
    critical_confidences = [scores.get(f, 0.3) for f in critical]
    critical_avg = (
        sum(critical_confidences) / len(critical_confidences)
        if critical_confidences else overall
    )

    # ── Policy flag penalty ────────────────────────────────────────────────
    def _triggered(f) -> bool:
        return f.triggered if hasattr(f, "triggered") else bool(f.get("triggered"))
    def _rule(f) -> str:
        return f.rule_name if hasattr(f, "rule_name") else f["rule_name"]
    triggered_rules = {_rule(f) for f in policy_flags if _triggered(f)}
    has_hard_block  = bool(triggered_rules & _HARD_BLOCK_RULES)
    has_soft_flag   = bool(triggered_rules & _SOFT_ESCALATE_RULES)

    # ── Compute uncertainty score (0 = certain, 1 = highly uncertain) ─────
    # Base: invert confidence → uncertainty
    base_uncertainty = 1.0 - (0.6 * critical_avg + 0.4 * overall)

    if has_hard_block:
        base_uncertainty = max(base_uncertainty, 0.85)  # force HIGH uncertainty
    elif has_soft_flag:
        base_uncertainty = max(base_uncertainty, 0.40)  # at least MEDIUM

    uncertainty_score = round(min(1.0, max(0.0, base_uncertainty)), 3)

    # ── Map to band using config thresholds ────────────────────────────────
    # CONFIDENCE thresholds: HIGH ≥ 0.85, MEDIUM 0.65-0.85, LOW < 0.65
    # Uncertainty is the inverse: LOW uncertainty → HIGH confidence
    confidence_score = 1.0 - uncertainty_score
    if confidence_score >= config.CONFIDENCE_HIGH:
        band = "HIGH"
    elif confidence_score >= config.CONFIDENCE_LOW:
        band = "MEDIUM"
    else:
        band = "LOW"

    return {
        "uncertainty_score": uncertainty_score,
        "uncertainty_band":  band,
    }
