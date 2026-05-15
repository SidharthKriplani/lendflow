from __future__ import annotations
"""
Node 4: Policy Check
Deterministic rule evaluation on extracted fields + optional RAG retrieval for citations.
"""
import config
from pipeline.state import LendFlowState, PolicyFlag, TokenLog

# Employment types that indicate unverifiable income source
_UNVERIFIABLE_EMP_TYPES = {"UNKNOWN", "N/A", "", "CONTRACT", "GIG", "FREELANCE", "OTHER"}

# FOIR borderline margin — escalate if within this % of the limit
_FOIR_BORDERLINE_MARGIN = 0.05   # 5 percentage points

# Minimum monthly income thresholds (INR)
_MIN_INCOME_BANK_STMT  = 20_000   # bank statement
_MIN_INCOME_SALARY     = 22_000   # salary slip

# Vehicle condition grades
_HARD_FAIL_GRADES   = {"D", "E", "F"}    # reject
_BORDERLINE_GRADES  = {"B", "C"}         # escalate


def _evaluate_rules(fields: dict, doc_type: str) -> list[dict]:
    """
    Pure rule-based policy evaluation. No LLM.
    Returns list of PolicyFlag dicts.
    """
    if hasattr(fields, "model_dump"):
        fields = fields.model_dump()
    elif not isinstance(fields, dict):
        fields = {}

    flags: list[dict] = []

    # ── FOIR_LIMIT & FOIR_BORDERLINE ──────────────────────────────────────
    foir = fields.get("foir")
    emp  = str(fields.get("employment_type") or "SALARIED").upper()
    if foir is not None and doc_type == config.DOC_BANK_STATEMENT:
        limit = 0.50 if emp == "SELF_EMPLOYED" else 0.55
        hard_breach = float(foir) > limit
        borderline  = (not hard_breach) and (limit - float(foir)) <= _FOIR_BORDERLINE_MARGIN

        flags.append(PolicyFlag(
            rule_name="FOIR_LIMIT",
            triggered=hard_breach,
            detail=f"FOIR {foir:.1%} {'exceeds' if hard_breach else 'within'} {limit:.0%} limit ({emp})",
            policy_citation=None,
        ).model_dump())

        if borderline:
            flags.append(PolicyFlag(
                rule_name="FOIR_BORDERLINE",
                triggered=True,
                detail=f"FOIR {foir:.1%} is within {_FOIR_BORDERLINE_MARGIN:.0%} of {limit:.0%} limit — manual review recommended",
                policy_citation=None,
            ).model_dump())

    # ── RC_ENCUMBRANCE ─────────────────────────────────────────────────────
    rc = fields.get("rc_encumbrance")
    if rc is not None or doc_type == config.DOC_VEHICLE_REPORT:
        triggered = bool(rc)
        flags.append(PolicyFlag(
            rule_name="RC_ENCUMBRANCE",
            triggered=triggered,
            detail="Existing hypothecation found on RC" if triggered else "Vehicle title is clear",
            policy_citation=None,
        ).model_dump())

    # ── INSPECTION_FAILED (vehicle only) ──────────────────────────────────
    if doc_type == config.DOC_VEHICLE_REPORT:
        insp = fields.get("inspection_passed")
        grade = str(fields.get("condition_grade") or "").upper()

        # Hard fail: inspection not passed or failing grade
        insp_fail = insp is False or insp == "false"
        grade_fail = grade in _HARD_FAIL_GRADES
        triggered_hard = insp_fail or grade_fail

        flags.append(PolicyFlag(
            rule_name="INSPECTION_FAILED",
            triggered=triggered_hard,
            detail=(
                f"Inspection failed (grade {grade})" if grade_fail
                else "Inspection not passed" if insp_fail
                else "Inspection passed"
            ),
            policy_citation=None,
        ).model_dump())

        # Soft fail: borderline grade but inspection passed
        triggered_borderline = (not triggered_hard) and (grade in _BORDERLINE_GRADES)
        flags.append(PolicyFlag(
            rule_name="VEHICLE_CONDITION",
            triggered=triggered_borderline,
            detail=(
                f"Vehicle condition grade {grade} — borderline, manual review recommended"
                if triggered_borderline
                else f"Vehicle condition grade {grade or 'N/A'}"
            ),
            policy_citation=None,
        ).model_dump())

    # ── RED_FLAGS ──────────────────────────────────────────────────────────
    red = fields.get("suspicious_transactions") or fields.get("red_flags")
    if red is not None:
        triggered = bool(red)
        flags.append(PolicyFlag(
            rule_name="RED_FLAGS",
            triggered=triggered,
            detail="Suspicious transaction patterns detected" if triggered else "No suspicious transactions",
            policy_citation=None,
        ).model_dump())

    # ── KYC_COMPLETE ───────────────────────────────────────────────────────
    if doc_type == config.DOC_KYC:
        kyc = fields.get("kyc_complete")

        # Sub-field fallback: only if kyc_complete is not explicitly True
        # Avoids false-positives on clean KYC docs where LLM omits optional fields
        critical_kyc = ["id_type", "id_number", "dob", "address"]
        missing_subfields = [k for k in critical_kyc if fields.get(k) is None]
        # Only use sub-field check when kyc_complete is ambiguous (None) AND 3+ fields missing
        subfield_incomplete = (kyc is None) and (len(missing_subfields) >= 3)

        triggered = kyc is False or kyc == "false" or subfield_incomplete
        detail_parts = []
        if kyc is False:
            detail_parts.append("kyc_complete=False")
        if subfield_incomplete:
            detail_parts.append(f"missing fields: {missing_subfields[:3]}")
        flags.append(PolicyFlag(
            rule_name="KYC_COMPLETE",
            triggered=triggered,
            detail="; ".join(detail_parts) if detail_parts else "KYC documents verified",
            policy_citation=None,
        ).model_dump())

    # ── INCOME_VERIFIABLE ──────────────────────────────────────────────────
    if doc_type in (config.DOC_BANK_STATEMENT, config.DOC_SALARY_SLIP):
        income = fields.get("estimated_monthly_income") or fields.get("net_salary")
        emp_type = str(fields.get("employment_type") or "").upper()

        emp_unverifiable = emp_type in _UNVERIFIABLE_EMP_TYPES
        income_missing   = income is None or income == 0
        min_thresh = _MIN_INCOME_SALARY if doc_type == config.DOC_SALARY_SLIP else _MIN_INCOME_BANK_STMT
        income_low = income is not None and income < min_thresh

        triggered = income_missing or emp_unverifiable or income_low
        if income_missing:
            detail = "Monthly income could not be verified from documents"
        elif emp_unverifiable:
            detail = f"Employment type '{emp_type}' is non-standard — income source unverifiable"
        elif income_low:
            detail = f"Income ₹{income:,.0f}/mo is below minimum threshold ₹{min_thresh:,.0f}"
        else:
            detail = f"Income verified: ₹{income:,.0f} ({emp_type})"

        flags.append(PolicyFlag(
            rule_name="INCOME_VERIFIABLE",
            triggered=triggered,
            detail=detail,
            policy_citation=None,
        ).model_dump())

    return flags


def _build_policy_query(doc_type: str, fields: dict) -> str:
    parts = [f"Policy rules for {doc_type.replace('_', ' ')} application."]
    if foir := fields.get("foir"):
        parts.append(f"FOIR is {foir:.0%}.")
    if fields.get("rc_encumbrance"):
        parts.append("Vehicle has existing hypothecation.")
    if not fields.get("kyc_complete", True):
        parts.append("KYC may be incomplete.")
    if fields.get("suspicious_transactions"):
        parts.append("Transaction red flags detected.")
    return " ".join(parts)


def policy_node(state: LendFlowState) -> dict:
    if state.get("error"):
        return {}

    fields   = state.get("extracted_fields", {})
    doc_type = state.get("doc_type", config.DOC_UNKNOWN)
    existing_token_log = state.get("token_log", {})

    # Deterministic rule evaluation
    policy_flags = _evaluate_rules(fields, doc_type)

    # RAG retrieval for policy citations (best-effort)
    policy_chunks: list = []
    try:
        from rag.retriever import retrieve_policy_chunks
        q_fields = fields if isinstance(fields, dict) else fields.model_dump()
        query = _build_policy_query(doc_type, q_fields)
        policy_chunks = retrieve_policy_chunks(query, top_k=config.MAX_POLICY_CHUNKS)
    except Exception:
        pass

    tl = TokenLog(**existing_token_log) if existing_token_log else TokenLog()
    tl.update_totals()

    return {
        "policy_chunks": policy_chunks,
        "policy_flags":  policy_flags,
        "token_log":     tl.model_dump(),
    }
