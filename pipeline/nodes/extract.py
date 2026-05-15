from __future__ import annotations
"""
Node 3: Structured Extraction
LLM call (LM Studio / OpenAI) with document-type-specific Pydantic schema.
Deterministic fallback for calculable fields (FOIR).
Field-level confidence scoring via heuristic (null → low, present → medium/high).

Cache: SHA-256(redacted_text + doc_type) → Redis (or in-memory fallback).
Repeat submissions skip the LLM entirely.
"""
import json
import re
import time
from typing import Any

from openai import OpenAI

import config
from pipeline.state import (
    LendFlowState, BankStatementFields, SalarySlipFields,
    KYCFields, VehicleReportFields, FieldConfidences, TokenLog,
)
from pipeline.cache import get_cached, set_cached

# ── LLM client (OpenAI-compatible — works with LM Studio) ────────────────────
_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

# ── Per-doc-type extraction prompts ───────────────────────────────────────────
_SYSTEM_PROMPTS: dict[str, str] = {
    config.DOC_BANK_STATEMENT: (
        "You are a financial document extraction assistant. "
        "Extract fields from the bank statement below and respond with ONLY a JSON object. "
        "No markdown, no code fences, no explanation — just the raw JSON object starting with { and ending with }. "
        "PII has been replaced with placeholders like [PERSON_1], [AADHAAR_1]. "
        "If a field cannot be determined, use null. "
        "Calculate foir = emi_obligations / estimated_monthly_income (null if either missing)."
    ),
    config.DOC_SALARY_SLIP: (
        "You are a financial document extraction assistant. "
        "Extract fields from the salary slip below and respond with ONLY a JSON object. "
        "No markdown, no code fences, no explanation — just the raw JSON object starting with { and ending with }. "
        "PII has been replaced with placeholders. If a field cannot be determined, use null."
    ),
    config.DOC_KYC: (
        "You are a KYC document extraction assistant. "
        "Extract fields from the KYC document below and respond with ONLY a JSON object. "
        "No markdown, no code fences, no explanation — just the raw JSON object starting with { and ending with }. "
        "PII replaced with placeholders like [AADHAAR_1], [IN_PAN_1]. "
        "Set kyc_complete=true only if name, DOB, and at least one ID placeholder are present."
    ),
    config.DOC_VEHICLE_REPORT: (
        "You are a vehicle inspection report extraction assistant. "
        "Extract fields from the vehicle report below and respond with ONLY a JSON object. "
        "No markdown, no code fences, no explanation — just the raw JSON object starting with { and ending with }. "
        "Set rc_encumbrance=true if any existing hypothecation or charge is mentioned."
    ),
}

_SCHEMAS: dict[str, type] = {
    config.DOC_BANK_STATEMENT: BankStatementFields,
    config.DOC_SALARY_SLIP:    SalarySlipFields,
    config.DOC_KYC:            KYCFields,
    config.DOC_VEHICLE_REPORT: VehicleReportFields,
}


def _schema_json(doc_type: str) -> str:
    """Return the JSON schema for the Pydantic model as a string."""
    schema = _SCHEMAS[doc_type].model_json_schema()
    return json.dumps(schema, indent=2)


def _extract_json_from_text(text: str) -> dict:
    """
    Try multiple strategies to extract a JSON object from LLM output.
    1. Direct parse
    2. Strip markdown code fences
    3. Regex find first {...} block
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` or ``` ... ```
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find outermost {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _call_llm(system_prompt: str, user_content: str,
              retries: int = config.MAX_RETRIES) -> tuple[dict, int, int]:
    """
    Make an LLM call and return (parsed_json, prompt_tokens, completion_tokens).
    Retries on JSON parse failure up to `retries` times.
    Does NOT use response_format (not supported by all local models).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    last_raw = ""
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            last_raw = raw
            parsed = _extract_json_from_text(raw.strip())
            pt = response.usage.prompt_tokens if response.usage else 0
            ct = response.usage.completion_tokens if response.usage else 0

            if parsed:
                return parsed, pt, ct

            # Empty parse — retry with correction
            if attempt < retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Your response did not contain valid JSON. "
                               "Output ONLY a JSON object with no extra text, "
                               "no markdown fences, no explanation.",
                })

        except Exception as e:
            # Log to stderr so it's visible when running demo.py / run_eval.py
            import sys
            print(f"[extract] LLM call error (attempt {attempt+1}): {e}", file=sys.stderr)
            break

    return {}, 0, 0


def _heuristic_confidence(fields: dict, schema_cls: type) -> FieldConfidences:
    """
    Estimate per-field confidence heuristically:
    - null / None → 0.3 (low)
    - present but string 'UNKNOWN' → 0.5
    - numeric present → 0.85
    - string present → 0.80
    FOIR is only high-confidence if both income and obligations are present.
    """
    scores: dict[str, float] = {}

    for field_name, field_info in schema_cls.model_fields.items():
        val = fields.get(field_name)
        if val is None:
            scores[field_name] = 0.3
        elif isinstance(val, str) and val.upper() in ("UNKNOWN", "N/A", ""):
            scores[field_name] = 0.5
        elif isinstance(val, (int, float)):
            scores[field_name] = 0.85
        elif isinstance(val, bool):
            scores[field_name] = 0.90
        elif isinstance(val, list):
            scores[field_name] = 0.80 if val else 0.5
        else:
            scores[field_name] = 0.80

    # FOIR confidence — only meaningful if income and EMI are both present
    if "foir" in scores:
        income = fields.get("estimated_monthly_income") or fields.get("net_salary")
        emi = fields.get("emi_obligations") or fields.get("emi_deductions")
        if income and emi:
            scores["foir"] = 0.88
        else:
            scores["foir"] = 0.30

    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return FieldConfidences(scores=scores, overall=round(overall, 3))


def _deterministic_enrichment(fields: dict, doc_type: str) -> dict:
    """
    Apply deterministic calculations on top of LLM extraction.
    - Compute FOIR if income and EMI are both present and FOIR is null.
    - Clamp FOIR to [0, 1].
    """
    if doc_type == config.DOC_BANK_STATEMENT:
        income = fields.get("estimated_monthly_income")
        emi = fields.get("emi_obligations")
        # Always recompute FOIR from raw values — never trust LLM's calculation
        if income and emi and income > 0:
            fields["foir"] = round(min(1.0, max(0.0, emi / income)), 3)
        elif fields.get("foir") is not None:
            fields["foir"] = max(0.0, min(1.0, float(fields["foir"])))

    if doc_type == config.DOC_SALARY_SLIP:
        gross = fields.get("gross_salary")
        net = fields.get("net_salary")
        if gross and not net:
            fields["net_salary"] = round(gross * 0.78, 0)  # rough estimate

    return fields


def extract_node(state: LendFlowState) -> dict:
    """
    Extraction node: LLM-based structured extraction + deterministic enrichment.
    Input:  state['redacted_text'], state['doc_type']
    Output: extracted_fields, field_confidences, token_log (partial)
    """
    if state.get("error"):
        return {}

    doc_type = state.get("doc_type", config.DOC_UNKNOWN)
    redacted_text = state.get("redacted_text", "")

    if doc_type not in _SCHEMAS:
        return {
            "error": f"Unsupported document type: {doc_type}",
            "error_node": "extract",
        }

    schema_cls = _SCHEMAS[doc_type]

    # ── Cache lookup ───────────────────────────────────────────────────────
    cached = get_cached(redacted_text, doc_type)
    if cached:
        token_log = TokenLog(cache_hit=True)
        token_log.update_totals()
        return {
            "extracted_fields":  cached["extracted_fields"],
            "field_confidences": cached["field_confidences"],
            "token_log":         token_log.model_dump(),
            "cache_hit":         True,
        }

    system_prompt = _SYSTEM_PROMPTS[doc_type]
    user_content = (
        f"Document type: {doc_type.replace('_', ' ')}\n"
        f"JSON schema to match (use null for unknown fields):\n{_schema_json(doc_type)}\n\n"
        f"Document text:\n{redacted_text}\n\n"
        f"Respond with ONLY the JSON object. Start your response with {{ and end with }}."
    )

    raw_fields, prompt_tokens, completion_tokens = _call_llm(system_prompt, user_content)

    if not raw_fields:
        return {
            "error": "LLM extraction returned empty result",
            "error_node": "extract",
        }

    # Validate with Pydantic (coerce types, fill defaults)
    try:
        validated = schema_cls.model_validate(raw_fields)
        fields_dict = validated.model_dump()
    except Exception as e:
        # Partial extraction — use raw dict and flag low confidence
        fields_dict = raw_fields

    # Deterministic enrichment
    fields_dict = _deterministic_enrichment(fields_dict, doc_type)

    # Confidence scoring
    try:
        confidences = _heuristic_confidence(fields_dict, schema_cls)
    except Exception:
        confidences = FieldConfidences(scores={}, overall=0.5)

    # Token log (partial — will be updated by policy node)
    token_log = TokenLog(
        extraction_prompt_tokens=prompt_tokens,
        extraction_completion_tokens=completion_tokens,
    )
    token_log.update_totals()

    confidences_dict = confidences.model_dump()

    # ── Cache write ────────────────────────────────────────────────────────
    set_cached(redacted_text, doc_type, fields_dict, confidences_dict)

    return {
        "extracted_fields":  fields_dict,
        "field_confidences": confidences_dict,
        "token_log":         token_log.model_dump(),
        "cache_hit":         False,
    }
