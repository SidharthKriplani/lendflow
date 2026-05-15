from __future__ import annotations
"""
LendFlow Pipeline State
The single TypedDict that flows through every LangGraph node.
Every node reads from state and returns a partial update dict.
"""
from typing import TypedDict, Optional, Any
from pydantic import BaseModel, Field


# ── Pydantic schemas for structured extraction ─────────────────────────────────

class BankStatementFields(BaseModel):
    """Extracted fields from a bank statement."""
    account_holder: Optional[str]        = Field(None, description="Account holder name (placeholder after PII redaction)")
    average_monthly_credit: Optional[float] = Field(None, description="Average monthly credits (income proxy), INR")
    average_monthly_debit:  Optional[float] = Field(None, description="Average monthly debits (obligations proxy), INR")
    estimated_monthly_income: Optional[float] = Field(None, description="Estimated net monthly income, INR")
    emi_obligations:          Optional[float] = Field(None, description="Total identified EMI outflows per month, INR")
    foir:                     Optional[float] = Field(None, description="Fixed Obligation to Income Ratio (0.0-1.0)")
    cash_flow_volatility:     Optional[str]   = Field(None, description="LOW / MEDIUM / HIGH — variability of monthly credits")
    employment_type:          Optional[str]   = Field(None, description="SALARIED / SELF_EMPLOYED / UNKNOWN")
    account_vintage_months:   Optional[int]   = Field(None, description="How many months of statements provided")
    red_flags:                list[str]       = Field(default_factory=list, description="Anomalies: round-tripping, sudden spikes, etc.")


class SalarySlipFields(BaseModel):
    """Extracted fields from a salary slip."""
    employee_name:    Optional[str]   = Field(None)
    employer_name:    Optional[str]   = Field(None)
    gross_salary:     Optional[float] = Field(None, description="Gross monthly salary, INR")
    net_salary:       Optional[float] = Field(None, description="Net take-home salary, INR")
    emi_deductions:   Optional[float] = Field(None, description="EMI deductions visible in salary slip, INR")
    employment_type:  Optional[str]   = Field(None, description="SALARIED / CONTRACT / UNKNOWN")
    designation:      Optional[str]   = Field(None)
    pan_placeholder:  Optional[str]   = Field(None, description="PAN placeholder if present in slip")


class KYCFields(BaseModel):
    """Extracted fields from a KYC document."""
    applicant_name:     Optional[str]  = Field(None)
    date_of_birth:      Optional[str]  = Field(None)
    aadhaar_placeholder: Optional[str] = Field(None, description="Aadhaar placeholder after redaction")
    pan_placeholder:    Optional[str]  = Field(None, description="PAN placeholder after redaction")
    address_present:    bool           = Field(False, description="Whether address is present in doc")
    kyc_complete:       bool           = Field(False, description="Whether all required KYC fields are present")
    verification_status: Optional[str] = Field(None, description="VERIFIED / PENDING / MISMATCH")


class VehicleReportFields(BaseModel):
    """Extracted fields from a vehicle inspection / Profecto report."""
    vehicle_make:       Optional[str]   = Field(None)
    vehicle_model:      Optional[str]   = Field(None)
    manufacture_year:   Optional[int]   = Field(None)
    assessed_value:     Optional[float] = Field(None, description="Assessed vehicle value, INR")
    condition_grade:    Optional[str]   = Field(None, description="A/B/C/D — overall condition grade")
    rc_encumbrance:     Optional[bool]  = Field(None, description="Whether RC shows existing hypothecation")
    odometer_km:        Optional[int]   = Field(None)
    inspection_passed:  Optional[bool]  = Field(None)


# ── Field confidence scores ───────────────────────────────────────────────────

class FieldConfidences(BaseModel):
    """Per-field confidence scores (0.0–1.0) from the extraction node."""
    scores: dict[str, float] = Field(default_factory=dict)
    overall: float = Field(0.0)

    def low_confidence_fields(self, threshold: float = 0.65) -> list[str]:
        return [f for f, s in self.scores.items() if s < threshold]


# ── Policy check result ───────────────────────────────────────────────────────

class PolicyFlag(BaseModel):
    rule_name:    str
    triggered:    bool
    detail:       str
    policy_citation: Optional[str] = None


# ── Token usage log ───────────────────────────────────────────────────────────

class TokenLog(BaseModel):
    extraction_prompt_tokens:    int = 0
    extraction_completion_tokens: int = 0
    policy_check_prompt_tokens:   int = 0
    policy_check_completion_tokens: int = 0
    total_prompt_tokens:    int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd:     float = 0.0
    cache_hit:              bool = False

    def update_totals(self):
        self.total_prompt_tokens = (
            self.extraction_prompt_tokens + self.policy_check_prompt_tokens
        )
        self.total_completion_tokens = (
            self.extraction_completion_tokens + self.policy_check_completion_tokens
        )
        # GPT-4o-mini pricing as reference; LM Studio = $0
        self.estimated_cost_usd = round(
            (self.total_prompt_tokens / 1_000_000) * 0.15
            + (self.total_completion_tokens / 1_000_000) * 0.60,
            6,
        )


# ── Main pipeline state ───────────────────────────────────────────────────────

class LendFlowState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    application_id:  str            # Unique ID assigned at intake
    raw_text:        str            # Raw document text (from PDF parser or direct input)
    doc_type:        str            # Classified document type (from config.DOC_*)

    # ── PII layer ──────────────────────────────────────────────────────────
    redacted_text:   str            # Text after PII placeholder replacement
    pii_map_id:      str            # ID pointing to secure PII placeholder map (not stored in state)
    pii_entity_count: int           # Number of PII entities detected

    # ── Extraction layer ───────────────────────────────────────────────────
    extracted_fields: dict[str, Any]   # Pydantic model serialized to dict
    field_confidences: dict[str, Any]  # FieldConfidences serialized

    # ── Policy RAG layer ───────────────────────────────────────────────────
    policy_chunks:   list[dict]     # Retrieved policy chunks [{text, source, score}]
    policy_flags:    list[dict]     # PolicyFlag list serialized

    # ── Confidence & routing ───────────────────────────────────────────────
    uncertainty_score:  float       # 0.0 (certain) to 1.0 (highly uncertain)
    uncertainty_band:   str         # HIGH / MEDIUM / LOW confidence → maps to routing
    routing_decision:   str         # APPROVE / ESCALATE / REJECT
    reason_codes:       list[str]   # Human-readable reason codes

    # ── HITL ───────────────────────────────────────────────────────────────
    human_review_required: bool
    human_override:        dict     # {reviewer_id, decision, justification, timestamp}

    # ── Token tracking ─────────────────────────────────────────────────────
    token_log:       dict           # TokenLog serialized

    # ── Audit ──────────────────────────────────────────────────────────────
    audit_id:        str
    pipeline_version: str

    # ── Error handling ─────────────────────────────────────────────────────
    error:           Optional[str]  # Set if any node fails; downstream nodes check this
    error_node:      Optional[str]  # Which node raised the error
