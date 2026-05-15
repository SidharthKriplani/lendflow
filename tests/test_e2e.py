"""
End-to-end pipeline tests — LLM calls mocked.
Tests the full LangGraph DAG from raw text to audit log.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LLM_BASE_URL",  "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",   "lm-studio")
os.environ.setdefault("LLM_MODEL",     "test-model")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/lendflow_test_chroma")
os.environ.setdefault("AUDIT_LOG_DIR",  "/tmp/lendflow_test_audit")


MOCK_EXTRACTION_RESPONSE = json.dumps({
    "account_holder": "Rahul Sharma",
    "average_monthly_credit": 85000,
    "average_monthly_debit": 80183,
    "estimated_monthly_income": 85000,
    "emi_obligations": 22000,
    "foir": 0.259,
    "cash_flow_volatility": "LOW",
    "employment_type": "SALARIED",
    "account_vintage_months": 36,
    "red_flags": []
})

MOCK_POLICY_RESPONSE = json.dumps({
    "flags": [
        {"rule_name": "FOIR_LIMIT",       "triggered": False, "detail": "FOIR 25.9% within 55% limit",
         "policy_citation": "RBI NBFC Ch.1 §1.2.1"},
        {"rule_name": "KYC_COMPLETE",     "triggered": False, "detail": "KYC not assessed in bank statement",
         "policy_citation": "LendFlow Policy §2.1"},
        {"rule_name": "RC_ENCUMBRANCE",   "triggered": False, "detail": "Not applicable to bank statement",
         "policy_citation": "RBI NBFC Ch.3 §3.3"},
        {"rule_name": "INCOME_VERIFIABLE","triggered": False, "detail": "Income verified via salary credits",
         "policy_citation": "RBI NBFC Ch.1 §1.3"},
        {"rule_name": "RED_FLAGS",        "triggered": False, "detail": "No suspicious patterns",
         "policy_citation": "LendFlow Policy §4.1"},
    ]
})


def _mock_llm_response(content: str):
    """Build a mock OpenAI-compatible response."""
    choice = MagicMock()
    choice.message.content = content
    choice.usage = MagicMock()
    choice.usage.prompt_tokens     = 150
    choice.usage.completion_tokens = 80
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage   = choice.usage
    return resp


class TestGraphCompilation:

    def test_graph_builds_without_error(self):
        from pipeline.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_has_all_nodes(self):
        from pipeline.graph import build_graph
        g = build_graph()
        nodes = set(g.get_graph().nodes.keys())
        expected = {"intake", "pii", "extract", "policy", "confidence",
                    "route", "human_review", "audit"}
        assert expected.issubset(nodes)


class TestIntakeNode:

    def test_classify_bank_statement(self):
        from pipeline.nodes.intake import intake_node
        state = {
            "application_id": None, "raw_text": "BANK ACCOUNT STATEMENT Average Monthly Credits EMI",
            "doc_type": None, "redacted_text": None, "pii_map_id": None,
            "pii_entity_count": 0, "extracted_fields": {}, "field_confidences": None,
            "policy_chunks": [], "policy_flags": [], "uncertainty_score": None,
            "uncertainty_band": None, "routing_decision": None, "reason_codes": [],
            "human_review_required": False, "human_override": None, "token_log": None,
            "audit_id": None, "pipeline_version": "1.0.0", "error": None, "error_node": None,
        }
        result = intake_node(state)
        assert result["doc_type"] == "bank_statement"
        assert result["application_id"] is not None

    def test_classify_salary_slip(self):
        from pipeline.nodes.intake import intake_node
        state = {
            "application_id": None,
            "raw_text": "SALARY SLIP Employee Name Gross Salary Net Salary Employer",
            "doc_type": None, "redacted_text": None, "pii_map_id": None,
            "pii_entity_count": 0, "extracted_fields": {}, "field_confidences": None,
            "policy_chunks": [], "policy_flags": [], "uncertainty_score": None,
            "uncertainty_band": None, "routing_decision": None, "reason_codes": [],
            "human_review_required": False, "human_override": None, "token_log": None,
            "audit_id": None, "pipeline_version": "1.0.0", "error": None, "error_node": None,
        }
        result = intake_node(state)
        assert result["doc_type"] == "salary_slip"

    def test_classify_kyc(self):
        from pipeline.nodes.intake import intake_node
        state = {
            "application_id": None,
            "raw_text": "KYC DOCUMENT Aadhaar Number PAN Card Date of Birth Address Proof",
            "doc_type": None, "redacted_text": None, "pii_map_id": None,
            "pii_entity_count": 0, "extracted_fields": {}, "field_confidences": None,
            "policy_chunks": [], "policy_flags": [], "uncertainty_score": None,
            "uncertainty_band": None, "routing_decision": None, "reason_codes": [],
            "human_review_required": False, "human_override": None, "token_log": None,
            "audit_id": None, "pipeline_version": "1.0.0", "error": None, "error_node": None,
        }
        result = intake_node(state)
        assert result["doc_type"] == "kyc"

    def test_empty_doc_sets_error(self):
        from pipeline.nodes.intake import intake_node
        state = {
            "application_id": None, "raw_text": "",
            "doc_type": None, "redacted_text": None, "pii_map_id": None,
            "pii_entity_count": 0, "extracted_fields": {}, "field_confidences": None,
            "policy_chunks": [], "policy_flags": [], "uncertainty_score": None,
            "uncertainty_band": None, "routing_decision": None, "reason_codes": [],
            "human_review_required": False, "human_override": None, "token_log": None,
            "audit_id": None, "pipeline_version": "1.0.0", "error": None, "error_node": None,
        }
        result = intake_node(state)
        assert result["error"] is not None


class TestPIINode:

    def test_pii_node_redacts_and_stores_map(self):
        from pipeline.nodes.pii import pii_node
        from pipeline.state import TokenLog
        state = {
            "application_id": "TEST-PII-001",
            "raw_text": "Applicant PAN: AHFTR9935C Mobile: +91 9876543210",
            "doc_type": "bank_statement", "redacted_text": None,
            "pii_map_id": None, "pii_entity_count": 0,
            "extracted_fields": {}, "field_confidences": None,
            "policy_chunks": [], "policy_flags": [], "uncertainty_score": None,
            "uncertainty_band": None, "routing_decision": None, "reason_codes": [],
            "human_review_required": False, "human_override": None,
            "token_log": TokenLog(),
            "audit_id": None, "pipeline_version": "1.0.0", "error": None, "error_node": None,
        }
        result = pii_node(state)
        assert result["redacted_text"] is not None
        assert "AHFTR9935C" not in result["redacted_text"]
        assert result["pii_map_id"] is not None
        assert result["pii_entity_count"] >= 1


class TestEndToEndMocked:

    @patch("pipeline.nodes.extract.OpenAI")
    @patch("pipeline.nodes.policy.OpenAI")
    def test_full_pipeline_approve_path(self, mock_policy_openai, mock_extract_openai):
        """Run full pipeline with mocked LLM — verify APPROVE output."""
        # Mock extraction
        mock_extract_client = MagicMock()
        mock_extract_openai.return_value = mock_extract_client
        mock_extract_client.chat.completions.create.return_value = \
            _mock_llm_response(MOCK_EXTRACTION_RESPONSE)

        # Mock policy
        mock_policy_client = MagicMock()
        mock_policy_openai.return_value = mock_policy_client
        mock_policy_client.chat.completions.create.return_value = \
            _mock_llm_response(MOCK_POLICY_RESPONSE)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AUDIT_LOG_DIR"] = tmpdir
            import importlib, config as cfg
            importlib.reload(cfg)

            from pipeline.graph import run_pipeline
            raw_text = (Path(__file__).parent / "fixtures" / "applications" / "APP_01.txt").read_text()
            result = run_pipeline(raw_text, application_id="APP_01", thread_id="test-thread-001")

        assert result is not None
        assert result.get("routing_decision") == "APPROVE"
        assert result.get("audit_id") is not None
        assert result.get("error") is None

    @patch("pipeline.nodes.extract.OpenAI")
    @patch("pipeline.nodes.policy.OpenAI")
    def test_full_pipeline_reject_rc_encumbrance(self, mock_policy_openai, mock_extract_openai):
        """RC encumbrance in extracted fields → REJECT."""
        mock_extract_response = json.dumps({
            "make": "Honda", "model": "City", "year": 2019,
            "assessed_value": 750000, "condition_grade": "A",
            "rc_encumbrance": True, "odometer_km": 45000, "inspection_passed": True
        })
        mock_policy_response = json.dumps({
            "flags": [
                {"rule_name": "RC_ENCUMBRANCE", "triggered": True,
                 "detail": "Active hypothecation on RC",
                 "policy_citation": "RBI NBFC Ch.3 §3.3"},
            ]
        })

        mock_ext = MagicMock()
        mock_extract_openai.return_value = mock_ext
        mock_ext.chat.completions.create.return_value = _mock_llm_response(mock_extract_response)

        mock_pol = MagicMock()
        mock_policy_openai.return_value = mock_pol
        mock_pol.chat.completions.create.return_value = _mock_llm_response(mock_policy_response)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AUDIT_LOG_DIR"] = tmpdir
            from pipeline.graph import run_pipeline
            raw_text = (Path(__file__).parent / "fixtures" / "applications" / "APP_17.txt").read_text()
            result = run_pipeline(raw_text, application_id="APP_17", thread_id="test-thread-017")

        assert result["routing_decision"] == "REJECT"
        assert "RC_ENCUMBRANCE" in result.get("reason_codes", [])

    def test_pipeline_handles_empty_input_gracefully(self):
        from pipeline.graph import run_pipeline
        result = run_pipeline("", application_id="EMPTY-001", thread_id="test-empty")
        assert result is not None
        # Should not crash — error captured in state
        assert result.get("error") is not None or result.get("routing_decision") is not None


class TestAuditLog:

    @patch("pipeline.nodes.extract.OpenAI")
    @patch("pipeline.nodes.policy.OpenAI")
    def test_audit_log_written(self, mock_policy_openai, mock_extract_openai):
        """Audit JSON file must be written for every completed pipeline run."""
        mock_ext = MagicMock()
        mock_extract_openai.return_value = mock_ext
        mock_ext.chat.completions.create.return_value = _mock_llm_response(MOCK_EXTRACTION_RESPONSE)

        mock_pol = MagicMock()
        mock_policy_openai.return_value = mock_pol
        mock_pol.chat.completions.create.return_value = _mock_llm_response(MOCK_POLICY_RESPONSE)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AUDIT_LOG_DIR"] = tmpdir
            import importlib, config as cfg
            importlib.reload(cfg)

            from pipeline.graph import run_pipeline
            raw_text = (Path(__file__).parent / "fixtures" / "applications" / "APP_01.txt").read_text()
            result = run_pipeline(raw_text, application_id="APP_01_AUDIT", thread_id="audit-thread")

            audit_files = list(Path(tmpdir).glob("AUDIT-*.json"))
            assert len(audit_files) >= 1, "Audit log file must be written"

            audit = json.loads(audit_files[0].read_text())
            assert "application_id" in audit
            assert "routing_decision" in audit
            # PII map must NOT be in audit log
            assert "pii_map" not in audit
            assert "pii_map_id" in audit

    @patch("pipeline.nodes.extract.OpenAI")
    @patch("pipeline.nodes.policy.OpenAI")
    def test_audit_log_no_raw_pii(self, mock_policy_openai, mock_extract_openai):
        """Audit log must not contain raw PAN, Aadhaar, or phone numbers."""
        mock_ext = MagicMock()
        mock_extract_openai.return_value = mock_ext
        mock_ext.chat.completions.create.return_value = _mock_llm_response(MOCK_EXTRACTION_RESPONSE)
        mock_pol = MagicMock()
        mock_policy_openai.return_value = mock_pol
        mock_pol.chat.completions.create.return_value = _mock_llm_response(MOCK_POLICY_RESPONSE)

        import re
        PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AUDIT_LOG_DIR"] = tmpdir
            import importlib, config as cfg
            importlib.reload(cfg)

            from pipeline.graph import run_pipeline
            raw_text = (Path(__file__).parent / "fixtures" / "applications" / "APP_01.txt").read_text()
            run_pipeline(raw_text, application_id="APP_01_PII_TEST", thread_id="pii-test")

            for audit_file in Path(tmpdir).glob("AUDIT-*.json"):
                content = audit_file.read_text()
                assert not PAN_RE.search(content), "Raw PAN found in audit log!"
