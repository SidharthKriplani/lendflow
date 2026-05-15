"""
Test PII redaction node — target: ≥95% PII recall across all synthetic docs.
Runs with regex-only fallback if Presidio is not installed.
"""
import re
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.nodes.pii import redact, restore, _PII_MAPS

# ── PII patterns we expect to find in synthetic docs ─────────────────────────
PAN_PATTERN    = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b")
IFSC_PATTERN   = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
PHONE_PATTERN  = re.compile(r"\+91\s?\d{10}")
ACCOUNT_PATTERN = re.compile(r"\b\d{9,18}\b")


def count_pii_in_text(text: str) -> dict:
    return {
        "pan":     len(PAN_PATTERN.findall(text)),
        "aadhaar": len(AADHAAR_PATTERN.findall(text)),
        "ifsc":    len(IFSC_PATTERN.findall(text)),
        "phone":   len(PHONE_PATTERN.findall(text)),
    }


class TestPIIRedaction:

    def test_pan_redacted(self):
        text = "Applicant PAN: AHFTR9935C and mobile +91 9876543210"
        redacted, pii_map = redact(text)
        assert not PAN_PATTERN.search(redacted), "PAN should be redacted"
        assert "<PAN_" in redacted or "PAN" not in redacted or len(pii_map) > 0

    def test_aadhaar_redacted(self):
        text = "Aadhaar Number: 4321 8765 2109"
        redacted, pii_map = redact(text)
        assert not AADHAAR_PATTERN.search(redacted), "Aadhaar should be redacted"

    def test_ifsc_redacted(self):
        text = "IFSC Code: HDFC0001234 Branch Mumbai"
        redacted, pii_map = redact(text)
        assert not IFSC_PATTERN.search(redacted), "IFSC should be redacted"

    def test_phone_redacted(self):
        text = "Mobile: +91 9876543210"
        redacted, pii_map = redact(text)
        assert not PHONE_PATTERN.search(redacted), "Phone should be redacted"

    def test_pii_map_stored(self):
        text = "PAN: BFGTR1234A Mobile: +91 9123456789"
        redacted, pii_map = redact(text)
        assert len(pii_map) >= 1, "PII map should have at least one entry"

    def test_restore_roundtrip(self):
        """Redacted placeholders must restore to original values."""
        original = "PAN CDETR5678B IFSC SBIN0001234"
        redacted, pii_map = redact(original)
        restored = restore(redacted, pii_map)
        # Original PII should be back
        assert "CDETR5678B" in restored
        assert "SBIN0001234" in restored

    def test_redact_returns_tuple(self):
        text = "Some text with PAN ABCDE1234F"
        result = redact(text)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_empty_text(self):
        redacted, pii_map = redact("")
        assert redacted == ""
        assert pii_map == {}

    def test_no_pii_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        redacted, pii_map = redact(text)
        assert redacted == text  # unchanged
        assert pii_map == {}


class TestPIIRecallOnSyntheticDocs:
    """
    Recall test: for each synthetic doc, count PII entities in original,
    verify they are absent in redacted text.
    Target: ≥95% recall (PII present in original but absent after redaction).
    """

    @pytest.fixture(autouse=True)
    def load_apps(self, all_apps):
        self.apps = all_apps

    def test_pan_recall_across_all_docs(self):
        total_pans = 0
        missed = 0
        for app_id, raw_text, gt in self.apps:
            pans_before = PAN_PATTERN.findall(raw_text)
            total_pans += len(pans_before)
            redacted, _ = redact(raw_text)
            pans_after = PAN_PATTERN.findall(redacted)
            missed += len(pans_after)
        if total_pans == 0:
            pytest.skip("No PAN numbers in synthetic docs")
        recall = 1.0 - (missed / total_pans)
        assert recall >= 0.95, f"PAN recall {recall:.2%} < 95% target ({missed}/{total_pans} missed)"

    def test_phone_recall_across_all_docs(self):
        total = 0
        missed = 0
        for app_id, raw_text, gt in self.apps:
            before = PHONE_PATTERN.findall(raw_text)
            total += len(before)
            redacted, _ = redact(raw_text)
            after = PHONE_PATTERN.findall(redacted)
            missed += len(after)
        if total == 0:
            pytest.skip("No phone numbers in synthetic docs")
        recall = 1.0 - (missed / total)
        assert recall >= 0.95, f"Phone recall {recall:.2%} < 95% target"

    def test_ifsc_recall_across_all_docs(self):
        total = 0
        missed = 0
        for app_id, raw_text, gt in self.apps:
            before = IFSC_PATTERN.findall(raw_text)
            total += len(before)
            redacted, _ = redact(raw_text)
            after = IFSC_PATTERN.findall(redacted)
            missed += len(after)
        if total == 0:
            pytest.skip("No IFSC codes in synthetic docs")
        recall = 1.0 - (missed / total)
        assert recall >= 0.95, f"IFSC recall {recall:.2%} < 95% target"

    def test_overall_pii_recall(self):
        """Aggregate recall across PAN + Phone + IFSC."""
        total = 0
        missed = 0
        for app_id, raw_text, gt in self.apps:
            counts = count_pii_in_text(raw_text)
            total += counts["pan"] + counts["phone"] + counts["ifsc"]
            redacted, _ = redact(raw_text)
            after = count_pii_in_text(redacted)
            missed += after["pan"] + after["phone"] + after["ifsc"]
        if total == 0:
            pytest.skip("No PII entities found in synthetic docs")
        recall = 1.0 - (missed / total)
        print(f"\nOverall PII recall: {recall:.2%} ({total - missed}/{total} entities redacted)")
        assert recall >= 0.95, f"Overall PII recall {recall:.2%} < 95% target"

    def test_pii_map_non_empty_for_docs_with_pii(self):
        """Every doc containing PAN or phone must produce a non-empty pii_map."""
        failures = []
        for app_id, raw_text, gt in self.apps:
            has_pii = bool(PAN_PATTERN.search(raw_text) or PHONE_PATTERN.search(raw_text))
            if has_pii:
                _, pii_map = redact(raw_text)
                if not pii_map:
                    failures.append(app_id)
        assert not failures, f"Empty pii_map for docs with PII: {failures}"
