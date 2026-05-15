from __future__ import annotations
"""
Node 2: PII Detection & Redaction
Uses Microsoft Presidio to detect and replace PII with typed placeholders
BEFORE any text reaches the LLM. Stores a secure restoration map.

PII never reaches the LLM API. This satisfies RBI data-residency framing:
even if using a cloud LLM, sensitive identifiers are never transmitted.
"""
import re
import uuid
from typing import Optional

from pipeline.state import LendFlowState

# ── Presidio imports (graceful degradation if not installed) ──────────────────
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    print("[PII] Warning: presidio not installed. Using regex-only fallback.")

# ── In-memory PII map store (keyed by pii_map_id) ────────────────────────────
# In production: replace with an encrypted key-value store (Redis with TTL).
_PII_MAPS: dict[str, dict[str, str]] = {}


def _build_analyzer() -> Optional[object]:
    """Build Presidio analyzer with India-specific custom recognizers."""
    if not _PRESIDIO_AVAILABLE:
        return None

    analyzer = AnalyzerEngine()

    # Aadhaar: 12 digits, optionally space/hyphen separated
    aadhaar_recognizer = PatternRecognizer(
        supported_entity="AADHAAR",
        patterns=[
            Pattern("aadhaar_spaced",  r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b", 0.9),
            Pattern("aadhaar_compact", r"\b\d{12}\b", 0.6),
        ],
    )
    # PAN: AAANNNNNA format
    pan_recognizer = PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=[Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.95)],
    )
    # Indian mobile: 10 digits starting with 6-9
    mobile_recognizer = PatternRecognizer(
        supported_entity="IN_PHONE",
        patterns=[
            Pattern("mobile_prefix", r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", 0.85),
        ],
    )
    # IFSC code
    ifsc_recognizer = PatternRecognizer(
        supported_entity="IFSC",
        patterns=[Pattern("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.9)],
    )

    analyzer.registry.add_recognizer(aadhaar_recognizer)
    analyzer.registry.add_recognizer(pan_recognizer)
    analyzer.registry.add_recognizer(mobile_recognizer)
    analyzer.registry.add_recognizer(ifsc_recognizer)
    return analyzer


def _regex_redact_fallback(text: str) -> tuple[str, dict[str, str]]:
    """
    Fallback PII redaction using pure regex when Presidio is unavailable.
    Returns (redacted_text, restoration_map).
    """
    mapping: dict[str, str] = {}
    counter: dict[str, int] = {}

    patterns = [
        ("AADHAAR", r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"),
        ("AADHAAR", r"\b\d{12}\b"),
        ("IN_PAN",  r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        ("IN_PHONE",r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
        ("IFSC",    r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        ("EMAIL",   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ]

    for entity_type, pattern in patterns:
        for match in re.finditer(pattern, text):
            original = match.group()
            if original not in mapping.values():
                counter[entity_type] = counter.get(entity_type, 0) + 1
                placeholder = f"[{entity_type}_{counter[entity_type]}]"
                mapping[placeholder] = original
                text = text.replace(original, placeholder, 1)

    return text, mapping


def redact(text: str, analyzer=None, anonymizer=None) -> tuple[str, dict[str, str]]:
    """
    Detect and replace PII. Returns (redacted_text, placeholder→original map).
    """
    if not _PRESIDIO_AVAILABLE or analyzer is None:
        return _regex_redact_fallback(text)

    entities_to_detect = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "AADHAAR",
        "IN_PAN", "IN_PHONE", "IFSC", "CREDIT_CARD", "IBAN_CODE",
    ]

    results = analyzer.analyze(text=text, entities=entities_to_detect, language="en")

    # Sort by position descending to replace without offset issues
    results_sorted = sorted(results, key=lambda r: r.start, reverse=True)

    mapping: dict[str, str] = {}
    entity_counters: dict[str, int] = {}
    redacted = text

    for result in results_sorted:
        entity_type = result.entity_type
        entity_counters[entity_type] = entity_counters.get(entity_type, 0) + 1
        placeholder = f"[{entity_type}_{entity_counters[entity_type]}]"
        original = redacted[result.start:result.end]
        mapping[placeholder] = original
        redacted = redacted[:result.start] + placeholder + redacted[result.end:]

    return redacted, mapping


def restore(text: str, pii_map_or_id) -> str:
    """
    Restore PII placeholders back to original values.
    Accepts either a pii_map_id string (looks up _PII_MAPS) or a dict map directly.
    """
    if isinstance(pii_map_or_id, dict):
        mapping = pii_map_or_id
    else:
        mapping = _PII_MAPS.get(pii_map_or_id, {})
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


# ── Module-level Presidio instances (initialised once) ────────────────────────
_analyzer  = _build_analyzer()
_anonymizer = None
if _PRESIDIO_AVAILABLE:
    try:
        _anonymizer = AnonymizerEngine()
    except Exception:
        pass


def pii_node(state: LendFlowState) -> dict:
    """
    PII redaction node.
    Input:  state['raw_text']
    Output: redacted_text, pii_map_id, pii_entity_count
    """
    if state.get("error"):
        return {}  # Skip if upstream node failed

    raw_text = state.get("raw_text", "")

    redacted_text, pii_map = redact(raw_text, _analyzer, _anonymizer)

    # Store map securely under a UUID — the state only holds the ID, not the map
    pii_map_id = f"PII-{uuid.uuid4().hex}"
    _PII_MAPS[pii_map_id] = pii_map

    return {
        "redacted_text":    redacted_text,
        "pii_map_id":       pii_map_id,
        "pii_entity_count": len(pii_map),
    }
