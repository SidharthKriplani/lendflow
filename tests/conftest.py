"""Shared pytest fixtures for LendFlow tests."""
import json
import os
import sys
from pathlib import Path
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LLM_BASE_URL",  "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",   "lm-studio")
os.environ.setdefault("LLM_MODEL",     "test-model")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "chroma_db"))
os.environ.setdefault("AUDIT_LOG_DIR",  str(Path(__file__).parent.parent / "audit_logs"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
APPS_DIR     = FIXTURES_DIR / "applications"
GT_DIR       = FIXTURES_DIR / "ground_truth"


@pytest.fixture(scope="session")
def all_apps():
    """Return list of (app_id, raw_text, ground_truth) for all 20 synthetic apps."""
    apps = []
    for gt_file in sorted(GT_DIR.glob("*_gt.json")):
        app_id = gt_file.stem.replace("_gt", "")
        txt_file = APPS_DIR / f"{app_id}.txt"
        if not txt_file.exists():
            continue
        raw_text = txt_file.read_text(encoding="utf-8")
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        apps.append((app_id, raw_text, gt))
    return apps


@pytest.fixture(scope="session")
def bank_statement_apps(all_apps):
    return [(a, t, g) for a, t, g in all_apps if g["doc_type"] == "bank_statement"]


@pytest.fixture(scope="session")
def salary_slip_apps(all_apps):
    return [(a, t, g) for a, t, g in all_apps if g["doc_type"] == "salary_slip"]


@pytest.fixture(scope="session")
def kyc_apps(all_apps):
    return [(a, t, g) for a, t, g in all_apps if g["doc_type"] == "kyc"]


@pytest.fixture(scope="session")
def vehicle_report_apps(all_apps):
    return [(a, t, g) for a, t, g in all_apps if g["doc_type"] == "vehicle_report"]
