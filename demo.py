"""
LendFlow Demo Script
Runs 5 representative applications (one per expected routing outcome)
and prints a formatted summary table.

Usage: python demo.py
Requires: LM Studio running at http://localhost:1234/v1
          python rag/indexer.py run first
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("LLM_BASE_URL",  "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",   "lm-studio")
os.environ.setdefault("LLM_MODEL",     os.getenv("LLM_MODEL", "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).parent / "chroma_db"))
os.environ.setdefault("AUDIT_LOG_DIR",  str(Path(__file__).parent / "audit_logs"))

DEMO_APPS = [
    ("APP_01", "SALARIED_CLEAR",     "APPROVE"),   # clean salaried, FOIR 25.9%
    ("APP_02", "SALARIED_HI_FOIR",   "REJECT"),    # FOIR 60%, over limit
    ("APP_04", "THIN_FILE",          "ESCALATE"),  # UNKNOWN employment, low confidence
    ("APP_11", "KYC_COMPLETE",       "APPROVE"),   # full KYC verified
    ("APP_17", "VEH_ENCUMBERED",     "REJECT"),    # RC encumbrance hard block
]

APPS_DIR = Path(__file__).parent / "tests" / "fixtures" / "applications"
GT_DIR   = Path(__file__).parent / "tests" / "fixtures" / "ground_truth"

ROUTING_ICON = {"APPROVE": "✅", "REJECT": "❌", "ESCALATE": "🔄"}


def print_banner():
    print("\n" + "="*70)
    print("  🏦  LendFlow — Vehicle Loan Document Intelligence Pipeline")
    print("  Local LLM · Privacy-First · Audit-Ready · LangGraph")
    print("="*70)


def run_demo():
    print_banner()

    from pipeline.graph import run_pipeline

    print(f"\n{'APP':8s}  {'LABEL':22s}  {'EXPECTED':10s}  {'RESULT':10s}  {'MATCH':6s}  {'LATENCY':8s}  REASONS")
    print("-" * 100)

    for app_id, label, expected in DEMO_APPS:
        txt_path = APPS_DIR / f"{app_id}.txt"
        if not txt_path.exists():
            print(f"{app_id:8s}  File not found — run scripts/generate_test_data.py first")
            continue

        raw_text = txt_path.read_text(encoding="utf-8")
        t0 = time.time()
        try:
            result   = run_pipeline(raw_text, application_id=app_id,
                                    thread_id=f"demo-{app_id}")
            latency  = time.time() - t0
            decision = result.get("routing_decision", "UNKNOWN")
            reasons  = ", ".join(result.get("reason_codes", [])[:2])
            match    = "✅" if decision == expected else "❌"
            icon     = ROUTING_ICON.get(decision, "?")
            print(f"{app_id:8s}  {label:22s}  {expected:10s}  {icon} {decision:8s}  {match:6s}  {latency:.1f}s  {reasons}")

            if result.get("human_review_required"):
                print(f"         → ⚠️  HITL gate triggered — human review required")

        except Exception as e:
            latency = time.time() - t0
            print(f"{app_id:8s}  {label:22s}  {expected:10s}  ERROR     ❌     {latency:.1f}s  {e}")

    print("-" * 100)
    print("\n📁 Audit logs written to:", os.environ["AUDIT_LOG_DIR"])
    print("🔍 Run `python scripts/run_eval.py` for full 20-app evaluation\n")


if __name__ == "__main__":
    run_demo()
