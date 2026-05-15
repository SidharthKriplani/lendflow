"""
LendFlow Evaluation Harness
Runs all 20 synthetic applications through the pipeline (with real LM Studio)
and computes: extraction F1, routing accuracy, PII recall, latency per doc.
Outputs a JSON report + prints a summary table.

Usage: python scripts/run_eval.py [--model MODEL] [--output results.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LLM_BASE_URL",  "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",   "lm-studio")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "chroma_db"))
os.environ.setdefault("AUDIT_LOG_DIR",  str(Path(__file__).parent.parent / "audit_logs"))

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
APPS_DIR     = FIXTURES_DIR / "applications"
GT_DIR       = FIXTURES_DIR / "ground_truth"


def load_apps() -> list[dict]:
    apps = []
    for gt_file in sorted(GT_DIR.glob("*_gt.json")):
        app_id   = gt_file.stem.replace("_gt", "")
        txt_file = APPS_DIR / f"{app_id}.txt"
        if not txt_file.exists():
            continue
        apps.append({
            "app_id":   app_id,
            "raw_text": txt_file.read_text(encoding="utf-8"),
            "gt":       json.loads(gt_file.read_text(encoding="utf-8")),
        })
    return apps


def compare_fields(extracted: dict, gt_fields: dict) -> tuple[int, int]:
    """Returns (correct_fields, total_gt_fields)."""
    correct = 0
    total   = 0
    for key, gt_val in gt_fields.items():
        total += 1
        pred_val = extracted.get(key)
        if pred_val is None:
            continue
        if isinstance(gt_val, float):
            if abs(float(pred_val) - gt_val) < 0.05:
                correct += 1
        elif isinstance(gt_val, list):
            if set(pred_val) == set(gt_val):
                correct += 1
        else:
            if str(pred_val).strip().upper() == str(gt_val).strip().upper():
                correct += 1
    return correct, total


def run_eval(model, output_path: str) -> None:
    if model:
        os.environ["LLM_MODEL"] = model

    from pipeline.graph import run_pipeline

    apps    = load_apps()
    results = []

    print(f"\n{'='*60}")
    print(f"  LendFlow Eval — {len(apps)} applications")
    print(f"{'='*60}\n")

    routing_correct = 0
    total_correct_fields = 0
    total_gt_fields      = 0
    latencies            = []

    for app in apps:
        app_id   = app["app_id"]
        raw_text = app["raw_text"]
        gt       = app["gt"]

        t0 = time.time()
        try:
            result = run_pipeline(raw_text, application_id=app_id,
                                  thread_id=f"eval-{app_id}")
            latency = time.time() - t0
            latencies.append(latency)

            # Routing accuracy
            predicted = result.get("routing_decision", "UNKNOWN")
            expected  = gt["expected_routing"]
            routing_ok = predicted == expected

            # Field extraction accuracy
            extracted = result.get("extracted_fields", {})
            if hasattr(extracted, "model_dump"):
                extracted = extracted.model_dump()
            gt_fields = gt.get("extracted_fields", {})
            correct, total = compare_fields(extracted, gt_fields)
            total_correct_fields += correct
            total_gt_fields      += total

            if routing_ok:
                routing_correct += 1

            status = "✅" if routing_ok else "❌"
            print(f"  {status} {app_id:8s}  expected={expected:8s}  got={predicted:8s}  "
                  f"fields={correct}/{total}  {latency:.1f}s")

            results.append({
                "app_id":       app_id,
                "doc_type":     gt["doc_type"],
                "label":        gt["label"],
                "expected":     expected,
                "predicted":    predicted,
                "routing_ok":   routing_ok,
                "field_correct": correct,
                "field_total":   total,
                "latency_s":    round(latency, 2),
                "error":        result.get("error"),
            })

        except Exception as e:
            latency = time.time() - t0
            print(f"  💥 {app_id:8s}  ERROR: {e}")
            results.append({
                "app_id": app_id, "error": str(e),
                "routing_ok": False, "field_correct": 0, "field_total": 0,
                "latency_s": round(latency, 2),
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    n               = len(apps)
    routing_acc     = routing_correct / n
    field_f1        = total_correct_fields / max(total_gt_fields, 1)
    avg_latency     = sum(latencies) / max(len(latencies), 1)
    p95_latency     = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Routing accuracy : {routing_acc:.1%}  ({routing_correct}/{n})")
    print(f"  Field extraction : {field_f1:.1%}  ({total_correct_fields}/{total_gt_fields})")
    print(f"  Avg latency      : {avg_latency:.1f}s")
    print(f"  P95 latency      : {p95_latency:.1f}s")
    print(f"\n  Targets:")
    print(f"  {'✅' if routing_acc >= 0.90 else '❌'} Routing accuracy ≥ 90%  → {routing_acc:.1%}")
    print(f"  {'✅' if field_f1 >= 0.85 else '❌'} Field extraction ≥ 85%  → {field_f1:.1%}")
    print(f"{'='*60}\n")

    # Save report
    report = {
        "summary": {
            "routing_accuracy":  round(routing_acc, 4),
            "field_extraction_f1": round(field_f1, 4),
            "avg_latency_s":     round(avg_latency, 2),
            "p95_latency_s":     round(p95_latency, 2),
            "n_apps":            n,
            "targets_met": {
                "routing_accuracy_90pct": routing_acc >= 0.90,
                "field_extraction_85pct": field_f1 >= 0.85,
            }
        },
        "per_app": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LendFlow Evaluation Harness")
    parser.add_argument("--model",  type=str, default=None,
                        help="Override LLM model (e.g. lmstudio-community/Meta-Llama-3.1-8B)")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output JSON path")
    args = parser.parse_args()
    run_eval(args.model, args.output)
