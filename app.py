"""
LendFlow — Streamlit UI
Visual showcase for the vehicle loan decisioning pipeline.
Run: streamlit run app.py
"""
from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path

import streamlit as st

# ── Path & env setup ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("LLM_BASE_URL",  "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",   "lm-studio")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).parent / "chroma_db"))
os.environ.setdefault("AUDIT_LOG_DIR",  str(Path(__file__).parent / "audit_logs"))

# ── Sample documents ──────────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent / "tests" / "fixtures" / "applications"

def load_samples() -> dict[str, str]:
    samples = {"— paste your own —": ""}
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.glob("*.txt"))[:10]:
            samples[f.stem] = f.read_text(encoding="utf-8")
    return samples

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LendFlow — Loan Decision Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.decision-approve  { background:#16a34a; color:white; padding:12px 28px; border-radius:8px;
                     font-size:1.4rem; font-weight:700; text-align:center; }
.decision-reject   { background:#dc2626; color:white; padding:12px 28px; border-radius:8px;
                     font-size:1.4rem; font-weight:700; text-align:center; }
.decision-escalate { background:#d97706; color:white; padding:12px 28px; border-radius:8px;
                     font-size:1.4rem; font-weight:700; text-align:center; }
.decision-unknown  { background:#6b7280; color:white; padding:12px 28px; border-radius:8px;
                     font-size:1.4rem; font-weight:700; text-align:center; }
.flag-triggered    { background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:4px;
                     font-size:0.85rem; font-weight:600; margin:2px; display:inline-block; }
.flag-clear        { background:#dcfce7; color:#166534; padding:4px 10px; border-radius:4px;
                     font-size:0.85rem; font-weight:600; margin:2px; display:inline-block; }
.flag-na           { background:#f3f4f6; color:#6b7280; padding:4px 10px; border-radius:4px;
                     font-size:0.85rem; margin:2px; display:inline-block; }
.metric-card       { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                     padding:12px 16px; margin:4px 0; }
.hitl-banner       { background:#fef3c7; border:2px solid #d97706; border-radius:8px;
                     padding:12px 16px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 LendFlow")
    st.markdown("*Vehicle Loan Decision Intelligence*")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    llm_url = st.text_input("LM Studio URL", value=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"))
    llm_model = st.text_input("Model", value=os.environ.get("LLM_MODEL", "google/gemma-3-4b"))

    if llm_url:
        os.environ["LLM_BASE_URL"] = llm_url
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model

    st.divider()
    st.markdown("### 🧪 Test Connectivity")
    if st.button("Ping LM Studio"):
        try:
            from openai import OpenAI
            import config
            client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
            models = client.models.list()
            st.success(f"✅ Connected — {len(list(models))} model(s) loaded")
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()
    st.markdown("""
    **Pipeline nodes**
    1. `intake` — doc classification
    2. `pii` — redaction
    3. `extract` — LLM field extraction
    4. `policy` — rule evaluation
    5. `confidence` — uncertainty scoring
    6. `route` — APPROVE / ESCALATE / REJECT
    7. `audit` — structured log
    """)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏦 LendFlow — Loan Decision Intelligence")
st.caption("Local LLM · Privacy-First · Audit-Ready · LangGraph")

tab_run, tab_audit, tab_about = st.tabs(["▶ Run Pipeline", "📋 Audit Logs", "ℹ️ About"])

# ── Tab 1: Run Pipeline ────────────────────────────────────────────────────────
with tab_run:
    samples = load_samples()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Document Input")
        selected = st.selectbox("Load a sample document", list(samples.keys()))
        doc_text = st.text_area(
            "Document text (paste or load sample above)",
            value=samples[selected],
            height=340,
            placeholder="Paste bank statement, salary slip, KYC document, or vehicle inspection report here...",
        )
        app_id = st.text_input("Application ID (optional)", placeholder="Auto-generated if blank")

        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    with col_right:
        st.markdown("### 📊 Results")

        if run_btn:
            if not doc_text.strip():
                st.warning("Please paste a document or select a sample.")
            else:
                with st.spinner("Running pipeline..."):
                    t0 = time.time()
                    try:
                        from pipeline.graph import run_pipeline
                        result = run_pipeline(
                            doc_text,
                            application_id=app_id.strip() or None,
                            thread_id=None,
                        )
                        elapsed = time.time() - t0
                        st.session_state["last_result"] = result
                        st.session_state["last_elapsed"] = elapsed
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        st.session_state["last_result"] = None

        result = st.session_state.get("last_result")
        if result:
            elapsed = st.session_state.get("last_elapsed", 0)

            # ── Routing decision banner ──────────────────────────────────────
            decision = result.get("routing_decision") or "UNKNOWN"
            css_class = {
                "APPROVE":  "decision-approve",
                "REJECT":   "decision-reject",
                "ESCALATE": "decision-escalate",
            }.get(decision, "decision-unknown")

            icon = {"APPROVE": "✅", "REJECT": "❌", "ESCALATE": "⚠️"}.get(decision, "❓")
            st.markdown(
                f'<div class="{css_class}">{icon} {decision}</div>',
                unsafe_allow_html=True,
            )

            reason_codes = result.get("reason_codes") or []
            if reason_codes:
                st.caption("Reason codes: " + " · ".join(reason_codes))
            st.caption(f"⏱ {elapsed:.1f}s · App ID: {result.get('application_id', '—')}")

            # ── HITL Banner ──────────────────────────────────────────────────
            if result.get("human_review_required"):
                st.markdown(
                    '<div class="hitl-banner">⚠️ <b>Human Review Required</b> — '
                    'pipeline paused. POST to <code>/review</code> to resume.</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Doc type + fields ────────────────────────────────────────────
            doc_type = result.get("doc_type", "—")
            st.markdown(f"**Document type:** `{doc_type}`")

            fields = result.get("extracted_fields") or {}
            if isinstance(fields, dict) and fields:
                st.markdown("**Extracted fields**")
                rows = []
                for k, v in fields.items():
                    if v is not None:
                        if isinstance(v, float) and k == "foir":
                            rows.append({"Field": k, "Value": f"{v:.1%}"})
                        elif isinstance(v, float):
                            rows.append({"Field": k, "Value": f"{v:,.0f}"})
                        else:
                            rows.append({"Field": k, "Value": str(v)})
                if rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Policy flags ─────────────────────────────────────────────────
            flags = result.get("policy_flags") or []
            if flags:
                st.markdown("**Policy flags**")
                badges = ""
                for f in flags:
                    name = f.get("rule_name") if isinstance(f, dict) else str(f)
                    triggered = f.get("triggered") if isinstance(f, dict) else False
                    detail = f.get("detail", "") if isinstance(f, dict) else ""
                    cls = "flag-triggered" if triggered else "flag-clear"
                    icon_f = "🔴" if triggered else "🟢"
                    badges += f'<span class="{cls}" title="{detail}">{icon_f} {name}</span> '
                st.markdown(badges, unsafe_allow_html=True)

            # ── Confidence ───────────────────────────────────────────────────
            band = result.get("uncertainty_band")
            score = result.get("uncertainty_score")
            if band:
                color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}.get(band, "gray")
                conf_score = round((1 - (score or 0)) * 100)
                st.markdown(f"**Confidence:** :{color}[{band} band] — {conf_score}%")
                st.progress(conf_score / 100)

# ── Tab 2: Audit Logs ─────────────────────────────────────────────────────────
with tab_audit:
    audit_dir = Path(os.environ.get("AUDIT_LOG_DIR", "./audit_logs"))
    audit_files = sorted(audit_dir.glob("AUDIT-*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not audit_files:
        st.info("No audit logs yet. Run the pipeline first.")
    else:
        st.markdown(f"**{len(audit_files)} audit log(s)** in `{audit_dir}`")

        # Summary table
        rows = []
        for af in audit_files[:50]:
            try:
                d = json.loads(af.read_text())
                rows.append({
                    "Audit ID": d.get("audit_id", af.stem),
                    "App ID": d.get("application_id", "—"),
                    "Doc Type": d.get("doc_type", "—"),
                    "Decision": d.get("routing_decision", "—"),
                    "Error": "❌" if d.get("pipeline_error") else "✅",
                    "Timestamp": d.get("timestamp_utc", "")[:19],
                })
            except Exception:
                pass

        if rows:
            import pandas as pd
            selected_row = st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        # Detail viewer
        selected_audit = st.selectbox(
            "View full audit log",
            [af.stem for af in audit_files[:50]],
        )
        if selected_audit:
            af_path = audit_dir / f"{selected_audit}.json"
            if af_path.exists():
                audit_data = json.loads(af_path.read_text())
                st.json(audit_data)

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## LendFlow — Vehicle Loan Document Intelligence

    LendFlow automates the document review stage of vehicle loan underwriting.
    A loan officer uploads application documents; the pipeline classifies them,
    redacts PII, extracts structured fields via a local LLM, validates against
    RBI/NBFC credit policy, scores confidence, and routes to **APPROVE / ESCALATE / REJECT**.

    ### Architecture

    ```
    intake → pii → extract → policy → confidence → route → [HITL interrupt?] → audit
    ```

    | Node | Purpose |
    |------|---------|
    | intake | Keyword-based doc type classification |
    | pii | Regex PII redaction (Aadhaar, PAN, IFSC, mobile) |
    | extract | LLM-based structured field extraction (Gemma 3 4B) |
    | policy | Deterministic rule engine (5 NBFC rules) |
    | confidence | Field-level uncertainty scoring |
    | route | APPROVE / ESCALATE / REJECT decision |
    | audit | Immutable JSON audit log |

    ### Tech Stack

    - **LangGraph** — stateful DAG with HITL interrupt/resume
    - **LM Studio** — privacy-first local LLM (Gemma 3 4B, MLX)
    - **ChromaDB + BM25** — hybrid RAG for policy retrieval
    - **FastAPI** — REST API with /process, /review, /audit endpoints
    - **Pydantic v2** — typed extraction schemas per doc type
    - **Python 3.9** — fully compatible

    ### Design Principles

    - **Privacy-first**: no PII leaves the machine; local LLM only
    - **Deterministic policy**: rules, not LLM, for pass/fail decisions
    - **Auditable**: every decision logged with full state snapshot
    - **Resumable**: LangGraph checkpointer enables HITL override
    """)

    st.info("Run `python scripts/run_eval.py` to benchmark on 20 synthetic applications.")
