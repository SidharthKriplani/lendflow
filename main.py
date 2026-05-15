"""
LendFlow FastAPI Application
POST /process  — run full pipeline on a raw document text
GET  /health   — liveness probe
GET  /audit/{audit_id} — retrieve audit log for a completed application
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from pipeline.graph import run_pipeline
import config

app = FastAPI(
    title="LendFlow",
    description="LLM-powered vehicle loan document decisioning pipeline",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    raw_text:       str   = Field(..., description="Raw document text (bank statement, salary slip, KYC, or vehicle report)")
    application_id: Optional[str] = Field(None, description="Optional caller-supplied application ID")
    thread_id:      Optional[str] = Field(None, description="LangGraph thread ID for checkpointing")


class ProcessResponse(BaseModel):
    application_id:      str
    audit_id:            Optional[str]
    doc_type:            Optional[str]
    routing_decision:    Optional[str]
    reason_codes:        list[str]
    human_review_required: bool
    uncertainty_band:    Optional[str]
    uncertainty_score:   Optional[float]
    pii_entity_count:    int
    error:               Optional[str]
    pipeline_version:    str


class HumanOverrideRequest(BaseModel):
    application_id: str
    thread_id:      str
    override_decision: str = Field(..., description="APPROVE | REJECT | ESCALATE")
    override_reason:   str = Field(..., description="Officer's override justification")
    officer_id:        str = Field(..., description="Reviewing officer identifier")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0",
            "llm_endpoint": config.LLM_BASE_URL,
            "llm_model":    config.LLM_MODEL}


@app.post("/process", response_model=ProcessResponse)
def process_document(req: ProcessRequest):
    """
    Run the full LendFlow pipeline on a raw document text.
    Returns routing decision, reason codes, and audit reference.
    """
    if not req.raw_text or not req.raw_text.strip():
        raise HTTPException(status_code=422, detail="raw_text must not be empty")

    result = run_pipeline(
        raw_text=req.raw_text,
        application_id=req.application_id,
        thread_id=req.thread_id,
    )

    return ProcessResponse(
        application_id=    result.get("application_id", "UNKNOWN"),
        audit_id=          result.get("audit_id"),
        doc_type=          result.get("doc_type"),
        routing_decision=  result.get("routing_decision"),
        reason_codes=      result.get("reason_codes", []),
        human_review_required= result.get("human_review_required", False),
        uncertainty_band=  result.get("uncertainty_band"),
        uncertainty_score= result.get("uncertainty_score"),
        pii_entity_count=  result.get("pii_entity_count", 0),
        error=             result.get("error"),
        pipeline_version=  result.get("pipeline_version", "1.0.0"),
    )


@app.post("/human-review/override")
def human_override(req: HumanOverrideRequest):
    """
    Submit a human officer's override decision for an escalated application.
    Resumes the paused LangGraph thread and records the override in the audit log.
    """
    from pipeline.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(MemorySaver())
    config_dict = {"configurable": {"thread_id": req.thread_id}}

    # Inject human override into state and resume
    override_update = {
        "human_override": {
            "decision":     req.override_decision,
            "reason":       req.override_reason,
            "officer_id":   req.officer_id,
        }
    }

    try:
        final_state = None
        for event in graph.stream(override_update, config=config_dict):
            for node_name, node_state in event.items():
                final_state = node_state
        return {
            "status":           "override_applied",
            "application_id":   req.application_id,
            "override_decision": req.override_decision,
            "audit_id":         final_state.get("audit_id") if final_state else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/{audit_id}")
def get_audit(audit_id: str):
    """Retrieve the audit log for a completed pipeline run."""
    audit_path = Path(config.AUDIT_LOG_DIR) / f"{audit_id}.json"
    if not audit_path.exists():
        raise HTTPException(status_code=404, detail=f"Audit log not found: {audit_id}")
    return JSONResponse(content=json.loads(audit_path.read_text(encoding="utf-8")))


@app.get("/audit")
def list_audits(limit: int = 20, offset: int = 0):
    """List recent audit logs."""
    audit_dir = Path(config.AUDIT_LOG_DIR)
    if not audit_dir.exists():
        return {"audits": [], "total": 0}
    files = sorted(audit_dir.glob("AUDIT-*.json"), reverse=True)
    total = len(files)
    page  = files[offset: offset + limit]
    audits = []
    for f in page:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            audits.append({
                "audit_id":         data.get("audit_id"),
                "application_id":   data.get("application_id"),
                "doc_type":         data.get("doc_type"),
                "routing_decision": data.get("routing_decision"),
                "timestamp":        data.get("timestamp"),
            })
        except Exception:
            pass
    return {"audits": audits, "total": total, "limit": limit, "offset": offset}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
