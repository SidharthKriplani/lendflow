"""
LendFlow — LangGraph Pipeline
7-node DAG:
  intake → pii → extract → policy → confidence → route → [interrupt?] → audit

Human-in-the-loop: graph interrupts before audit_node when
human_review_required == True, waiting for /review endpoint input.
"""
from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from pipeline.state import LendFlowState
from pipeline.nodes.intake     import intake_node
from pipeline.nodes.pii        import pii_node
from pipeline.nodes.extract    import extract_node
from pipeline.nodes.policy     import policy_node
from pipeline.nodes.confidence import confidence_node
from pipeline.nodes.route      import route_node
from pipeline.nodes.audit      import audit_node
import config


# ── Conditional edge functions ────────────────────────────────────────────────

def should_continue_after_intake(state: LendFlowState) -> str:
    """Stop early if intake failed (empty doc or unknown type)."""
    if state.get("error"):
        return "audit"
    if state.get("doc_type") == config.DOC_UNKNOWN:
        return "audit"
    return "pii"


def should_continue_after_extract(state: LendFlowState) -> str:
    """Stop early if extraction failed completely."""
    if state.get("error"):
        return "audit"
    if not state.get("extracted_fields"):
        return "audit"
    return "policy"


def should_interrupt_for_review(state: LendFlowState) -> str:
    """
    After routing: if human review is required, interrupt before audit.
    The graph will pause here — FastAPI /review endpoint resumes it.
    """
    if state.get("human_review_required", False):
        return "human_review"   # this is the interrupt node
    return "audit"


# ── Human review pass-through node ────────────────────────────────────────────

def human_review_node(state: LendFlowState) -> dict:
    """
    This node is where the graph INTERRUPTS.
    When the graph resumes (after /review endpoint provides override),
    the state will have human_override populated and this node just passes through.
    """
    override = state.get("human_override")
    if override:
        # Apply the human's decision
        return {
            "routing_decision": override.get("decision", state.get("routing_decision")),
            "reason_codes": [
                f"Human override by {override.get('reviewer_id', 'reviewer')}: "
                f"{override.get('justification', 'Manual review')}"
            ],
        }
    return {}  # No override — original routing stands


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the LendFlow LangGraph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to MemorySaver (in-memory, for development).
                      Use SqliteSaver or RedisSaver for production.

    Returns:
        Compiled LangGraph application.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(LendFlowState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    workflow.add_node("intake",       intake_node)
    workflow.add_node("pii",          pii_node)
    workflow.add_node("extract",      extract_node)
    workflow.add_node("policy",       policy_node)
    workflow.add_node("confidence",   confidence_node)
    workflow.add_node("route",        route_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("audit",        audit_node)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("intake")

    # ── Edges ──────────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "intake",
        should_continue_after_intake,
        {"pii": "pii", "audit": "audit"},
    )
    workflow.add_edge("pii", "extract")
    workflow.add_conditional_edges(
        "extract",
        should_continue_after_extract,
        {"policy": "policy", "audit": "audit"},
    )
    workflow.add_edge("policy",     "confidence")
    workflow.add_edge("confidence", "route")
    workflow.add_conditional_edges(
        "route",
        should_interrupt_for_review,
        {"human_review": "human_review", "audit": "audit"},
    )
    workflow.add_edge("human_review", "audit")
    workflow.add_edge("audit", END)

    # ── Compile with interrupt_before human_review ─────────────────────────
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # Graph pauses here for HITL
    )
    return app


# ── Module-level compiled graph (singleton for production use) ────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Convenience run function ──────────────────────────────────────────────────

def run_pipeline(raw_text: str, application_id: Optional[str] = None,
                 thread_id: Optional[str] = None) -> dict:
    """
    Run the LendFlow pipeline on a document.

    Args:
        raw_text:       Raw document text (before PII redaction).
        application_id: Optional pre-assigned ID.
        thread_id:      LangGraph thread ID for checkpointing.

    Returns:
        Final pipeline state dict.
    """
    import uuid
    thread_id = thread_id or str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

    initial_state: LendFlowState = {
        "raw_text":       raw_text,
        "application_id": application_id,
    }

    graph = get_graph()
    final_state = None

    for event in graph.stream(initial_state, config=config_dict):
        node_name = list(event.keys())[0]
        final_state = event[node_name]

    # Retrieve full state after completion
    full_state = graph.get_state(config_dict)
    return dict(full_state.values) if full_state else {}
