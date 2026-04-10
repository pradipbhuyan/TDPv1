from typing import TypedDict, Any
import time

from langgraph.graph import StateGraph, END

from core import (
    detect_document_type,
    extract_structured_json,
    get_current_metrics_snapshot,
    diff_metrics_snapshot,
    generate_technical_summary_markdown,
    generate_technical_report_data,
    build_technical_architecture_pdf,
)


class IDPState(TypedDict, total=False):
    text: str
    filename: str
    template: bytes
    progress: Any
    event_callback: Any
    ocr_used: bool
    extraction_mode: str
    exception_reason: str

    doc_type: str
    data: dict
    result: dict
    error: str
    step_metrics: list
    validation: dict
    confidence: dict


def safe_progress(state: IDPState, percent: int, message: str):
    progress = state.get("progress")
    if progress:
        try:
            progress(percent, message)
        except Exception:
            pass


def emit_agent_event(state: IDPState, agent: str, status: str, message: str):
    callback = state.get("event_callback")
    if callback:
        try:
            callback(agent, status, message)
        except Exception:
            pass


def add_step_metric(state: IDPState, step_name: str, started_at: float, before: dict, note: str = ""):
    after = get_current_metrics_snapshot()
    diff = diff_metrics_snapshot(before, after)

    if "step_metrics" not in state or state["step_metrics"] is None:
        state["step_metrics"] = []

    state["step_metrics"].append({
        "step": step_name,
        "duration_sec": round(time.time() - started_at, 2),
        "tokens": diff.get("tokens", 0),
        "input_tokens": diff.get("input_tokens", 0),
        "output_tokens": diff.get("output_tokens", 0),
        "cost": round(diff.get("cost", 0.0), 6),
        "calls": diff.get("calls", 0),
        "note": note,
    })


def detect_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Classification Agent", "running", "Classifying technical document")
    safe_progress(state, 40, "Classification Agent — classifying technical document")

    state["doc_type"] = detect_document_type(state.get("text", ""))

    emit_agent_event(
        state,
        "Classification Agent",
        "done",
        f"Document identified as {state.get('doc_type', 'technical_doc')}"
    )

    add_step_metric(state, "Detect document type", started_at, before, state.get("doc_type", "technical_doc"))
    return state


def extract_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Structuring Agent", "running", "Extracting technical document structure")
    safe_progress(state, 60, "Structuring Agent — extracting technical structure")

    state["data"] = extract_structured_json(state.get("text", ""), "technical_doc")

    emit_agent_event(state, "Structuring Agent", "done", "Technical document structure extracted")

    add_step_metric(state, "Extract technical document data", started_at, before, "technical_doc")
    return state


def technical_doc_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    data = state.get("data") or {}
    report_data = generate_technical_report_data(data)

    summary_md = generate_technical_summary_markdown(report_data)
    summary_pdf = build_technical_architecture_pdf(report_data)

    emit_agent_event(state, "Output Agent", "running", "Preparing architecture report")
    safe_progress(state, 85, "Output Agent — preparing architecture report")

    title = (report_data.get("document_title") or "technical_architecture_report").strip()
    safe_name = "".join(ch for ch in title if ch not in '\\/*?:"<>|').strip() or "technical_architecture_report"

    state["result"] = {
        "type": "technical_doc",
        "data": report_data,
        "summary_markdown": summary_md,
        "file_name": f"{safe_name}.md",
        "pdf_file_name": f"{safe_name}.pdf",
    }

    emit_agent_event(state, "Output Agent", "done", "Architecture report prepared")
    safe_progress(state, 95, "Output Agent — architecture report ready")

    add_step_metric(state, "Prepare architecture report", started_at, before, state["result"]["file_name"])
    return state


def build_graph():
    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("extract", extract_node)
    builder.add_node("technical_doc", technical_doc_node)

    builder.set_entry_point("detect")
    builder.add_edge("detect", "extract")
    builder.add_edge("extract", "technical_doc")
    builder.add_edge("technical_doc", END)

    return builder.compile()
