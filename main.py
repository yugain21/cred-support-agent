import time
import uuid
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import rag_core
import cache
import governance
import guardrails
from memory import ask_with_memory, reset_session
from knowledge_base import DOCUMENTS
import config

LOG_PATH = os.path.join(os.path.dirname(__file__), "api_logs.jsonl")

app = FastAPI(title="Cred Domain Support Agent")

@app.on_event("startup")
def startup_calibration():
    fixed_col, sent_col = rag_core.build_indices()
    app.state.fixed_col = fixed_col
    app.state.sent_col = sent_col 

    in_scope = ["What are the late fees?", "What is the interest rate on savings?", "How do I dispute a fraud charge?"]
    out_scope = ["How do I bake a chocolate cake?", "What's the weather like today?"]
    calibration = rag_core.calibrate_threshold(sent_col, in_scope, out_scope)
    config.set_calibrated_threshold(calibration["threshold"])
    app.state.calibration = calibration

def _log_request(trace_id: str, query: str, duration: float, status: str) -> None:
    entry = {
        "trace_id": trace_id,
        "duration_seconds": round(duration, 4),
        "query_masked": guardrails.scrub_for_log(query),
        "status": status,
        "timestamp": time.time(),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

class QueryRequest(BaseModel):
    query: str

class AddDocumentRequest(BaseModel):
    doc_id: str
    text: str

def _handle_query(query: str, session_id: str) -> dict:
    trace_id = str(uuid.uuid4())
    start = time.time()

    try:
        governance.enforce_budget(query)
    except governance.BudgetExceededError as e:
        _log_request(trace_id, query, time.time() - start, "rejected_budget")
        return {"status": "error", "trace_id": trace_id, "response": str(e)}

    try:
        input_check = guardrails.validate_input_guardrails(query)
        if not input_check["safe"]:
            _log_request(trace_id, query, time.time() - start, "rejected_guardrail")
            return {"status": "error", "trace_id": trace_id, "response": input_check["reason"]}

        def compute():
            return ask_with_memory(session_id, input_check["masked_query"])

        raw_result, was_cache_hit = cache.get_or_compute(input_check["masked_query"], compute)

        # SAFE NORMALIZATION: Ensure result is always a dictionary
        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                result = {"output": raw_result, "tool_used": "unknown", "grounded": True}
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            result = {"output": str(raw_result), "tool_used": "unknown", "grounded": True}

        is_rag_query = result.get("tool_used") == "rag_lookup"
        rag_shaped_result = {"grounded": result.get("grounded", False)}
        output_check = guardrails.validate_output_groundedness(rag_shaped_result, is_rag_query)
        
        if not output_check["safe"]:
            _log_request(trace_id, query, time.time() - start, "rejected_ungrounded")
            return {"status": "error", "trace_id": trace_id, "response": output_check["reason"]}

        _log_request(trace_id, query, time.time() - start, "success")
        return {
            "status": "success",
            "trace_id": trace_id,
            "response": result,
            "cache_hit": was_cache_hit,
        }
    except Exception as e:
        _log_request(trace_id, query, time.time() - start, "internal_error")
        return {
            "status": "error",
            "trace_id": trace_id,
            "response": f"Internal pipeline execution error: {str(e)}"
        }

@app.post("/ask")
def ask_agent(request: QueryRequest):
    return _handle_query(request.query, session_id=str(uuid.uuid4()))

@app.post("/add-document")
async def add_document(request: AddDocumentRequest):
    DOCUMENTS[request.doc_id] = request.text
    fixed_col, sent_col = rag_core.build_indices(force_rebuild=True)
    app.state.fixed_col = fixed_col
    app.state.sent_col = sent_col
    return {"status": "success", "doc_id": request.doc_id, "total_documents": len(DOCUMENTS)}

@app.get("/cache-stats")
async def cache_stats():
    return cache.stats()

@app.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            query = await websocket.receive_text()
            response = _handle_query(query, session_id)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        reset_session(session_id)