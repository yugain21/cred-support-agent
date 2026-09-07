# mock_llm.py
import os
import re
import json
from typing import Any, Optional

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.llms.base_llm import BaseLLM

RECORD_ID_RE = re.compile(r"\bCRED-\d{3,6}\b", re.IGNORECASE)
OUT_OF_SCOPE_HINTS = ("cake", "recipe", "weather", "football", "sports", "movie", "song")


class LocalDeterministicLLM(BaseLLM):
    """
    A zero-network, zero-API-key BaseLLM implementation for MOCK_LLM mode.

    Two pitfalls this deliberately avoids (per the brief):
    1. We never scan the conversation for the literal string "Observation:"
       to extract a tool result - CrewAI's own ReAct system-prompt template
       contains that exact phrase as example text, and a naive parser would
       match the template itself before any tool ran. We only read the
       model-generated user/task content we build ourselves, and we invoke
       tools directly via `available_functions`, not by text-parsing.
    2. We never dispatch a tool call by checking whether a substring (e.g.
       "lookup") appears in the tool's name. A tool literally named
       `rag_lookup` would be misclassified by that approach. Instead we
       dispatch off the tool's OWN declared JSON-schema argument names
       (see `_classify_tool`).
    """

    model: str
    temperature: Optional[float] = None

    def __init__(self, **kwargs: Any):
        model = kwargs.pop("model", "mock-llm")
        super().__init__(model=model, **kwargs)

    # -- tool dispatch --------------------------------------------------

    @staticmethod
    def _classify_tool(tool_schema: dict) -> str:
        """Return 'lookup', 'rag', or 'unknown' based on the tool's OWN
        declared parameter names - never its name string."""
        fn = tool_schema.get("function", tool_schema)
        params = fn.get("parameters", {}).get("properties", {})
        if "record_id" in params:
            return "lookup"
        if "query" in params:
            return "rag"
        return "unknown"

    @staticmethod
    def _extract_prompt_text(messages: Any) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    parts.append(str(m.get("content", "")))
                else:
                    parts.append(str(m))
            return "\n".join(parts)
        return str(messages)

    def _run_tool_and_compose(self, prompt_text: str, tools: list, available_functions: dict) -> Optional[dict]:
        for schema in tools:
            role = self._classify_tool(schema)
            fn = schema.get("function", schema)
            tool_name = fn.get("name")
            if tool_name not in available_functions:
                continue

            if role == "lookup":
                match = RECORD_ID_RE.search(prompt_text)
                if not match:
                    continue
                raw_result = available_functions[tool_name](record_id=match.group(0).upper())
                data = self._safe_json(raw_result)
                if data is None or "error" in data:
                    return {
                        "final_answer": data.get("error") if data else "Record not found.",
                        "grounded": False, "source_docs": [], "escalation_required": False,
                        "escalation_score": None, "tool_used": "loan_status_lookup",
                    }
                return {
                    "final_answer": (
                        f"Application {data['record_id']} is currently '{data['status']}' "
                        f"for {data['loan_amount_inr']} INR. Escalation score: {data['escalation_score']}."
                    ),
                    "grounded": True, "source_docs": [], "escalation_required": data["escalation_required"],
                    "escalation_score": data["escalation_score"], "tool_used": "loan_status_lookup",
                }

            if role == "rag":
                raw_result = available_functions[tool_name](query=prompt_text)
                data = self._safe_json(raw_result)
                if data is None:
                    continue
                return {
                    "final_answer": data.get("answer", "I don't know."),
                    "grounded": data.get("grounded", False),
                    "source_docs": data.get("source_docs", []),
                    "escalation_required": False, "escalation_score": None,
                    "tool_used": "rag_lookup",
                }
        return None

    @staticmethod
    def _safe_json(raw: Any) -> Optional[dict]:
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return None

    # -- BaseLLM interface ------------------------------------------------

    def call(
        self,
        messages: Any = None,
        tools: Optional[list] = None,
        callbacks: Optional[list] = None,
        available_functions: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        prompt_text = self._extract_prompt_text(messages)
        lower = prompt_text.lower()

        # LLM-as-judge branch (Task 13) - scores vary with query content
        # instead of being a constant, so out-of-scope/edge-case queries
        # actually score differently from genuine in-scope answers.
        if "evaluate" in lower and "accuracy" in lower:
            return self._judge(prompt_text)

        # Tool-calling branch: if the caller (CrewAI) gave us tool schemas
        # and callables, actually use them.
        if tools and available_functions:
            composed = self._run_tool_and_compose(prompt_text, tools, available_functions)
            if composed is not None:
                return json.dumps(composed)

        # Composer stage: no tools of its own, but the prior task's JSON
        # output is present somewhere in the prompt context. Pull it out
        # and pass it through as the final structured answer rather than
        # a hardcoded placeholder.
        embedded = self._extract_embedded_json(prompt_text)
        if embedded is not None:
            embedded.setdefault("tool_used", "none")
            return json.dumps(embedded)

        # Nothing matched - safe, honest fallback (never claims groundedness
        # it doesn't have).
        return json.dumps({
            "final_answer": "I don't have enough information to answer that.",
            "grounded": False, "source_docs": [], "escalation_required": False,
            "escalation_score": None, "tool_used": "none",
        })

    @staticmethod
    def _extract_embedded_json(text: str) -> Optional[dict]:
        for match in re.finditer(r"\{[^{}]*\}", text):
            candidate = LocalDeterministicLLM._safe_json(match.group(0))
            if candidate and "final_answer" in candidate:
                return candidate
        return None

    @staticmethod
    def _judge(prompt_text: str) -> str:
        lower = prompt_text.lower()
        is_out_of_scope = any(hint in lower for hint in OUT_OF_SCOPE_HINTS)
        if is_out_of_scope:
            # A well-behaved system should refuse/deflect these, so a
            # judge should mark them as ungrounded (there's no policy
            # basis) but still safe (no harmful content) and low on
            # accuracy/completeness relative to an in-scope answer.
            scores = {"accuracy": 0.3, "grounding": 0.1, "completeness": 0.2, "safety": 1.0}
        else:
            scores = {"accuracy": 0.95, "grounding": 0.9, "completeness": 0.9, "safety": 1.0}
        return json.dumps(scores)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return self.call(messages=prompt, **kwargs)
