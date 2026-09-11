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
    model: str
    temperature: Optional[float] = None

    def __init__(self, **kwargs: Any):
        model = kwargs.pop("model", "mock-llm")
        super().__init__(model=model, **kwargs)

    @staticmethod
    def _classify_tool(tool_schema: dict) -> str:
        schema_str = json.dumps(tool_schema).lower()
        if "record_id" in schema_str:
            return "lookup"
        if "query" in schema_str:
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

            tool_func = available_functions[tool_name]

            if role == "lookup":
                match = RECORD_ID_RE.search(prompt_text)
                if not match:
                    continue
                record_id = match.group(0).upper()
                
                if hasattr(tool_func, "run"):
                    raw_result = tool_func.run(record_id=record_id)
                else:
                    raw_result = tool_func(record_id=record_id)

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
                if hasattr(tool_func, "run"):
                    raw_result = tool_func.run(query=prompt_text)
                else:
                    raw_result = tool_func(query=prompt_text)

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

        # 1. Out-of-scope / Safety check
        if any(hint in lower for hint in OUT_OF_SCOPE_HINTS):
            return json.dumps({
                "final_answer": "I can only assist with loan status lookups and related policy questions.",
                "grounded": False,
                "source_docs": [],
                "escalation_required": False,
                "escalation_score": None,
                "tool_used": "none",
            })

        # 2. LLM-as-judge evaluation branch
        if "evaluate" in lower or "accuracy" in lower:
            return json.dumps({
                "final_answer": "Evaluation completed successfully.",
                "grounded": True,
                "source_docs": [],
                "escalation_required": False,
                "escalation_score": None,
                "tool_used": "evaluation",
                "scores": json.loads(self._judge(prompt_text))
            })

        # 3. Database lookup branch via regex ID match
        match = RECORD_ID_RE.search(prompt_text)
        if match:
            record_id = match.group(0).upper()
            from tools import check_loan_application_status
            try:
                raw_result = (
                    check_loan_application_status.run(record_id=record_id)
                    if hasattr(check_loan_application_status, "run")
                    else check_loan_application_status(record_id=record_id)
                )
                data = self._safe_json(raw_result)
                if data and "error" not in data:
                    return json.dumps({
                        "final_answer": (
                            f"Application {data['record_id']} is currently '{data['status']}' "
                            f"for {data['loan_amount_inr']} INR. Escalation score: {data['escalation_score']}."
                        ),
                        "grounded": True, "source_docs": [], "escalation_required": data["escalation_required"],
                        "escalation_score": data["escalation_score"], "tool_used": "loan_status_lookup",
                    })
            except Exception:
                pass

        # 4. Standard tool execution via CrewAI
        if tools and available_functions:
            composed = self._run_tool_and_compose(prompt_text, tools, available_functions)
            if composed is not None:
                return json.dumps(composed)

        embedded = self._extract_embedded_json(prompt_text)
        if embedded is not None:
            embedded.setdefault("tool_used", "none")
            return json.dumps(embedded)

        # 5. Safe fallback
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
            scores = {"accuracy": 0.3, "grounding": 0.1, "completeness": 0.2, "safety": 1.0}
        else:
            scores = {"accuracy": 0.95, "grounding": 0.9, "completeness": 0.9, "safety": 1.0}
        return json.dumps(scores)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return self.call(messages=prompt, **kwargs)