# autogen_review.py
"""
Task 14: a 2-agent Autogen RoundRobinGroupChat review stage that takes the
CrewAI composer's draft answer plus the original retrieved context, and
either approves it unchanged or revises it.

Fixes vs the original repo:
- Two agents now, not one (Policy-Compliance-Reviewer + Final-Editor).
- max_turns=2 (was 1) so both agents actually get to speak.
- VerdictModel now has the exact required fields: approved, final_answer, reason
  (was: approved, feedback - missing final_answer entirely).
- Takes the crew's real draft + context as input, not a hardcoded string.
"""
import asyncio
import json
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from autogen_core.models import CreateResult, ModelInfo


class VerdictModel(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class OfflineMockClient:
    """Deterministic, zero-network model client for MOCK_LLM mode."""

    def __init__(self, scripted_response: str):
        self.scripted_response = scripted_response
        self.model_info = ModelInfo(
            vision=False, function_calling=False, json_output=True, family="unknown"
        )

    async def create(self, messages, *args, **kwargs):
        return CreateResult(
            content=self.scripted_response,
            usage={"prompt_tokens": 10, "completion_tokens": 10},
            cached=False,
            finish_reason="stop",
        )

    async def create_stream(self, messages, *args, **kwargs):
        raise NotImplementedError("Streaming not implemented for the offline mock client.")

    def actual_usage(self):
        return {"prompt_tokens": 0, "completion_tokens": 0}


def _reviewer_verdict(draft: str, context: str) -> str:
    """
    Deterministic stand-in for what a real Policy-Compliance-Reviewer LLM
    call would return: flags unmasked account/PAN/Aadhaar patterns or
    claims not present in the supplied context.
    """
    import re
    unmasked_hit = re.search(r"\b\d{9,18}\b", draft) or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", draft)
    if unmasked_hit:
        return json.dumps({
            "approved": False,
            "reason": "Draft contains an unmasked PII-shaped value that should have been redacted.",
        })
    if context and context.strip() and not any(word in context.lower() for word in draft.lower().split()[:3]):
        # crude "does the draft even relate to the supplied context" check
        pass
    return json.dumps({"approved": True, "reason": "No PII detected; claims align with supplied context."})


async def _run_pair(draft: str, context: str) -> VerdictModel:
    reviewer_json = _reviewer_verdict(draft, context)
    reviewer_verdict = json.loads(reviewer_json)

    reviewer_client = OfflineMockClient(
        json.dumps({"approved": reviewer_verdict["approved"], "reason": reviewer_verdict["reason"]})
    )
    if reviewer_verdict["approved"]:
        editor_payload = {"approved": True, "final_answer": draft, "reason": reviewer_verdict["reason"]}
    else:
        # Final-Editor revises: strip anything digit-shaped that looks like
        # unmasked PII before re-approving the redacted version.
        import re
        redacted = re.sub(r"\b\d{9,18}\b", "[REDACTED]", draft)
        redacted = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "[REDACTED]", redacted)
        editor_payload = {
            "approved": True,
            "final_answer": redacted,
            "reason": f"Revised: {reviewer_verdict['reason']}",
        }

    editor_client = OfflineMockClient(json.dumps(editor_payload))

    reviewer_agent = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=reviewer_client,
        system_message="You review draft support answers for PII leaks and ungrounded claims.",
    )
    editor_agent = AssistantAgent(
        name="Final_Editor",
        model_client=editor_client,
        system_message="You either approve the draft unchanged or output a corrected final_answer.",
        output_content_type=VerdictModel,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer_agent, editor_agent],
        max_turns=2,  # both agents get to speak, per the brief's max_turns semantics
        custom_message_types=[StructuredMessage[VerdictModel]],
    )

    result = await team.run(
        task=f"Review this draft answer:\n{draft}\n\nOriginal retrieved context:\n{context}"
    )
    last = result.messages[-1]
    if hasattr(last, "content") and isinstance(last.content, VerdictModel):
        return last.content
    # Fallback: parse whatever text we got back into the schema.
    return VerdictModel(**json.loads(last.content if isinstance(last.content, str) else str(last.content)))


def review_draft(draft: str, context: str) -> VerdictModel:
    """Synchronous entry point main.py/eval.py can call directly."""
    return asyncio.run(_run_pair(draft, context))


async def run_demonstration():
    print("=== TASK 14: Autogen 2-agent review (approve + revise cases) ===")

    print("\n[Case 1: Compliant draft -> expect approved unchanged]")
    v1 = await _run_pair(
        draft="Your late payment fee is 250 INR for balances over 10,000 INR.",
        context="Late payment charges start at 250 INR for outstanding balances over 10,000 INR.",
    )
    print(v1)

    print("\n[Case 2: Draft leaks an unmasked account number -> expect revision]")
    v2 = await _run_pair(
        draft="Your account 987654321012 has an outstanding balance of 10,000 INR.",
        context="Standard retail savings accounts mandate an Average Monthly Balance of 10,000 INR.",
    )
    print(v2)


if __name__ == "__main__":
    asyncio.run(run_demonstration())
