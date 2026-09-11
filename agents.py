# agents.py
import os
import json
from crewai import Agent, Task, Crew

from mock_llm import LocalDeterministicLLM
from tools import rag_lookup, loan_status_lookup
from schemas import validate_crew_response, CrewResponseValidationError

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

mock_llm = LocalDeterministicLLM(model="mock-llm")

# --- Agents ------------------------------------------------------------
# CrewAI's Agent.llm field accepts either a model string or a BaseLLM
# instance directly - passing our mock in at construction time is the
# documented way to use a custom BaseLLM subclass, so no post-hoc
# attribute-swapping hack is needed.

retrieval_agent = Agent(
    role="Policy Retrieval Specialist",
    goal="Retrieve accurate policy constraints from the knowledge base and answer strictly from that context.",
    backstory="You strictly query documents and never invent policies that aren't in the retrieved context.",
    tools=[rag_lookup],
    llm=mock_llm,
    allow_delegation=False,
)

# Principle of least autonomy: check_loan_application_status
# (wired here as `loan_status_lookup`) is bound ONLY to lookup_agent's
# tools array. retrieval_agent and composer_agent below never receive a
# reference to it, so they are structurally incapable of calling it -
# there is no code path by which they could invoke a DB lookup during a
# generic policy chat.
lookup_agent = Agent(
    role="Database Lookup Specialist",
    goal="Check secure loan application statuses and compute escalation scores.",
    backstory="You are the only agent authorized to query the loan applications database.",
    tools=[loan_status_lookup],
    llm=mock_llm,
    allow_delegation=False,
)

composer_agent = Agent(
    role="Response Composer",
    goal="Format retrieved data into one final, structured JSON payload matching the CrewResponse schema.",
    backstory="You finalize support responses succinctly and never fabricate fields.",
    llm=mock_llm,
    allow_delegation=False,
)


def _is_lookup_query(query: str) -> bool:
    lowered = query.lower()
    return "cred-" in lowered or "application" in lowered or "record" in lowered


def process_query_with_crew(user_query: str) -> dict:
    """
    Runs the 3-agent crew and returns a dict that VALIDATES against
    schemas.CrewResponse. Raises CrewResponseValidationError if
    the composer's output doesn't conform - callers must handle that.
    """
    is_lookup = _is_lookup_query(user_query)

    if is_lookup:
        primary_task = Task(
            description=f"Look up the loan application status for: {user_query}",
            expected_output="A JSON payload with status, loan_amount_inr, and escalation_score.",
            agent=lookup_agent,
        )
    else:
        primary_task = Task(
            description=f"Answer this policy question using only retrieved context: {user_query}",
            expected_output="A JSON payload with the grounded answer and source_docs.",
            agent=retrieval_agent,
        )

    compose_task = Task(
        description="Synthesize the prior agent's JSON finding into one final CrewResponse-shaped JSON object "
                     "with fields: final_answer, grounded, source_docs, escalation_required, escalation_score, tool_used.",
        expected_output="A single valid JSON object matching the CrewResponse schema.",
        agent=composer_agent,
        context=[primary_task],
    )

    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[primary_task, compose_task],
        verbose=False,
    )

    raw_result = crew.kickoff()
    raw_text = str(raw_result)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Defensive fallback so a malformed composer response doesn't crash
        # the API layer - still explicit about what happened.
        parsed = {
            "final_answer": raw_text,
            "grounded": False, "source_docs": [], "escalation_required": False,
            "escalation_score": None, "tool_used": "unknown",
        }

    validated = validate_crew_response(parsed)  # raises CrewResponseValidationError on schema mismatch
    return validated.model_dump()
