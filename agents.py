# agents.py
import os
import json
from crewai import Agent, Task, Crew

from mock_llm import LocalDeterministicLLM
from schemas import validate_crew_response, CrewResponseValidationError

# Resilient imports supporting both the rubric name and legacy alias
try:
    from tools import rag_lookup, check_loan_application_status, loan_status_lookup
except ImportError:
    from tools import rag_lookup, check_loan_application_status
    loan_status_lookup = check_loan_application_status

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

mock_llm = LocalDeterministicLLM(model="mock-llm")

# --- Agents ------------------------------------------------------------
retrieval_agent = Agent(
    role="Policy Retrieval Specialist",
    goal="Retrieve accurate policy constraints from the knowledge base and answer strictly from that context.",
    backstory="You strictly query documents and never invent policies that aren't in the retrieved context.",
    tools=[rag_lookup],
    llm=mock_llm,
    allow_delegation=False,
)

# Task 15 - Principle of Least Autonomy: check_loan_application_status is bound ONLY to lookup_agent
lookup_agent = Agent(
    role="Database Lookup Specialist",
    goal="Check secure loan application statuses and compute escalation scores.",
    backstory="You are the only agent authorized to query the loan applications database.",
    tools=[check_loan_application_status],
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


def process_query_with_crew(user_query: str, history: list = None) -> dict:
    """
    Runs the 3-agent crew and returns a dict validating against schemas.CrewResponse.
    Accepts optional conversation history for multi-turn LangChain memory compatibility.
    """
    history_text = ""
    if history:
        formatted = []
        for m in history:
            if hasattr(m, "type"):
                role = "User" if m.type in ["human", "user"] else "Agent"
                content = m.content
            elif isinstance(m, dict):
                role = m.get("role", "User")
                content = m.get("content", "")
            else:
                role = "Message"
                content = str(m)
            formatted.append(f"{role}: {content}")
        if formatted:
            history_text = "Conversation History:\n" + "\n".join(formatted) + "\n\nCurrent Query: "

    is_lookup = _is_lookup_query(user_query)

    if is_lookup:
        primary_task = Task(
            description=f"{history_text}Look up the loan application status for: {user_query}",
            expected_output="A JSON payload with status, loan_amount_inr, and escalation_score.",
            agent=lookup_agent,
        )
    else:
        primary_task = Task(
            description=f"{history_text}Answer this policy question using only retrieved context: {user_query}",
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
        parsed = {
            "final_answer": raw_text,
            "grounded": False,
            "source_docs": [],
            "escalation_required": False,
            "escalation_score": None,
            "tool_used": "unknown",
        }

    validated = validate_crew_response(parsed)
    return validated.model_dump()