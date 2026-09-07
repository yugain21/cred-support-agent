# schemas.py
"""
Task 9: structured output schema every crew response must conform to.
Every call site validates the composer's output against this model
before returning it to the caller (see agents.py: process_query_with_crew).
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


class CrewResponse(BaseModel):
    final_answer: str = Field(..., description="The answer surfaced to the end user.")
    grounded: bool = Field(..., description="Whether the answer is backed by retrieved policy context.")
    source_docs: List[str] = Field(default_factory=list, description="Parent doc IDs the answer draws from.")
    escalation_required: bool = Field(default=False)
    escalation_score: Optional[float] = Field(default=None)
    tool_used: str = Field(..., description="'rag_lookup' or 'loan_status_lookup' or 'none'")


def validate_crew_response(raw: dict) -> CrewResponse:
    """Raises pydantic.ValidationError if `raw` doesn't conform."""
    return CrewResponse(**raw)


# Re-exported so callers can catch a single, obvious exception type.
CrewResponseValidationError = ValidationError
