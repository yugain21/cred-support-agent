# governance.py
"""
Task 15, Runtime layer: a per-request token/cost budget cap that rejects
an oversized request rather than silently exceeding budget.

Under MOCK_LLM there's no real token metering, so we approximate cost
with a simple, documented proxy: ~4 characters per token (a standard
rough heuristic), multiplied by a flat per-token cost, capped per request.
"""

CHARS_PER_TOKEN_ESTIMATE = 4
COST_PER_TOKEN_USD = 0.000002  # arbitrary illustrative rate
MAX_REQUEST_TOKENS = 500
MAX_REQUEST_COST_USD = MAX_REQUEST_TOKENS * COST_PER_TOKEN_USD


class BudgetExceededError(Exception):
    def __init__(self, estimated_tokens: int, estimated_cost: float):
        self.estimated_tokens = estimated_tokens
        self.estimated_cost = estimated_cost
        super().__init__(
            f"Request rejected: estimated {estimated_tokens} tokens "
            f"(${estimated_cost:.6f}) exceeds the per-request cap of "
            f"{MAX_REQUEST_TOKENS} tokens (${MAX_REQUEST_COST_USD:.6f})."
        )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def enforce_budget(query: str) -> dict:
    """
    Raises BudgetExceededError if the request is oversized; otherwise
    returns the estimate so it can be logged.
    """
    tokens = estimate_tokens(query)
    cost = tokens * COST_PER_TOKEN_USD
    if tokens > MAX_REQUEST_TOKENS:
        raise BudgetExceededError(tokens, cost)
    return {"estimated_tokens": tokens, "estimated_cost_usd": cost}
