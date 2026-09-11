# guardrails.py
"""
Input guardrails: PII masking + prompt-injection detection.
Output guardrail: groundedness check on RAG responses.

Only fixed-format PII is masked (PAN, Aadhaar, account numbers) —
names and income figures are free text with no reliable pattern.
"""
import re

PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{9,18}\b")

PROMPT_INJECTION_PATTERNS = [
    r"ignore (the )?(previous|prior|above) instructions",
    r"disregard (your|the) (rules|instructions|guidelines)",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|system message)",
    r"act as (if you (are|were)|a different)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"do anything now",
    r"override (your|the) (safety|guardrails|rules)",
]
_INJECTION_RE = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)

OUT_OF_SCOPE_TOPICS = ["cake", "recipe", "weather", "sports", "football", "movie"]


def mask_pii(text: str) -> str:
    """
    Masks PAN, Aadhaar, and account-number-shaped substrings.
    A bare 12-digit run matches Aadhaar before account number.
    """
    masked = PAN_PATTERN.sub("[PAN_REDACTED]", text)
    masked = AADHAAR_PATTERN.sub("[AADHAAR_REDACTED]", masked)
    masked = ACCOUNT_NUMBER_PATTERN.sub("[ACCOUNT_NUMBER_REDACTED]", masked)
    return masked


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def validate_input_guardrails(query: str) -> dict:
    """
    Runs prompt-injection detection (blocks on match) and PII
    masking (redacts but lets the request through).
    """

    if detect_prompt_injection(query):
        return {
            "safe": False,
            "reason": "Prompt-injection attempt detected in the request.",
            "masked_query": mask_pii(query),
        }

    masked_query = mask_pii(query)
    return {"safe": True, "reason": "Passed input guardrails.", "masked_query": masked_query}


def validate_output_groundedness(rag_result: dict, is_rag_query: bool) -> dict:
    """
    Blocks RAG answers below the similarity threshold. Lookup-tool
    queries are grounded in the DB record itself, so they're exempt.
    """
    if not is_rag_query:
        return {"safe": True, "reason": "Not a RAG query; groundedness check not applicable."}

    if not rag_result.get("grounded", False):
        return {
            "safe": False,
            "reason": "Output-side groundedness check failed: retrieved context "
                      "did not support the question (below similarity threshold).",
        }
    return {"safe": True, "reason": "Answer is grounded in retrieved context."}


def scrub_for_log(text: str) -> str:
    return mask_pii(text)
