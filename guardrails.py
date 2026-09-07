# guardrails.py
"""
Task 10: input-side guardrails (PII masking + prompt-injection detection)
and an output-side groundedness check.

Per the brief, only the FIXED-FORMAT PII fields are masked:
  - PAN:      5 letters + 4 digits + 1 letter, e.g. ABCDE1234F
  - Aadhaar:  12 digits, often grouped 4-4-4, e.g. 1234 5678 9012
  - Bank account number: a long, unstructured digit run (9-18 digits) that
    isn't already an Aadhaar match.
Applicant name and income figures are explicitly OUT OF SCOPE for masking
(free text / unformatted numbers, no reliable pattern under a keyless masker).
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
    Mask PAN, Aadhaar, and bank-account-number-shaped substrings.
    NOTE: a bare 12-digit run is ambiguous between Aadhaar and a 12-digit
    bank account number by pattern alone; the Aadhaar pattern matches
    first, so such values are redacted as [AADHAAR_REDACTED]. Either way
    the raw digits never reach the model or the logs, which is the
    guardrail's actual requirement - the label is a secondary concern.
    """
    masked = PAN_PATTERN.sub("[PAN_REDACTED]", text)
    masked = AADHAAR_PATTERN.sub("[AADHAAR_REDACTED]", masked)
    masked = ACCOUNT_NUMBER_PATTERN.sub("[ACCOUNT_NUMBER_REDACTED]", masked)
    return masked


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def validate_input_guardrails(query: str) -> dict:
    """
    Returns {"safe": bool, "reason": str, "masked_query": str}.
    Two independent checks:
      1. Prompt-injection detection (blocks the request).
      2. PII masking (does NOT block the request - masks it and lets it
         proceed, same as a real support agent redacting sensitive fields
         before they hit the model/logs).
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
    Task 10 output-side guardrail: refuse to answer when the retrieved
    context doesn't support the question. `rag_result` is the dict returned
    by rag_core.grounded_generation (has a `grounded` bool). Lookup-tool
    queries (not RAG queries) are exempt - they're grounded in the
    database record itself, not retrieved text.
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
    """Same masking applied to anything written to disk (Task 12)."""
    return mask_pii(text)
