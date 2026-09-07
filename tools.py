"""
The two tools the crew actually uses. Both wrap REAL systems built in
Part 1 (rag_core.py, dataset.json) instead of returning hardcoded strings.

Task 6: check_loan_application_status computes a designed escalation_score
combining flagged_for_fraud_review with a normalized recency signal.
"""
import json
import os

try:
    from crewai.tools import tool
except ImportError:
    from langchain_core.tools import tool

import rag_core

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")

# --- Escalation formula (Task 6) -------------------------------------------
ESCALATION_FRAUD_WEIGHT = 0.5
ESCALATION_RECENCY_WEIGHT = 0.5
MAX_AGE_DAYS = 30


def _load_dataset() -> list:
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_escalation_score(flagged_for_fraud_review: bool, days_since_created: int) -> float:
    fraud_component = ESCALATION_FRAUD_WEIGHT * (1 if flagged_for_fraud_review else 0)
    recency_component = ESCALATION_RECENCY_WEIGHT * min(days_since_created / MAX_AGE_DAYS, 1.0)
    return round(fraud_component + recency_component, 4)


def escalation_threshold_from_dataset(records: list = None) -> float:
    records = records or _load_dataset()
    if not records:
        return 0.4  # Safe default if dataset.json has not been generated yet
    ages = sorted(r["days_since_created"] for r in records)
    p80_idx = int(0.8 * (len(ages) - 1))
    p80_age = ages[p80_idx]
    return round(ESCALATION_RECENCY_WEIGHT * (p80_age / MAX_AGE_DAYS), 4)


ESCALATION_THRESHOLD = escalation_threshold_from_dataset()


def rag_core_threshold() -> float:
    from config import CALIBRATED_THRESHOLD
    return CALIBRATED_THRESHOLD


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Answer a loan-policy question using ONLY the retrieved knowledge-base context."""
    fixed_col, sent_col = rag_core.build_indices()
    result = rag_core.grounded_generation(query, sent_col, threshold=rag_core_threshold())
    return json.dumps(result)


@tool("check_loan_application_status")
def check_loan_application_status(record_id: str) -> str:
    """Look up a loan application's status, amount, and escalation score by record_id (e.g. CRED-1001)."""
    records = _load_dataset()
    match = next((r for r in records if r.get("record_id") == record_id.strip().upper()), None)
    if match is None:
        return json.dumps({"error": f"No application found with record_id={record_id}"})

    score = compute_escalation_score(match["flagged_for_fraud_review"], match["days_since_created"])
    return json.dumps({
        "record_id": match["record_id"],
        "status": match["status"],
        "loan_amount_inr": match["loan_amount_inr"],
        "escalation_score": score,
        "escalation_required": score >= ESCALATION_THRESHOLD,
    })


# Strict backward compatibility alias so legacy imports never fail
loan_status_lookup = check_loan_application_status