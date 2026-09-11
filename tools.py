# tools.py

import json
import os

try:
    from crewai.tools import tool
except ImportError:
    from langchain_core.tools import tool

import rag_core

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")

ESCALATION_FRAUD_WEIGHT = 0.7
ESCALATION_RECENCY_WEIGHT = 0.3
MAX_AGE_DAYS = 30


def _load_dataset() -> list:
    if not os.path.exists(DATASET_PATH):
        return []
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def compute_escalation_score(flagged_for_fraud_review: bool, days_since_created: int) -> float:
    fraud_component = ESCALATION_FRAUD_WEIGHT * (1 if flagged_for_fraud_review else 0)
    recency_component = ESCALATION_RECENCY_WEIGHT * min(days_since_created / MAX_AGE_DAYS, 1.0)
    return round(fraud_component + recency_component, 4)


def escalation_threshold_from_dataset(records: list = None) -> float:
    records = records or _load_dataset()
    if not records:
        return 0.65

    flagged_ages = sorted(r["days_since_created"] for r in records if r["flagged_for_fraud_review"])
    if not flagged_ages:
        return 0.65

    p80_idx = int(0.8 * (len(flagged_ages) - 1))
    p80_age = flagged_ages[p80_idx]

    return round(
        ESCALATION_FRAUD_WEIGHT * 1 + ESCALATION_RECENCY_WEIGHT * (p80_age / MAX_AGE_DAYS),
        4,
    )


ESCALATION_THRESHOLD = escalation_threshold_from_dataset()


def rag_core_threshold() -> float:
    from config import CALIBRATED_THRESHOLD
    return CALIBRATED_THRESHOLD


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Answer a loan-policy question using ONLY the retrieved knowledge-base context."""
    fixed_col, sent_col = rag_core.build_indices()  # cheap: memoize in rag_core
    result = rag_core.grounded_generation(query, sent_col, threshold=rag_core_threshold())
    return json.dumps(result)


@tool("check_loan_application_status")
def check_loan_application_status(record_id: str) -> str:
    """Look up a loan application's status, amount, and escalation score by record_id (e.g. CRED-1001)."""
    records = _load_dataset()
    if not records:
        return json.dumps({"error": "Dataset not found. Please run python dataset.py first."})

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


loan_status_lookup = check_loan_application_status