# dataset.py
"""
Deterministic, seeded loan-application dataset generator.

Design choices (also stated in README.md):
- Seed: 42
- Category weights: uniform, with a guaranteed floor of 3 records/category
- Status weights: uniform, with a guaranteed floor of 1 record/status
- Amount range: 50,000 - 5,000,000 INR (covers small personal loans up to
  large home loans, per Cred's stated loan categories)
- Fraud rate: drawn from Bernoulli(p=0.22) per record. 0.22 sits in the
  middle of the required 10-30% band; because it's a random draw the
  realized percentage can drift slightly per seed, so we assert the band
  post-generation rather than trusting the parameter alone.
"""
import random
import json
from collections import Counter

CATEGORIES = ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"]
STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

MIN_PER_CATEGORY = 3
FRAUD_RATE = 0.22
FRAUD_BAND = (0.10, 0.30)


def _make_record(category: str, status: str) -> dict:
    return {
        "category": category,
        "status": status,
        # Amount range reasoning: 50k INR covers small personal loans;
        # 5M INR covers large home/business loans. Single realistic range
        # for all categories keeps the generator simple and reproducible.
        "loan_amount_inr": random.randint(50, 5000) * 1000,
        "days_since_created": random.randint(0, 30),
        "flagged_for_fraud_review": random.random() < FRAUD_RATE,
    }


def generate_dataset(seed: int = 42, num_records: int = 45):
    random.seed(seed)
    records = []

    # Guarantee every category appears at least MIN_PER_CATEGORY times
    for cat in CATEGORIES:
        for _ in range(MIN_PER_CATEGORY):
            records.append(_make_record(cat, random.choice(STATUSES)))

    # Guarantee every status appears at least once (on top of whatever
    # the category loop already produced)
    for stat in STATUSES:
        records.append(_make_record(random.choice(CATEGORIES), stat))

    # Fill remaining records to hit target length
    while len(records) < num_records:
        records.append(_make_record(random.choice(CATEGORIES), random.choice(STATUSES)))

    random.shuffle(records)
    for idx, r in enumerate(records):
        r["record_id"] = f"CRED-{1000 + idx}"

    fraud_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    fraud_pct = (fraud_count / len(records)) * 100

    if not (FRAUD_BAND[0] * 100 <= fraud_pct <= FRAUD_BAND[1] * 100):
        raise SystemExit(
            f"Fraud rate {fraud_pct:.1f}% fell outside the required "
            f"{FRAUD_BAND[0]*100:.0f}-{FRAUD_BAND[1]*100:.0f}% band for seed={seed}. "
            f"Per the brief: change the seed or weights and regenerate — "
            f"never hand-edit individual records."
        )

    return records, fraud_pct


def report(records: list, fraud_pct: float) -> None:
    by_category = Counter(r["category"] for r in records)
    by_status = Counter(r["status"] for r in records)

    print(f"Generated {len(records)} records (seed=42).\n")

    print("Count per category:")
    for cat in CATEGORIES:
        print(f"  {cat:<16} {by_category.get(cat, 0)}")

    print("\nCount per status:")
    for stat in STATUSES:
        print(f"  {stat:<14} {by_status.get(stat, 0)}")

    print(f"\nFlagged-for-fraud-review rate: {fraud_pct:.1f}% "
          f"(required band: 10-30%)")


if __name__ == "__main__":
    data, pct = generate_dataset()
    report(data, pct)
    with open("dataset.json", "w") as f:
        json.dump(data, f, indent=2)
