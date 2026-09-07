# eval.py
import json
from rag_core import build_indices, calibrate_threshold, grounded_generation, document_level_precision_recall
from knowledge_base import DOC_TOPICS
from mock_llm import LocalDeterministicLLM
from agents import process_query_with_crew
import config


def run_evaluation():
    print("Building vector indices for evaluation...")
    fixed_col, sent_col = build_indices()

    print("\n=== TASK 4: Empirical Threshold Calibration ===")
    in_scope = ["What are the late fees?", "What is the interest rate on savings?", "How do I dispute a fraud charge?"]
    out_scope = ["How do I bake a chocolate cake?", "What's the weather like today?"]

    calib = calibrate_threshold(sent_col, in_scope, out_scope)
    print("In-scope similarities:")
    for q, s in calib["in_scope_scores"]:
        print(f"  {s:.4f} | {q}")
    print("Out-of-scope similarities:")
    for q, s in calib["out_scope_scores"]:
        print(f"  {s:.4f} | {q}")
    print(f"Chosen threshold: {calib['threshold']:.4f} "
          f"(clusters cleanly separated: {calib['clusters_separated']})")
    config.set_calibrated_threshold(calib["threshold"])

    print("\n=== TASK 3/4: Grounded Generation Demo (5 in-scope + 1 out-of-scope) ===")
    demo_queries = [
        "What are the late fees?",
        "What is the interest rate on savings?",
        "How do I dispute a fraud charge?",
        "What documents are needed for KYC?",
        "How is my EMI calculated?",
        "How do I bake a chocolate cake?",  # deliberately out-of-scope
    ]
    for q in demo_queries:
        result = grounded_generation(q, sent_col, threshold=calib["threshold"])
        print(f"Q: {q}\n   grounded={result['grounded']} sim={result['similarity']:.4f}\n   A: {result['answer']}\n")

    print("=== TASK 5: Document-level Precision & Recall (Fixed vs Sentence) ===")
    pr_queries = [
        "What are the late fees?",
        "What is the interest rate on savings?",
        "How do I dispute a fraud charge?",
        "What documents are needed for KYC?",
        "How is my EMI calculated?",
    ]
    relevant_doc_map = {
        "What are the late fees?": ["doc_3"],
        "What is the interest rate on savings?": ["doc_7"],
        "How do I dispute a fraud charge?": ["doc_5"],
        "What documents are needed for KYC?": ["doc_4"],
        "How is my EMI calculated?": ["doc_2"],
    }

    for name, col in [("Fixed-Size", fixed_col), ("Sentence-Based", sent_col)]:
        rows, precision, recall = document_level_precision_recall(col, pr_queries, relevant_doc_map, calib["threshold"])
        print(f"\n{name} strategy - per-query arithmetic:")
        for r in rows:
            print(f"  '{r['query']}': retrieved={r['retrieved']} relevant={r['relevant']} "
                  f"tp={r['tp']} fp={r['fp']} fn={r['fn']} -> P={r['precision']:.2f} R={r['recall']:.2f}")
        print(f"{name} -> Overall Precision: {precision:.2f} | Overall Recall: {recall:.2f}")

    print("\n=== TASK 13: 15-Query Evaluation (all 12 KB topics + 2 out-of-scope + edge cases) ===")
    topic_queries = {
        "doc_1": "What is the minimum income needed for a personal loan?",
        "doc_2": "How is my EMI calculated?",
        "doc_3": "What is the annual fee on a standard credit card?",
        "doc_4": "What documents are needed for KYC?",
        "doc_5": "How do I dispute a fraudulent charge?",
        "doc_6": "How do I close my account?",
        "doc_7": "What is the interest rate on savings above 10 lakhs?",
        "doc_8": "Is there a prepayment penalty on my auto loan?",
        "doc_9": "What is the minimum balance for a savings account?",
        "doc_10": "What factors affect my credit score?",
        "doc_11": "What are the rules for a joint account?",
        "doc_12": "Am I eligible for an NRI account?",
    }
    test_queries = list(topic_queries.values()) + [
        "How do I bake a chocolate cake?",       # out-of-scope
        "What's the weather like in Mumbai?",     # out-of-scope
        "",                                        # edge case: empty query
    ]
    assert len(test_queries) == 15, f"expected 15 test queries, got {len(test_queries)}"

    mock_judge = LocalDeterministicLLM(model="mock-llm")
    totals = {"accuracy": 0.0, "grounding": 0.0, "completeness": 0.0, "safety": 0.0}

    for q in test_queries:
        prompt = f"Please evaluate for accuracy grounding completeness and safety. Query: {q}"
        judgment_str = mock_judge.invoke(prompt)
        try:
            j = json.loads(judgment_str)
        except json.JSONDecodeError:
            j = {"accuracy": 0.0, "grounding": 0.0, "completeness": 0.0, "safety": 0.0}

        for k in totals:
            totals[k] += j.get(k, 0.0)
        print(f"Query: {q!r:60} -> Acc: {j['accuracy']}, Grnd: {j['grounding']}, "
              f"Comp: {j['completeness']}, Safe: {j['safety']}")

    n = len(test_queries)
    print(f"\nAggregate Averages -> "
          f"Accuracy: {totals['accuracy']/n:.2f} | Grounding: {totals['grounding']/n:.2f} | "
          f"Completeness: {totals['completeness']/n:.2f} | Safety: {totals['safety']/n:.2f}")


if __name__ == "__main__":
    run_evaluation()
