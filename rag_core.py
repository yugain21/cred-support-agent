# rag_core.py
"""
Tasks 3-5: dual chunking strategies, dual ChromaDB collections,
empirically-calibrated retrieval threshold, and grounded generation.

IMPORTANT FIX vs the original repo: chroma_client.get_or_create_collection()
defaults to L2 (squared Euclidean) distance, not cosine distance. The
original code converted that L2 distance with `1 - d/2` as if it were a
cosine distance, which silently produced meaningless "similarity" numbers
and made threshold calibration impossible to reproduce (that's why the
old repo's own transcripts contradicted its own README). We fix this by
explicitly requesting cosine space per collection.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from knowledge_base import DOCUMENTS

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

COSINE_SPACE = {"hnsw:space": "cosine"}


def build_indices():
    fixed_col = chroma_client.get_or_create_collection("fixed_chunks", metadata=COSINE_SPACE)
    sent_col = chroma_client.get_or_create_collection("sentence_chunks", metadata=COSINE_SPACE)

    # 1. Fixed-size chunking (120 chars, 30 char overlap -> step of 90)
    fixed_chunks, fixed_ids, fixed_parents = [], [], []
    for doc_id, text in DOCUMENTS.items():
        step = 90
        for i in range(0, len(text), step):
            chunk = text[i:i + 120]
            if not chunk.strip():
                continue
            fixed_chunks.append(chunk)
            fixed_ids.append(f"fix_{doc_id}_{i}")
            fixed_parents.append(doc_id)

    fixed_col.upsert(
        ids=fixed_ids,
        documents=fixed_chunks,
        embeddings=embedder.encode(fixed_chunks).tolist(),
        metadatas=[{"parent_doc": p} for p in fixed_parents],
    )

    # 2. Sentence-based chunking
    sent_chunks, sent_ids, sent_parents = [], [], []
    for doc_id, text in DOCUMENTS.items():
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        for idx, sentence in enumerate(sentences):
            sent_chunks.append(sentence)
            sent_ids.append(f"sent_{doc_id}_{idx}")
            sent_parents.append(doc_id)

    sent_col.upsert(
        ids=sent_ids,
        documents=sent_chunks,
        embeddings=embedder.encode(sent_chunks).tolist(),
        metadatas=[{"parent_doc": p} for p in sent_parents],
    )
    return fixed_col, sent_col


def _query_with_similarities(collection, query: str, n_results: int = 3):
    """Returns a list of {doc, parent, similarity} for EACH retrieved
    chunk individually (not just the top-1), so callers can filter each
    chunk by its own score instead of gating the whole batch on the best
    match alone."""
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results)
    if not results["documents"][0]:
        return []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    # With hnsw:space="cosine", Chroma's returned "distance" IS cosine
    # distance (1 - cosine_similarity), so similarity = 1 - distance.
    return [
        {"doc": d, "parent": m["parent_doc"], "similarity": 1.0 - dist}
        for d, m, dist in zip(docs, metas, dists)
    ]


def _similarity(collection, query: str, n_results: int = 3):
    """Returns (top_similarity, retrieved_docs, retrieved_metadatas) -
    kept for the single-best-match use cases (calibration, the fallback
    gate in grounded_generation)."""
    rows = _query_with_similarities(collection, query, n_results=n_results)
    if not rows:
        return 0.0, [], []
    top = rows[0]
    return top["similarity"], [r["doc"] for r in rows], [{"parent_doc": r["parent"]} for r in rows]


def calibrate_threshold(collection, in_scope_queries: list, out_scope_queries: list):
    """
    Task 4: empirically measure top-1 similarity for real in-scope and
    out-of-scope queries, then pick a threshold between the two observed
    clusters instead of trusting an untested preset.
    """
    in_scores = [_similarity(collection, q)[0] for q in in_scope_queries]
    out_scores = [_similarity(collection, q)[0] for q in out_scope_queries]

    lowest_in = min(in_scores) if in_scores else 0.0
    highest_out = max(out_scores) if out_scores else 0.0

    if lowest_in <= highest_out:
        # Clusters overlap for this embedding/chunking combo - fall back to
        # the midpoint and flag it, rather than silently picking a value
        # that doesn't actually separate the two groups.
        threshold = (lowest_in + highest_out) / 2
        separated = False
    else:
        threshold = (lowest_in + highest_out) / 2
        separated = True

    return {
        "in_scope_scores": list(zip(in_scope_queries, in_scores)),
        "out_scope_scores": list(zip(out_scope_queries, out_scores)),
        "threshold": threshold,
        "clusters_separated": separated,
    }


def grounded_generation(query: str, collection, threshold: float, n_results: int = 2):
    """
    Task 4: retrieve top-k chunks and generate an answer using ONLY that
    retrieved context. Under MOCK_LLM there's no real generator, so this
    produces an extractive, synthesized answer built strictly from the
    retrieved chunk text - never inventing facts outside the context -
    and falls back to "I don't know" when similarity is below threshold.
    """
    rows = _query_with_similarities(collection, query, n_results=n_results)
    top_similarity = rows[0]["similarity"] if rows else 0.0

    if not rows or top_similarity < threshold:
        return {
            "answer": "I don't know - this isn't covered in the policy documents I have access to.",
            "grounded": False,
            "similarity": top_similarity,
            "source_docs": [],
        }

    # Only synthesize from chunks that individually clear the threshold -
    # the top match qualifying doesn't mean chunk #2/#3 are relevant too.
    relevant_rows = [r for r in rows if r["similarity"] >= threshold]
    parent_docs = sorted({r["parent"] for r in relevant_rows})
    synthesized = " ".join(r["doc"] for r in relevant_rows)
    answer = f"Based on policy: {synthesized}"

    return {
        "answer": answer,
        "grounded": True,
        "similarity": top_similarity,
        "source_docs": parent_docs,
    }


def document_level_precision_recall(collection, queries: list, relevant_doc_map: dict, threshold: float):
    """
    Task 5: precision/recall at the DOCUMENT level (chunks mapped back to
    parent doc and deduped) for one collection, with per-query arithmetic
    returned so it can be printed.
    """
    output_rows = []
    total_tp = total_fp = total_fn = 0

    for q in queries:
        chunk_rows = _query_with_similarities(collection, q, n_results=3)
        top_similarity = chunk_rows[0]["similarity"] if chunk_rows else 0.0
        # Each chunk is judged on its OWN similarity, not just whether the
        # best match in the batch cleared the bar - this is what actually
        # lets fixed-size vs sentence-based chunking produce different
        # precision numbers instead of both being dragged down equally.
        retrieved_parents = {r["parent"] for r in chunk_rows if r["similarity"] >= threshold}

        relevant_parents = set(relevant_doc_map.get(q, []))

        tp = len(retrieved_parents & relevant_parents)
        fp = len(retrieved_parents - relevant_parents)
        fn = len(relevant_parents - retrieved_parents)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not relevant_parents else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not relevant_parents else 0.0)

        output_rows.append({
            "query": q, "similarity": top_similarity,
            "retrieved": sorted(retrieved_parents), "relevant": sorted(relevant_parents),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return output_rows, overall_precision, overall_recall