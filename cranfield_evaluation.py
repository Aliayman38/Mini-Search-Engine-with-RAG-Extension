"""
Phase 6: Evaluation
  - MAP  (Mean Average Precision) — binary relevance
  - NDCG (Normalized Discounted Cumulative Gain) — graded relevance

Relevance conversion (per project spec):
  Binary  : scores 1,2,3,4 → relevant=1 | score -1 or missing → 0
  Graded  : 1→4, 2→3, 3→2, 4→1, -1/missing→0
"""

import math
from cranfield_parser import parse_documents, parse_queries, parse_qrels
from cranfield_preprocessor import TextPreprocessor
from cranfield_tfidf import build_inverted_index, build_tfidf_index
from cranfield_retrieval import retrieve_all


# ─────────────────────────────────────────────
# Relevance Converters
# ─────────────────────────────────────────────
def binary_relevance(score: int) -> int:
    """Treat 1,2,3,4 as relevant (1); -1 or missing as not relevant (0)."""
    return 1 if score in (1, 2, 3, 4) else 0


def graded_relevance(score: int) -> int:
    """
    Convert Cranfield relevance score to gain value.
    Cranfield scale is reversed: 1=best, 4=worst.
    Project mapping: 1→4, 2→3, 3→2, 4→1, -1/missing→0
    """
    mapping = {1: 4, 2: 3, 3: 2, 4: 1, -1: 0}
    return mapping.get(score, 0)


# ─────────────────────────────────────────────
# Average Precision (for one query)
# ─────────────────────────────────────────────
def average_precision(ranked_docs: list, relevant_set: dict, k: int = None) -> float:
    """
    Compute Average Precision for a single query.

    AP = (1 / R) * sum_{k: doc_k is relevant} Precision@k

    Args:
        ranked_docs  : [(doc_id, score), ...] in ranked order
        relevant_set : {doc_id -> relevance_score} from qrels
        k            : cutoff (None = use all ranked docs)

    Returns:
        float AP score
    """
    # Binary relevant set
    binary_rel = {did: binary_relevance(s) for did, s in relevant_set.items()}
    R = sum(binary_rel.values())   # total number of relevant docs

    if R == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    docs_to_check = ranked_docs[:k] if k else ranked_docs

    for rank, (doc_id, _) in enumerate(docs_to_check, start=1):
        if binary_rel.get(doc_id, 0) == 1:
            hits += 1
            precision_sum += hits / rank   # Precision@rank

    return precision_sum / R


# ─────────────────────────────────────────────
# MAP (Mean Average Precision)
# ─────────────────────────────────────────────
def mean_average_precision(all_results: dict, qrels: dict, k: int = None) -> tuple:
    """
    Compute MAP across all queries.

    Returns:
        (MAP score, {query_id -> AP score})
    """
    ap_scores = {}
    for qid, ranked_docs in all_results.items():
        relevant_set = qrels.get(qid, {})
        ap_scores[qid] = average_precision(ranked_docs, relevant_set, k)

    map_score = sum(ap_scores.values()) / len(ap_scores) if ap_scores else 0.0
    return map_score, ap_scores


# ─────────────────────────────────────────────
# DCG / NDCG (for one query)
# ─────────────────────────────────────────────
def dcg(ranked_docs: list, relevant_set: dict, k: int) -> float:
    """
    Discounted Cumulative Gain at cutoff k.

    DCG@k = sum_{i=1}^{k}  gain_i / log2(i + 1)

    Uses graded relevance: 1→4, 2→3, 3→2, 4→1, missing→0
    """
    score = 0.0
    for rank, (doc_id, _) in enumerate(ranked_docs[:k], start=1):
        raw_score = relevant_set.get(doc_id, 0)
        gain = graded_relevance(raw_score)
        score += gain / math.log2(rank + 1)
    return score


def ideal_dcg(relevant_set: dict, k: int) -> float:
    """
    Ideal DCG: sort all relevant docs by their graded gain, descending.
    """
    gains = sorted(
        [graded_relevance(s) for s in relevant_set.values()],
        reverse=True
    )
    score = 0.0
    for rank, gain in enumerate(gains[:k], start=1):
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_docs: list, relevant_set: dict, k: int) -> float:
    """
    NDCG@k = DCG@k / IDCG@k
    """
    idcg = ideal_dcg(relevant_set, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked_docs, relevant_set, k) / idcg


# ─────────────────────────────────────────────
# Mean NDCG across all queries
# ─────────────────────────────────────────────
def mean_ndcg(all_results: dict, qrels: dict, k: int) -> tuple:
    """
    Compute mean NDCG@k across all queries.

    Returns:
        (mean NDCG score, {query_id -> NDCG score})
    """
    ndcg_scores = {}
    for qid, ranked_docs in all_results.items():
        relevant_set = qrels.get(qid, {})
        ndcg_scores[qid] = ndcg_at_k(ranked_docs, relevant_set, k)

    mean = sum(ndcg_scores.values()) / len(ndcg_scores) if ndcg_scores else 0.0
    return mean, ndcg_scores


# ─────────────────────────────────────────────
# Precision@k and Recall@k helpers
# ─────────────────────────────────────────────
def precision_at_k(ranked_docs: list, relevant_set: dict, k: int) -> float:
    binary_rel = {did: binary_relevance(s) for did, s in relevant_set.items()}
    hits = sum(1 for doc_id, _ in ranked_docs[:k] if binary_rel.get(doc_id, 0) == 1)
    return hits / k if k > 0 else 0.0


def recall_at_k(ranked_docs: list, relevant_set: dict, k: int) -> float:
    binary_rel = {did: binary_relevance(s) for did, s in relevant_set.items()}
    R = sum(binary_rel.values())
    if R == 0:
        return 0.0
    hits = sum(1 for doc_id, _ in ranked_docs[:k] if binary_rel.get(doc_id, 0) == 1)
    return hits / R


# ─────────────────────────────────────────────
# Full Evaluation Report
# ─────────────────────────────────────────────
def evaluate(all_results: dict, qrels: dict, k: int = 10):
    """
    Run full evaluation and print a detailed report.
    """
    map_score,  ap_scores   = mean_average_precision(all_results, qrels)
    ndcg_score, ndcg_scores = mean_ndcg(all_results, qrels, k)

    # P@k and R@k (averaged over all queries)
    pk_vals = []
    rk_vals = []
    for qid, ranked_docs in all_results.items():
        rel = qrels.get(qid, {})
        pk_vals.append(precision_at_k(ranked_docs, rel, k))
        rk_vals.append(recall_at_k(ranked_docs, rel, k))

    mean_pk = sum(pk_vals) / len(pk_vals) if pk_vals else 0.0
    mean_rk = sum(rk_vals) / len(rk_vals) if rk_vals else 0.0

    print("=" * 58)
    print("          PHASE 6: EVALUATION REPORT")
    print("=" * 58)
    print(f"\n  📐 Cutoff k = {k}")
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  MAP  (Mean Avg Precision)  = {map_score:.4f}    │")
    print(f"  │  NDCG@{k:<2d}                  = {ndcg_score:.4f}    │")
    print(f"  │  Precision@{k:<2d}              = {mean_pk:.4f}    │")
    print(f"  │  Recall@{k:<2d}                = {mean_rk:.4f}    │")
    print(f"  └─────────────────────────────────────────┘")

    # Top 5 best queries by AP
    top5_ap = sorted(ap_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  🏆 Top-5 Queries by AP Score:")
    for qid, ap in top5_ap:
        print(f"    Query {qid:3d}  AP={ap:.4f}  NDCG={ndcg_scores[qid]:.4f}")

    # Bottom 5 worst queries
    bot5_ap = sorted(ap_scores.items(), key=lambda x: x[1])[:5]
    print(f"\n  📉 Bottom-5 Queries by AP Score:")
    for qid, ap in bot5_ap:
        print(f"    Query {qid:3d}  AP={ap:.4f}  NDCG={ndcg_scores[qid]:.4f}")

    # Evaluation at different k values
    print(f"\n  📊 NDCG at different cutoffs:")
    for cutoff in [5, 10, 20, 50]:
        m, _ = mean_ndcg(all_results, qrels, cutoff)
        print(f"    NDCG@{cutoff:<3d} = {m:.4f}")

    print(f"\n✅ Phase 6 Complete — Evaluation done!")
    print("=" * 58)

    return map_score, ndcg_score, ap_scores, ndcg_scores


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    DOCS_PATH    = '/mnt/user-data/uploads/data/cran.all.1400'
    QUERIES_PATH = '/mnt/user-data/uploads/data/cran.qry'
    QRELS_PATH   = '/mnt/user-data/uploads/cran_data/cranqrel'

    K = 10

    print("Loading and preprocessing...")
    documents = parse_documents(DOCS_PATH)
    queries   = parse_queries(QUERIES_PATH)
    qrels     = parse_qrels(QRELS_PATH)

    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    print("Building index...")
    inverted_index = build_inverted_index(proc_docs)
    tfidf_matrix, idf_scores, _ = build_tfidf_index(inverted_index, proc_docs)

    print(f"Retrieving top-{K} docs for all 225 queries...")
    all_results = retrieve_all(proc_queries, tfidf_matrix, idf_scores, inverted_index, k=K)

    evaluate(all_results, qrels, k=K)
