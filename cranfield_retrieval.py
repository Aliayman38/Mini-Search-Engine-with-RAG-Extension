"""
Phase 5: Query Processing & Cosine Similarity Ranking

Steps:
  1. Preprocess query using the same pipeline as documents
  2. Build query TF-IDF vector
  3. Compute cosine similarity between query vector and every document vector
  4. Return top-k ranked documents
"""

import math
from collections import defaultdict
from cranfield_parser import parse_documents, parse_queries, parse_qrels
from cranfield_preprocessor import TextPreprocessor
from cranfield_tfidf import (
    build_inverted_index, build_tfidf_index,
    compute_tf, compute_idf
)


# ─────────────────────────────────────────────
# Query Vector Builder
# ─────────────────────────────────────────────
def build_query_vector(tokens: list, idf_scores: dict) -> dict:
    """
    Build a TF-IDF vector for a query.
    Uses the same TF formula as documents, with the corpus IDF scores.

    Args:
        tokens     : preprocessed query tokens
        idf_scores : {term -> idf} from the document corpus

    Returns:
        {term -> tfidf_score}  (only terms that exist in the corpus)
    """
    if not tokens:
        return {}

    # Count raw term frequencies in the query
    raw_counts = defaultdict(int)
    for token in tokens:
        raw_counts[token] += 1

    query_len = len(tokens)
    vector = {}
    for term, count in raw_counts.items():
        if term in idf_scores:          # ignore OOV terms
            tf = compute_tf(count, query_len)
            vector[term] = tf * idf_scores[term]

    return vector


# ─────────────────────────────────────────────
# Cosine Similarity
# ─────────────────────────────────────────────
def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Compute cosine similarity between two sparse TF-IDF vectors.

    cos(q, d) = (q · d) / (|q| * |d|)

    Args:
        vec_a, vec_b : {term -> tfidf_score}

    Returns:
        float in [0, 1]
    """
    # Dot product — only over shared terms
    shared_terms = set(vec_a.keys()) & set(vec_b.keys())
    if not shared_terms:
        return 0.0

    dot_product = sum(vec_a[t] * vec_b[t] for t in shared_terms)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ─────────────────────────────────────────────
# Retrieval Engine
# ─────────────────────────────────────────────
def retrieve(query_tokens: list,
             tfidf_matrix: dict,
             idf_scores: dict,
             inverted_index: dict,
             k: int = 10) -> list:
    """
    Retrieve top-k documents for a query using cosine similarity.

    Strategy:
      - Build query vector
      - Only score documents that share at least one term with the query
        (using the inverted index for efficiency)
      - Rank by cosine similarity, return top-k

    Returns:
        list of (doc_id, score) sorted descending by score
    """
    query_vector = build_query_vector(query_tokens, idf_scores)

    if not query_vector:
        return []

    # Candidate documents: union of posting lists for query terms
    candidate_docs = set()
    for term in query_vector:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term].keys())

    # Score each candidate
    scores = {}
    for doc_id in candidate_docs:
        doc_vector = tfidf_matrix.get(doc_id, {})
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)

    # Sort descending and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ─────────────────────────────────────────────
# Batch Retrieval (all queries)
# ─────────────────────────────────────────────
def retrieve_all(proc_queries: dict,
                 tfidf_matrix: dict,
                 idf_scores: dict,
                 inverted_index: dict,
                 k: int = 10) -> dict:
    """
    Run retrieval for all queries.

    Returns:
        {query_id -> [(doc_id, score), ...]}
    """
    results = {}
    for qid, tokens in proc_queries.items():
        results[qid] = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k)
    return results


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────
def verify(queries, proc_queries, documents, tfidf_matrix, idf_scores,
           inverted_index, qrels, all_results):

    print("=" * 62)
    print("     PHASE 5: RETRIEVAL & COSINE SIMILARITY VERIFICATION")
    print("=" * 62)

    # Show results for 3 sample queries
    sample_qids = [1, 2, 3]
    for qid in sample_qids:
        if qid not in queries:
            continue
        raw_query = queries[qid]
        tokens    = proc_queries[qid]
        results   = all_results[qid]
        rel_docs  = qrels.get(qid, {})

        print(f"\n{'─'*60}")
        print(f"Query {qid}: {raw_query}")
        print(f"Tokens   : {tokens}")
        print(f"\n  Top-10 Retrieved Documents:")
        print(f"  {'Rank':>4}  {'DocID':>6}  {'Score':>8}  {'Rel?':>5}  Title")
        print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*35}")
        for rank, (doc_id, score) in enumerate(results, start=1):
            rel = rel_docs.get(doc_id, -1)
            rel_label = f"✅ {rel}" if rel in (1, 2, 3, 4) else "  ✗"
            title = documents[doc_id]['title'].replace('\n', ' ')[:40]
            print(f"  {rank:>4}  {doc_id:>6}  {score:>8.5f}  {rel_label:>5}  {title}")

    # Coverage stats
    print(f"\n{'─'*60}")
    print(f"📊 Retrieval Coverage")
    total_queries = len(all_results)
    queries_with_hits = sum(1 for r in all_results.values() if len(r) > 0)
    print(f"  Queries processed     : {total_queries}")
    print(f"  Queries with results  : {queries_with_hits}")

    # Check how many relevant docs appear in top-10
    hits = 0
    total_rel = 0
    for qid, results in all_results.items():
        rel_docs = qrels.get(qid, {})
        retrieved_ids = {doc_id for doc_id, _ in results}
        relevant_ids  = {did for did, s in rel_docs.items() if s != -1}
        hits      += len(retrieved_ids & relevant_ids)
        total_rel += len(relevant_ids)

    print(f"  Relevant docs in corpus  : {total_rel}")
    print(f"  Relevant docs in top-10  : {hits}")
    print(f"  Hit rate                 : {hits/total_rel*100:.1f}% of relevant docs appear in top-10")

    print(f"\n✅ Phase 5 Complete — Retrieval engine working!")
    print("=" * 62)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    DOCS_PATH    = '/mnt/user-data/uploads/data/cran.all.1400'
    QUERIES_PATH = '/mnt/user-data/uploads/data/cran.qry'
    QRELS_PATH   = '/mnt/user-data/uploads/cran_data/cranqrel'

    print("Loading data...")
    documents = parse_documents(DOCS_PATH)
    queries   = parse_queries(QUERIES_PATH)
    qrels     = parse_qrels(QRELS_PATH)

    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    print("Building index...")
    inverted_index = build_inverted_index(proc_docs)
    tfidf_matrix, idf_scores, doc_lengths = build_tfidf_index(inverted_index, proc_docs)

    print("Running retrieval for all 225 queries (top-10)...")
    all_results = retrieve_all(proc_queries, tfidf_matrix, idf_scores, inverted_index, k=10)

    verify(queries, proc_queries, documents, tfidf_matrix, idf_scores,
           inverted_index, qrels, all_results)
