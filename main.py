"""
main.py — Cranfield Mini Search Engine with RAG Extension
AI356 Information Retrieval · Spring 2025-2026

Runs the complete pipeline:
  Phase 1: Parse dataset
  Phase 2: Preprocess
  Phase 3: Build inverted index
  Phase 4: TF-IDF weighting
  Phase 5: Cosine similarity retrieval
  Phase 6: MAP + NDCG evaluation
  Phase 7: RAG answer generation

Usage:
  python main.py                        # interactive query mode
  python main.py --eval                 # full evaluation report only
  python main.py --query "your query"   # single query mode
"""

import os
import sys
import argparse

from cranfield_parser      import parse_documents, parse_queries, parse_qrels
from cranfield_preprocessor import TextPreprocessor
from cranfield_tfidf        import build_inverted_index, build_tfidf_index
from cranfield_retrieval    import retrieve, retrieve_all
from cranfield_evaluation   import evaluate, mean_average_precision, mean_ndcg
from cranfield_rag          import generate_rag_answer

# ── File paths ──────────────────────────────────────────────────
DOCS_PATH    = 'data/cran.all.1400'
QUERIES_PATH = 'data/cran.qry'
QRELS_PATH   = 'data/cranqrel'

K_RETRIEVE   = 10    # top-k documents for display
K_RAG        = 5     # top-k documents for RAG context
K_EVAL_FULL  = 1400  # retrieve all docs for accurate MAP


def load_and_build():
    """Load dataset, preprocess, and build index. Returns all needed objects."""
    print("[ 1/4 ] Parsing dataset files...")
    documents = parse_documents(DOCS_PATH)
    queries   = parse_queries(QUERIES_PATH)
    qrels     = parse_qrels(QRELS_PATH)
    print(f"        {len(documents)} docs · {len(queries)} queries · {len(qrels)} qrel entries")

    print("[ 2/4 ] Preprocessing (tokenize → stopwords → stem)...")
    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    print("[ 3/4 ] Building inverted index...")
    inverted_index = build_inverted_index(proc_docs)

    print("[ 4/4 ] Computing TF-IDF matrix...")
    tfidf_matrix, idf_scores, _ = build_tfidf_index(inverted_index, proc_docs)

    print(f"        Vocabulary: {len(idf_scores):,} terms · Matrix: {len(tfidf_matrix):,} docs\n")
    return documents, queries, qrels, preprocessor, proc_docs, proc_queries, inverted_index, tfidf_matrix, idf_scores


def run_query(raw_query, preprocessor, documents, queries, qrels,
              inverted_index, tfidf_matrix, idf_scores, api_key=None):
    """Run retrieval + evaluation + RAG for a single query string."""

    tokens    = preprocessor.preprocess(raw_query)
    top_docs  = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k=K_RETRIEVE)
    top_full  = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k=K_EVAL_FULL)

    # ── Display retrieved docs ───────────────────────────────────
    print("=" * 65)
    print(f"Query: {raw_query}")
    print(f"Tokens: {tokens}")
    print("=" * 65)
    print(f"\n{'Rank':>4}  {'DocID':>6}  {'Score':>8}  {'Rel':>4}  Title")
    print(f"{'─'*4}  {'─'*6}  {'─'*8}  {'─'*4}  {'─'*38}")

    # Find the query ID for relevance lookup
    qid = None
    for qid_cand, q_text in queries.items():
        if q_text.strip().lower() == raw_query.strip().lower():
            qid = qid_cand
            break

    rel_map  = qrels.get(qid, {}) if qid else {}
    rel_conv = {1:'[1]best', 2:'[2]high', 3:'[3]usef', 4:'[4]min ', -1:'[✗]  '}

    for rank, (doc_id, score) in enumerate(top_docs, 1):
        rel   = rel_map.get(doc_id, None)
        rl    = rel_conv.get(rel, '     ') if rel else '     '
        title = documents[doc_id]['title'].replace('\n', ' ')[:40]
        print(f"{rank:>4}  {doc_id:>6}  {score:>8.5f}  {rl}  {title}")

    # ── Per-query evaluation (if qrel exists) ───────────────────
    if rel_map:
        from cranfield_evaluation import (
            average_precision, ndcg_at_k, precision_at_k, recall_at_k
        )
        ap   = average_precision(top_full, rel_map)
        ndcg = ndcg_at_k(top_docs, rel_map, K_RETRIEVE)
        pk   = precision_at_k(top_docs, rel_map, K_RETRIEVE)
        rk   = recall_at_k(top_docs, rel_map, K_RETRIEVE)
        print(f"\n  MAP (this query):  AP={ap:.4f}")
        print(f"  NDCG@{K_RETRIEVE}:         {ndcg:.4f}")
        print(f"  Precision@{K_RETRIEVE}:    {pk:.4f}")
        print(f"  Recall@{K_RETRIEVE}:       {rk:.4f}")

    # ── RAG ─────────────────────────────────────────────────────
    if api_key:
        print(f"\n🤖 Generating RAG answer (top-{K_RAG} docs → Claude API)...\n")
        result = generate_rag_answer(raw_query, top_docs[:K_RAG], documents, api_key)
        print("─" * 65)
        print("RAG Answer:")
        print("─" * 65)
        print(result["rag_answer"])
        print("─" * 65)
    else:
        print("  (RAG answer generated locally — no API key required)")


def interactive_mode(preprocessor, documents, queries, qrels,
                     inverted_index, tfidf_matrix, idf_scores, api_key):
    """REPL: keep accepting queries until user types 'quit'."""
    print("\n" + "="*65)
    print("  CRANFIELD SEARCH ENGINE — Interactive Mode")
    print("  Type a query and press Enter. Type 'quit' to exit.")
    print("="*65 + "\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ('quit', 'exit', 'q'):
            break
        run_query(raw, preprocessor, documents, queries, qrels,
                  inverted_index, tfidf_matrix, idf_scores, api_key)
        print()


# ── Entry point ─────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cranfield Mini Search Engine with RAG')
    parser.add_argument('--eval',  action='store_true', help='Run full evaluation report')
    parser.add_argument('--query', type=str, default=None, help='Single query to run')
    args = parser.parse_args()


    api_key = None  # RAG is fully local — no API key needed

    (documents, queries, qrels, preprocessor, proc_docs, proc_queries,
     inverted_index, tfidf_matrix, idf_scores) = load_and_build()

    if args.eval:
        print("Running full evaluation over all 225 queries...\n")
        all_results = retrieve_all(proc_queries, tfidf_matrix, idf_scores, inverted_index, k=K_EVAL_FULL)
        evaluate(all_results, qrels, k=10)

    elif args.query:
        run_query(args.query, preprocessor, documents, queries, qrels,
                  inverted_index, tfidf_matrix, idf_scores, api_key)

    else:
        interactive_mode(preprocessor, documents, queries, qrels,
                         inverted_index, tfidf_matrix, idf_scores, api_key)
