"""
Phase 3: Inverted Index Construction
Phase 4: TF-IDF Computation

Inverted Index structure:
    { term -> { doc_id -> term_frequency } }

TF-IDF:
    TF(t, d)  = count(t in d) / total_tokens(d)   [normalized TF]
    IDF(t)    = log( N / df(t) ) + 1              [smoothed IDF]
    TF-IDF(t,d) = TF(t,d) * IDF(t)
"""

import math
from collections import defaultdict
from cranfield_parser import parse_documents, parse_queries, parse_qrels
from cranfield_preprocessor import TextPreprocessor


# ─────────────────────────────────────────────
# Phase 3: Inverted Index
# ─────────────────────────────────────────────
def build_inverted_index(proc_docs: dict) -> dict:
    """
    Build inverted index from preprocessed documents.

    Args:
        proc_docs: {doc_id -> [tokens]}

    Returns:
        inverted_index: {term -> {doc_id -> raw_term_count}}
    """
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, tokens in proc_docs.items():
        for token in tokens:
            inverted_index[token][doc_id] += 1

    # Convert defaultdicts to regular dicts for clean access
    return {term: dict(postings) for term, postings in inverted_index.items()}


# ─────────────────────────────────────────────
# Phase 4: TF-IDF
# ─────────────────────────────────────────────
def compute_tf(term_count: int, doc_length: int) -> float:
    """
    Normalized Term Frequency.
    TF(t, d) = count(t in d) / total_tokens(d)
    """
    if doc_length == 0:
        return 0.0
    return term_count / doc_length


def compute_idf(N: int, df: int) -> float:
    """
    Smoothed Inverse Document Frequency.
    IDF(t) = log(N / df(t)) + 1
    Prevents division by zero; +1 ensures non-zero for very common terms.
    """
    if df == 0:
        return 0.0
    return math.log(N / df) + 1


def build_tfidf_index(inverted_index: dict, proc_docs: dict) -> tuple:
    """
    Build complete TF-IDF index.

    Returns:
        tfidf_matrix : {doc_id -> {term -> tfidf_score}}
        idf_scores   : {term -> idf_value}
        doc_lengths  : {doc_id -> number_of_tokens}
    """
    N = len(proc_docs)                          # total number of documents

    # Document lengths (total tokens per doc)
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in proc_docs.items()}

    # IDF for every term in the vocabulary
    idf_scores = {}
    for term, postings in inverted_index.items():
        df = len(postings)                      # number of docs containing term
        idf_scores[term] = compute_idf(N, df)

    # TF-IDF matrix
    tfidf_matrix = defaultdict(dict)
    for term, postings in inverted_index.items():
        idf = idf_scores[term]
        for doc_id, raw_count in postings.items():
            tf  = compute_tf(raw_count, doc_lengths[doc_id])
            tfidf_matrix[doc_id][term] = tf * idf

    return dict(tfidf_matrix), idf_scores, doc_lengths


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────
def verify(inverted_index, tfidf_matrix, idf_scores, doc_lengths):
    print("=" * 58)
    print("   PHASE 3 & 4: INVERTED INDEX + TF-IDF VERIFICATION")
    print("=" * 58)

    vocab_size = len(inverted_index)
    total_postings = sum(len(p) for p in inverted_index.values())
    avg_postings = total_postings / vocab_size if vocab_size else 0

    print(f"\n📚 INVERTED INDEX")
    print(f"  Vocabulary size       : {vocab_size:,} terms")
    print(f"  Total postings        : {total_postings:,}")
    print(f"  Avg docs per term     : {avg_postings:.2f}")

    # Top-10 most common terms (highest df)
    top_terms = sorted(inverted_index.items(),
                       key=lambda x: len(x[1]), reverse=True)[:10]
    print(f"\n  Top-10 most frequent terms (by doc frequency):")
    for term, postings in top_terms:
        print(f"    '{term:15s}'  df={len(postings):4d}  "
              f"idf={idf_scores[term]:.4f}")

    # Rare terms sample
    rare_terms = sorted(inverted_index.items(),
                        key=lambda x: len(x[1]))[:5]
    print(f"\n  5 rarest terms (df=1):")
    for term, postings in rare_terms:
        print(f"    '{term:15s}'  df={len(postings):4d}  "
              f"idf={idf_scores[term]:.4f}")

    print(f"\n📊 TF-IDF MATRIX")
    print(f"  Documents with TF-IDF vectors: {len(tfidf_matrix):,}")

    # Sample doc TF-IDF top terms
    sample_id = 1
    doc_vec = tfidf_matrix[sample_id]
    top_doc_terms = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"\n  Top TF-IDF terms in Document 1:")
    for term, score in top_doc_terms:
        print(f"    '{term:15s}'  tfidf={score:.5f}")

    # Sample IDF values
    print(f"\n  Sample IDF scores:")
    for term in ['aerodynam', 'flow', 'boundari', 'heat', 'stabil']:
        if term in idf_scores:
            print(f"    '{term:15s}'  idf={idf_scores[term]:.4f}")

    print(f"\n✅ Phase 3 & 4 Complete — Inverted index and TF-IDF ready!")
    print("=" * 58)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    DOCS_PATH    = '/mnt/user-data/uploads/data/cran.all.1400'
    QUERIES_PATH = '/mnt/user-data/uploads/data/cran.qry'
    QRELS_PATH   = '/mnt/user-data/uploads/cran_data/cranqrel'

    print("Loading and preprocessing...")
    documents = parse_documents(DOCS_PATH)
    queries   = parse_queries(QUERIES_PATH)

    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    print("Building inverted index...")
    inverted_index = build_inverted_index(proc_docs)

    print("Computing TF-IDF...")
    tfidf_matrix, idf_scores, doc_lengths = build_tfidf_index(inverted_index, proc_docs)

    verify(inverted_index, tfidf_matrix, idf_scores, doc_lengths)
