"""
Phase 1: Cranfield Dataset Parser
Parses:
  - cran_all.1400  → documents (id, title, abstract)
  - data/cran.qry       → queries (id, text)
  - data/cranqrel       → relevance judgments (query_id → {doc_id → score})
"""

import re
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. Document Parser
# ─────────────────────────────────────────────
def parse_documents(filepath: str) -> dict:
    """
    Parse cran_all.1400 file.
    Each document has fields: .I (id), .T (title), .A (author), .B (bib), .W (abstract)
    We use title + abstract as document content.

    Returns:
        dict: {doc_id (int) -> {'title': str, 'abstract': str, 'content': str}}
    """
    documents = {}

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Split on .I markers
    # Each block starts with ".I <number>"
    blocks = re.split(r'\.I\s+(\d+)', raw)

    # blocks[0] is anything before the first .I (empty or junk)
    # then pairs: blocks[1]=id, blocks[2]=content, blocks[3]=id, blocks[4]=content ...
    i = 1
    while i < len(blocks) - 1:
        doc_id = int(blocks[i].strip())
        block_text = blocks[i + 1]

        # Extract .T (title)
        title_match = re.search(r'\.T\s*(.*?)(?=\.[ABWI]|\Z)', block_text, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ''

        # Extract .W (abstract)
        abstract_match = re.search(r'\.W\s*(.*?)(?=\.[TABWI]|\Z)', block_text, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ''

        # Combine title + abstract as the document content for indexing
        content = f"{title} {abstract}".strip()

        documents[doc_id] = {
            'title': title,
            'abstract': abstract,
            'content': content
        }

        i += 2

    return documents


# ─────────────────────────────────────────────
# 2. Query Parser
# ─────────────────────────────────────────────
def parse_queries(filepath: str) -> dict:
    """
    Parse data/cran.qry file.
    Each query has fields: .I (id), .W (query text)

    Returns:
        dict: {query_id (int) -> query_text (str)}
    """
    queries = {}

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Split on .I markers
    blocks = re.split(r'\.I\s+(\d+)', raw)

    # data/cranqrel uses sequential positions (1-225), NOT the .I numbers.
    # So we assign query_id by position (1-indexed), not by the .I field.
    position = 1
    i = 1
    while i < len(blocks) - 1:
        block_text = blocks[i + 1]

        # Extract .W (query text)
        w_match = re.search(r'\.W\s*(.*?)(?=\.[IW]|\Z)', block_text, re.DOTALL)
        query_text = w_match.group(1).strip() if w_match else ''

        # Clean up: normalize whitespace
        query_text = re.sub(r'\s+', ' ', query_text)

        queries[position] = query_text
        position += 1
        i += 2

    return queries


# ─────────────────────────────────────────────
# 3. Relevance Judgments Parser
# ─────────────────────────────────────────────
def parse_qrels(filepath: str) -> dict:
    """
    Parse data/cranqrel file.
    Format: query_id  doc_id  relevance_score
    Scores: 1=highest, 2=high, 3=useful, 4=minimum, -1=not relevant

    Returns:
        dict: {query_id (int) -> {doc_id (int) -> score (int)}}
    """
    qrels = defaultdict(dict)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                score = int(parts[2])
                qrels[query_id][doc_id] = score

    return dict(qrels)


# ─────────────────────────────────────────────
# 4. Verification
# ─────────────────────────────────────────────
def verify_parsing(documents, queries, qrels):
    print("=" * 55)
    print("       PHASE 1: PARSING VERIFICATION")
    print("=" * 55)

    print(f"\n📄 Documents loaded   : {len(documents):,}")
    print(f"🔍 Queries loaded     : {len(queries):,}")
    print(f"✅ Qrel query entries  : {len(qrels):,}")

    # Sample document
    sample_doc_id = 1
    doc = documents[sample_doc_id]
    print(f"\n--- Sample Document (ID={sample_doc_id}) ---")
    print(f"  Title    : {doc['title'][:80]}")
    print(f"  Abstract : {doc['abstract'][:120]}...")
    print(f"  Content  : {len(doc['content'])} chars")

    # Sample query
    sample_qid = list(queries.keys())[0]
    print(f"\n--- Sample Query (ID={sample_qid}) ---")
    print(f"  Text: {queries[sample_qid]}")

    # Sample qrels
    sample_q = list(qrels.keys())[0]
    rel_docs = qrels[sample_q]
    print(f"\n--- Sample Qrels (Query {sample_q}) ---")
    print(f"  Relevant docs: {len(rel_docs)} documents")
    for did, score in list(rel_docs.items())[:5]:
        print(f"    Doc {did:4d}  →  score={score}")

    # Qrel score distribution
    all_scores = [s for q in qrels.values() for s in q.values()]
    from collections import Counter
    score_dist = Counter(all_scores)
    print(f"\n--- Relevance Score Distribution ---")
    for score in sorted(score_dist):
        label = {1: "Complete answer", 2: "High relevance",
                 3: "Useful background", 4: "Minimum interest", -1: "Not relevant"}
        print(f"  Score {score:2d} ({label.get(score,'?'):20s}): {score_dist[score]:4d} entries")

    print(f"\n✅ Phase 1 Complete — all files parsed successfully!")
    print("=" * 55)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    DOCS_PATH   = '/mnt/user-data/uploads/data/cran.all.1400'
    QUERIES_PATH = '/mnt/user-data/uploads/data/cran.qry'
    QRELS_PATH  = '/mnt/user-data/uploads/cran_data/cranqrel'

    print("Parsing documents...")
    documents = parse_documents(DOCS_PATH)

    print("Parsing queries...")
    queries = parse_queries(QUERIES_PATH)

    print("Parsing relevance judgments...")
    qrels = parse_qrels(QRELS_PATH)

    verify_parsing(documents, queries, qrels)
