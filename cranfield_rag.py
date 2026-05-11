"""
Phase 7: RAG — Google Gemini API
Uses gemini-1.5-flash (free tier available).
Get your API key at: https://aistudio.google.com/app/apikey
"""

import json
import urllib.request
import urllib.error


# ─────────────────────────────────────────────
# Call Gemini API
# ─────────────────────────────────────────────
def call_gemini(prompt: str, api_key: str, max_tokens: int = 800) -> str:
    """
    Send a prompt to Gemini 2.5 Flash and return the response text.
    API key is passed as a query parameter — no special headers needed.
    """
    model = "gemini-2.5-flash"
    url   = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "You are a precise scientific assistant specializing in "
                    "aerodynamics and aerospace engineering. "
                    "Answer questions using ONLY the provided document excerpts. "
                    "Be concise and factual. Always cite the Document IDs that "
                    "support each point. "
                    "If the documents do not contain enough information, say so clearly."
                )
            }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {e}")


# ─────────────────────────────────────────────
# Build RAG Prompt
# ─────────────────────────────────────────────
def build_rag_prompt(query: str, top_docs: list, documents: dict) -> str:
    """
    Build the prompt that includes retrieved documents as context.
    """
    context_parts = []
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        doc      = documents.get(doc_id, {})
        title    = doc.get("title",    "").replace("\n", " ").strip()
        abstract = doc.get("abstract", "").replace("\n", " ").strip()

        if len(abstract) > 700:
            abstract = abstract[:700] + "..."

        context_parts.append(
            f"[Document {doc_id} | Rank {rank} | Cosine Score: {score:.4f}]\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )

    context = "\n\n".join(context_parts)

    prompt = (
        f"User Query:\n\"{query}\"\n\n"
        f"Retrieved Documents (top-{len(top_docs)}):\n"
        f"{'─' * 60}\n"
        f"{context}\n"
        f"{'─' * 60}\n\n"
        f"Using ONLY the documents above, write a comprehensive answer to the query. "
        f"For each point you make, mention which Document ID supports it. "
        f"Structure your answer clearly."
    )
    return prompt


# ─────────────────────────────────────────────
# Main RAG Function
# ─────────────────────────────────────────────
def generate_rag_answer(query: str,
                        top_docs: list,
                        documents: dict,
                        api_key: str = "",
                        **kwargs) -> dict:
    """
    Full RAG pipeline for a single query using Gemini.

    Args:
        query    : raw query string
        top_docs : [(doc_id, score), ...] top-k retrieved docs
        documents: full document collection
        api_key  : Google Gemini API key

    Returns:
        dict with keys: query, top_docs, rag_answer
    """
    if not api_key:
        return {
            "query":      query,
            "top_docs":   _build_docs_info(top_docs, documents),
            "rag_answer": (
                "⚠️ No API key provided. "
                "Get a free Gemini key at https://aistudio.google.com/app/apikey "
                "and paste it in the API Key field."
            )
        }

    prompt = build_rag_prompt(query, top_docs, documents)

    try:
        answer = call_gemini(prompt, api_key)
    except RuntimeError as e:
        answer = f"❌ Gemini error: {e}"

    return {
        "query":      query,
        "top_docs":   _build_docs_info(top_docs, documents),
        "rag_answer": answer
    }


def _build_docs_info(top_docs: list, documents: dict) -> list:
    result = []
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        title = documents.get(doc_id, {}).get("title", "").replace("\n", " ").strip()
        result.append({
            "rank":   rank,
            "doc_id": doc_id,
            "score":  round(score, 5),
            "title":  title
        })
    return result


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────
def verify_rag(documents, queries, proc_queries,
               tfidf_matrix, idf_scores, inverted_index,
               api_key, k=5):
    from cranfield_retrieval import retrieve

    print("=" * 65)
    print("     PHASE 7: RAG VIA GEMINI API — VERIFICATION")
    print("=" * 65)

    for qid in [1, 2]:
        raw_query = queries[qid]
        tokens    = proc_queries[qid]
        top_docs  = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k=k)

        print(f"\n{'─'*65}")
        print(f"Query {qid}: {raw_query}")
        print(f"\nTop-{k} retrieved docs:")
        for rank, (did, score) in enumerate(top_docs, 1):
            title = documents[did]['title'].replace('\n', ' ')[:55]
            print(f"  {rank}. [Doc {did:4d}]  {score:.4f}  {title}")

        print(f"\n🤖 Calling Gemini API...")
        result = generate_rag_answer(raw_query, top_docs, documents, api_key)
        print(f"\n💡 RAG Answer:\n")
        for line in result['rag_answer'].split('\n'):
            print(f"   {line}")

    print(f"\n{'='*65}")
    print("✅ Phase 7 Complete — Gemini RAG working!")
    print("=" * 65)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import os, sys
    from cranfield_parser import parse_documents, parse_queries
    from cranfield_preprocessor import TextPreprocessor
    from cranfield_tfidf import build_inverted_index, build_tfidf_index

    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not API_KEY:
        # Paste your key directly here if you prefer:
        # API_KEY = "AIza..."
        print("ERROR: Set your Gemini API key:")
        print("  export GEMINI_API_KEY=AIza...")
        print("  OR paste it directly in this file as API_KEY = 'AIza...'")
        sys.exit(1)

    DOCS_PATH    = 'cran_cran_all.1400'
    QUERIES_PATH = 'cran_cran.qry'

    print("Loading data...")
    documents    = parse_documents(DOCS_PATH)
    queries      = parse_queries(QUERIES_PATH)
    preprocessor = TextPreprocessor()
    proc_docs    = preprocessor.preprocess_documents(documents)
    proc_queries = preprocessor.preprocess_queries(queries)

    print("Building index...")
    inverted_index = build_inverted_index(proc_docs)
    tfidf_matrix, idf_scores, _ = build_tfidf_index(inverted_index, proc_docs)

    verify_rag(documents, queries, proc_queries,
               tfidf_matrix, idf_scores, inverted_index,
               API_KEY, k=5)
