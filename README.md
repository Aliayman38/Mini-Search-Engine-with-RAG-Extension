# Cranfield Mini Search Engine with RAG Extension

A mini search engine built on the Cranfield dataset using classical IR techniques (TF-IDF + Cosine Similarity), evaluated with MAP and NDCG, and extended with a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini 2.5 Flash. Includes a Flask web interface.

---

## Results

| Metric | Score |
|---|---|
| MAP (Mean Average Precision) | 0.3070 |
| NDCG@10 | 0.3678 |
| Precision@10 | 0.2413 |
| Recall@10 | 0.4007 |

---
```
├── app.py                      # Flask web app (run this)
├── cranfield_parser.py         # Phase 1 — parse dataset files
├── cranfield_preprocessor.py   # Phase 2 — tokenize, stopwords, stemming
├── cranfield_tfidf.py          # Phase 3 & 4 — inverted index + TF-IDF
├── cranfield_retrieval.py      # Phase 5 — cosine similarity ranking
├── cranfield_evaluation.py     # Phase 6 — MAP and NDCG evaluation
├── cranfield_rag.py            # Phase 7 — generative RAG via Gemini API
├── main.py                     # CLI entry point
├── cran.all.1400               # 1400 documents
├── cran.qry                    # 225 queries
└── cranqrel                    # relevance judgments
```

## Installation

```bash
pip install flask nltk numpy requests
python -c "import nltk; nltk.download('stopwords')"
```

---

## Usage

### Web App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

Enter your Google Gemini API key in the web interface to enable RAG answer generation.
Get a free key at [Google AI Studio](https://aistudio.google.com).

### Command Line

```bash
# Interactive mode
python main.py

# Single query
python main.py --query "boundary layer turbulent flow flat plate"

# Full evaluation report
python main.py --eval
```

---

## How It Works

### Phase 1 — Parsing
Parses `cran.all.1400` (documents), `cran.qry` (queries), and `cranqrel` (relevance judgments).
Fixes the critical query-ID mapping bug where `cranqrel` uses sequential positions (1–225) but `cran.qry` uses non-sequential `.I` fields.

### Phase 2 — Preprocessing
Each document and query goes through: lowercasing → punctuation removal → tokenization → stopword removal → Porter stemming.

### Phase 3 & 4 — Inverted Index + TF-IDF
Builds an inverted index `{term → {doc_id → count}}` then computes TF-IDF weights:
- **TF** = term count / document length
- **IDF** = log(N / df) + 1

### Phase 5 — Retrieval
Converts the query into a TF-IDF vector and ranks documents by cosine similarity.

### Phase 6 — Evaluation
- **MAP** — binary relevance (scores 1–4 = relevant, missing = not relevant)
- **NDCG** — graded relevance (1→4, 2→3, 3→2, 4→1, missing→0)

### Phase 7 — RAG (Generative)
Retrieves the top-k documents via TF-IDF, then sends the query and retrieved document contexts to Google Gemini 2.5 Flash via API to synthesize a coherent, grounded natural-language answer. The LLM is instructed to answer using only the provided documents, significantly reducing hallucination risk compared to unconstrained generation.

---

## Dataset

The [Cranfield dataset](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/) is a standard IR benchmark containing 1,400 aerodynamics abstracts, 225 queries, and graded relevance judgments.

---

## API Key

RAG answer generation requires a Google Gemini API key:
- Get a free key at [Google AI Studio](https://aistudio.google.com)
- Enter it in the web interface before searching
- The key is sent with each request and is not stored server-side
