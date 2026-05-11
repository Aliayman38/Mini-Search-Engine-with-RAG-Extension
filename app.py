"""
app.py — Cranfield Search Engine Web App
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import sys, os

# ── Import all project modules ───────────────────────────────────
from cranfield_parser       import parse_documents, parse_queries, parse_qrels
from cranfield_preprocessor import TextPreprocessor
from cranfield_tfidf        import build_inverted_index, build_tfidf_index
from cranfield_retrieval    import retrieve, retrieve_all
from cranfield_evaluation   import (mean_average_precision, mean_ndcg,
                                    average_precision, ndcg_at_k,
                                    precision_at_k, recall_at_k)
from cranfield_rag          import generate_rag_answer

app = Flask(__name__)

# ── File paths ───────────────────────────────────────────────────
DOCS_PATH    = 'data/cran.all.1400'
QUERIES_PATH = 'data/cran.qry'
QRELS_PATH   = 'data/cranqrel'

# ── Global state (loaded once at startup) ────────────────────────
print("\n🔧 Building index, please wait...\n")
documents      = parse_documents(DOCS_PATH)
queries        = parse_queries(QUERIES_PATH)
qrels          = parse_qrels(QRELS_PATH)
preprocessor   = TextPreprocessor()
proc_docs      = preprocessor.preprocess_documents(documents)
proc_queries   = preprocessor.preprocess_queries(queries)
inverted_index = build_inverted_index(proc_docs)
tfidf_matrix, idf_scores, _ = build_tfidf_index(inverted_index, proc_docs)

# Pre-compute global MAP & NDCG over all queries
all_results_full = retrieve_all(proc_queries, tfidf_matrix, idf_scores, inverted_index, k=1400)
all_results_10   = retrieve_all(proc_queries, tfidf_matrix, idf_scores, inverted_index, k=10)
GLOBAL_MAP,  _   = mean_average_precision(all_results_full, qrels)
GLOBAL_NDCG, _   = mean_ndcg(all_results_10, qrels, 10)

print(f"\n✅ Ready!  MAP={GLOBAL_MAP:.4f}  NDCG@10={GLOBAL_NDCG:.4f}")
print("🌐 Open http://localhost:5000\n")

# ── Sample queries for the UI ────────────────────────────────────
SAMPLE_QUERIES = [
    queries[1], queries[2], queries[3],
    queries[9], queries[15], queries[18],
]


# ════════════════════════════════════════════════════════════════
# HTML PAGE
# ════════════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cranfield Search Engine</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5;
         color: #1a1a1a; line-height: 1.6; }
  a { color: inherit; }

  /* ── Header ── */
  header { background: #1a1a2e; color: #fff; padding: 1.5rem 2rem; }
  header h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: .2rem; }
  header p  { font-size: .85rem; color: #aaa; }
  .badge-row { display: flex; gap: .5rem; margin-top: .75rem; flex-wrap: wrap; }
  .badge { background: #2a2a4e; color: #ccc; font-size: .75rem;
           padding: .2rem .6rem; border-radius: 20px; }
  .badge strong { color: #7eb8f7; }

  /* ── Main layout ── */
  main { max-width: 860px; margin: 0 auto; padding: 1.5rem 1rem 4rem; }

  /* ── Search box ── */
  .search-card { background: #fff; border-radius: 10px; padding: 1.25rem;
                 box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1rem; }
  .search-row  { display: flex; gap: .5rem; }
  .search-row input { flex: 1; padding: .55rem .9rem; border: 1px solid #ddd;
                      border-radius: 7px; font-size: .95rem; }
  .search-row input:focus { outline: none; border-color: #4a7fe5; }
  .search-row button { padding: .55rem 1.2rem; background: #1a1a2e; color: #fff;
                       border: none; border-radius: 7px; cursor: pointer;
                       font-size: .95rem; white-space: nowrap; }
  .search-row button:hover { background: #2a2a5e; }

  .controls { display: flex; gap: 1rem; margin-top: .75rem;
              align-items: center; flex-wrap: wrap; }
  .controls label { font-size: .82rem; color: #555; display: flex;
                    align-items: center; gap: .35rem; }
  .controls select { padding: .25rem .5rem; border: 1px solid #ddd;
                     border-radius: 5px; font-size: .82rem; }

  /* ── Samples ── */
  .samples-title { font-size: .8rem; color: #888; margin-bottom: .4rem; }
  .samples { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1rem; }
  .sample-btn { font-size: .78rem; padding: .25rem .7rem; border: 1px solid #ccc;
                border-radius: 20px; background: #fff; cursor: pointer;
                color: #444; max-width: 280px; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
  .sample-btn:hover { background: #f0f0f0; border-color: #999; }

  /* ── Metrics row ── */
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr);
             gap: .6rem; margin-bottom: 1rem; }
  .metric  { background: #fff; border-radius: 8px; padding: .75rem 1rem;
             box-shadow: 0 1px 3px rgba(0,0,0,.07); }
  .metric .label { font-size: .72rem; color: #888; margin-bottom: .1rem; }
  .metric .value { font-size: 1.3rem; font-weight: 600; color: #1a1a2e; }

  /* ── Results section ── */
  .section { background: #fff; border-radius: 10px; margin-bottom: 1rem;
             box-shadow: 0 1px 4px rgba(0,0,0,.07); overflow: hidden; }
  .section-hd { padding: .65rem 1rem; background: #f8f8fb;
                border-bottom: 1px solid #eee; font-size: .82rem;
                font-weight: 600; color: #555; display: flex;
                justify-content: space-between; align-items: center; }
  .tag { font-size: .72rem; background: #eef; color: #447;
         padding: .15rem .55rem; border-radius: 20px; font-weight: 400; }

  /* ── Doc rows ── */
  .doc-row { display: flex; align-items: flex-start; gap: .75rem;
             padding: .7rem 1rem; border-bottom: 1px solid #f2f2f2; }
  .doc-row:last-child { border-bottom: none; }
  .rank { width: 24px; height: 24px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: .7rem;
          font-weight: 700; flex-shrink: 0; margin-top: 2px; }
  .r1 { background: #dbeafe; color: #1d4ed8; }
  .r2 { background: #dcfce7; color: #166534; }
  .r3 { background: #fef9c3; color: #854d0e; }
  .rn { background: #f3f4f6; color: #6b7280; }
  .doc-body { flex: 1; min-width: 0; }
  .doc-title { font-size: .88rem; font-weight: 600;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .doc-id    { font-size: .75rem; color: #888; margin-bottom: .15rem; }
  .doc-abstract { font-size: .78rem; color: #555; margin-top: .2rem;
                  display: -webkit-box; -webkit-line-clamp: 2;
                  -webkit-box-orient: vertical; overflow: hidden; }
  .score-wrap { display: flex; flex-direction: column; align-items: flex-end;
                flex-shrink: 0; gap: .3rem; }
  .score-num  { font-size: .78rem; font-weight: 600; color: #4a7fe5; }
  .score-bar-bg { width: 60px; height: 4px; background: #e5e7eb; border-radius: 2px; }
  .score-bar    { height: 4px; background: #4a7fe5; border-radius: 2px; }
  .rel-badge { font-size: .68rem; padding: .1rem .45rem; border-radius: 10px;
               font-weight: 600; }
  .rel-1 { background: #dcfce7; color: #166534; }
  .rel-2 { background: #dbeafe; color: #1d4ed8; }
  .rel-3 { background: #fef9c3; color: #854d0e; }
  .rel-4 { background: #f3f4f6; color: #6b7280; }
  .rel-n { background: #fef2f2; color: #991b1b; }

  /* ── RAG answer ── */
  .rag-body { padding: 1rem; font-size: .88rem; line-height: 1.75;
              white-space: pre-wrap; color: #222; }

  /* ── Loading / empty ── */
  .loading, .empty { padding: 2rem; text-align: center;
                     color: #888; font-size: .88rem; }
  .spinner { display: inline-block; width: 18px; height: 18px;
             border: 2px solid #ddd; border-top-color: #4a7fe5;
             border-radius: 50%; animation: spin .7s linear infinite;
             vertical-align: middle; margin-right: .4rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { background: #fef2f2; color: #991b1b; border-radius: 8px;
           padding: .75rem 1rem; font-size: .85rem; margin-bottom: 1rem; }
</style>
</head>
<body>

<header>
  <h1>🔍 Cranfield Mini Search Engine</h1>
  <p>AI356 Information Retrieval · Vector Space Model + TF-IDF + RAG</p>
  <div class="badge-row">
    <span class="badge">1,400 documents</span>
    <span class="badge">225 queries</span>
    <span class="badge">MAP <strong>{{ "%.4f"|format(map_score) }}</strong></span>
    <span class="badge">NDCG@10 <strong>{{ "%.4f"|format(ndcg_score) }}</strong></span>
  </div>
</header>

<main>

  <!-- Search box -->
  <div class="search-card">
    <div class="search-row">
      <input id="q" type="text" placeholder="e.g. boundary layer turbulent flow flat plate..." />
      <button onclick="search()">Search</button>
    </div>
    <div class="controls">
      <label>Top-k &nbsp;
        <select id="topk">
          <option>5</option><option selected>10</option>
          <option>15</option><option>20</option>
        </select>
      </label>
      <label>RAG docs &nbsp;
        <select id="ragk">
          <option>3</option><option selected>5</option><option>10</option>
        </select>
      </label>
    </div>
  </div>


  <!-- API Key -->
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
              padding:10px 14px;background:#f8f8fb;border-radius:8px;
              border:1px solid #e0e0e8;">
    <label style="font-size:.82rem;color:#555;white-space:nowrap;font-weight:500;">
      🔑 Anthropic API Key
    </label>
    <input id="apikey" type="password" placeholder="sk-ant-..."
      style="flex:1;padding:.35rem .7rem;border:1px solid #ddd;border-radius:6px;
             font-size:.82rem;font-family:monospace;" />
    <span style="font-size:.75rem;color:#aaa;white-space:nowrap">Free key at aistudio.google.com</span>
  </div>

  <!-- Sample queries -->
  <p class="samples-title">Try a sample query:</p>
  <div class="samples" id="samples"></div>

  <!-- Error box -->
  <div class="error" id="err" style="display:none"></div>

  <!-- Metrics -->
  <div class="metrics" id="metrics" style="display:none">
    <div class="metric"><div class="label">AP (this query)</div>
      <div class="value" id="m-ap">—</div></div>
    <div class="metric"><div class="label">NDCG@k</div>
      <div class="value" id="m-ndcg">—</div></div>
    <div class="metric"><div class="label">Precision@k</div>
      <div class="value" id="m-pk">—</div></div>
    <div class="metric"><div class="label">Recall@k</div>
      <div class="value" id="m-rk">—</div></div>
  </div>

  <!-- Results -->
  <div id="results-section" style="display:none">
    <div class="section">
      <div class="section-hd">
        <span>Retrieved Documents</span>
        <span class="tag" id="res-count"></span>
      </div>
      <div id="doc-list"></div>
    </div>
    <div class="section">
      <div class="section-hd">
        <span>RAG Answer · Google Gemini 2.5 Flash</span>
        <span class="tag" id="rag-tag">generating...</span>
      </div>
      <div id="rag-out"><div class="loading">
        <span class="spinner"></span>Generating answer...
      </div></div>
    </div>
  </div>

  <div class="empty" id="empty-state">
    Enter a query above to search 1,400 aerodynamics papers.
  </div>

</main>

<script>
const SAMPLES = {{ samples|tojson }};
const samplesEl = document.getElementById('samples');
SAMPLES.forEach(q => {
  const b = document.createElement('button');
  b.className = 'sample-btn';
  b.title = q;
  b.textContent = q.length > 70 ? q.slice(0,67)+'...' : q;
  b.onclick = () => { document.getElementById('q').value = q; search(); };
  samplesEl.appendChild(b);
});

document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter') search();
});

function rankClass(r){ return r===1?'r1':r===2?'r2':r===3?'r3':'rn'; }
function relLabel(s){
  if(s===null) return '';
  const map={1:'rel-1',2:'rel-2',3:'rel-3',4:'rel-4'};
  const txt={1:'score 1 — best',2:'score 2 — high',3:'score 3 — useful',4:'score 4 — marginal'};
  return `<span class="rel-badge ${map[s]||'rel-n'}">${txt[s]||'not relevant'}</span>`;
}

async function search(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  const k    = document.getElementById('topk').value;
  const ragk = document.getElementById('ragk').value;

  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('err').style.display = 'none';
  document.getElementById('results-section').style.display = 'block';
  document.getElementById('metrics').style.display = 'grid';
  document.getElementById('doc-list').innerHTML =
    '<div class="loading"><span class="spinner"></span>Searching...</div>';
  document.getElementById('rag-out').innerHTML =
    '<div class="loading"><span class="spinner"></span>Generating answer...</div>';
  document.getElementById('rag-tag').textContent = 'generating...';

  try {
    const res = await fetch('/search', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query: q, k: parseInt(k), rag_k: parseInt(ragk), api_key: document.getElementById('apikey').value.trim()})
    });
    const data = await res.json();
    if(data.error){ showErr(data.error); return; }

    // Metrics
    document.getElementById('m-ap').textContent   = data.metrics.ap.toFixed(4);
    document.getElementById('m-ndcg').textContent = data.metrics.ndcg.toFixed(4);
    document.getElementById('m-pk').textContent   = data.metrics.pk.toFixed(4);
    document.getElementById('m-rk').textContent   = data.metrics.rk.toFixed(4);
    document.getElementById('res-count').textContent = `${data.results.length} results`;

    // Doc list
    const maxScore = data.results.length ? data.results[0].score : 1;
    document.getElementById('doc-list').innerHTML = data.results.map(d => {
      const barW = Math.round((d.score / maxScore) * 56);
      return `<div class="doc-row">
        <div class="rank ${rankClass(d.rank)}">${d.rank}</div>
        <div class="doc-body">
          <div class="doc-id">Doc ${d.doc_id} &nbsp;${relLabel(d.rel)}</div>
          <div class="doc-title">${d.title}</div>
          <div class="doc-abstract">${d.abstract}</div>
        </div>
        <div class="score-wrap">
          <span class="score-num">${d.score.toFixed(4)}</span>
          <div class="score-bar-bg">
            <div class="score-bar" style="width:${barW}px"></div>
          </div>
        </div>
      </div>`;
    }).join('');

    // RAG
    document.getElementById('rag-out').innerHTML =
      `<div class="rag-body">${data.rag_answer}</div>`;
    document.getElementById('rag-tag').textContent = 'done';

  } catch(e) {
    showErr('Request failed: ' + e.message);
  }
}

function showErr(msg){
  const b = document.getElementById('err');
  b.textContent = msg; b.style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
}
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(
        HTML,
        map_score=GLOBAL_MAP,
        ndcg_score=GLOBAL_NDCG,
        samples=SAMPLE_QUERIES
    )


@app.route('/search', methods=['POST'])
def search():
    data    = request.get_json()
    query   = data.get('query', '').strip()
    k       = int(data.get('k', 10))
    rag_k   = int(data.get('rag_k', 5))
    api_key = data.get('api_key', '').strip()

    if not query:
        return jsonify({'error': 'Empty query'})

    # ── Preprocess & retrieve ────────────────────────────────────
    tokens   = preprocessor.preprocess(query)
    top_docs = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k=k)
    top_full = retrieve(tokens, tfidf_matrix, idf_scores, inverted_index, k=1400)

    # ── Find matching query ID for relevance lookup ──────────────
    qid = None
    for qid_cand, q_text in queries.items():
        if q_text.strip().lower() == query.strip().lower():
            qid = qid_cand
            break
    rel_map = qrels.get(qid, {})

    # ── Per-query metrics ────────────────────────────────────────
    ap   = average_precision(top_full, rel_map)
    ndcg = ndcg_at_k(top_docs, rel_map, k)
    pk   = precision_at_k(top_docs, rel_map, k)
    rk   = recall_at_k(top_docs, rel_map, k)

    # ── Build result list ────────────────────────────────────────
    results = []
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        doc     = documents[doc_id]
        title   = doc['title'].replace('\n', ' ').strip()
        abstract = doc['abstract'].replace('\n', ' ').strip()
        rel      = rel_map.get(doc_id, None)
        results.append({
            'rank': rank, 'doc_id': doc_id,
            'score': round(score, 5),
            'title': title,
            'abstract': abstract[:200] + ('...' if len(abstract) > 200 else ''),
            'rel': rel
        })

    # ── RAG ─────────────────────────────────────────────────────
    rag_result = generate_rag_answer(query, top_docs[:rag_k], documents, api_key=api_key)

    return jsonify({
        'results':    results,
        'metrics':    {'ap': ap, 'ndcg': ndcg, 'pk': pk, 'rk': rk},
        'rag_answer': rag_result['rag_answer'],
    })


# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=False, port=5000)
