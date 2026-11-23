"""Web UI for the search engine (Flask).

Run from the ``ir_search_engine`` directory::

    python web/app.py                  # http://127.0.0.1:5000
    python web/app.py --port 8000 --index index/wikir_en1k
"""
import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, abort, jsonify, render_template, request  # noqa: E402

from irsearch.index import InvertedIndex  # noqa: E402
from irsearch.presets import presets  # noqa: E402
from irsearch.searcher import Searcher  # noqa: E402
from irsearch.snippets import highlight, make_snippet, make_title  # noqa: E402

PER_PAGE = 10
MAX_RESULTS = 1000

app = Flask(__name__, template_folder="templates", static_folder="static")
STATE = {}


def init(index_dir: str, default_model: str = "bm25") -> None:
    t0 = time.time()
    idx = InvertedIndex(index_dir)
    STATE["index"] = idx
    STATE["searcher"] = Searcher(idx)
    STATE["presets"] = presets()
    STATE["default_model"] = default_model
    STATE["load_seconds"] = time.time() - t0
    app.jinja_env.globals["surface"] = idx.surface


def _system_info():
    idx = STATE["index"]
    return {
        "n_docs": idx.N,
        "n_terms": idx.V,
        "n_postings": int(idx.meta["n_postings"]),
        "avgdl": idx.avgdl,
        "stem": idx.meta.get("stem"),
        "dataset": idx.meta.get("dataset"),
    }


def _query_display_terms(res):
    """Index terms to highlight: original query terms and expansion terms."""
    main = set()
    if res.parsed is not None:
        main = {t.term for t in res.parsed.tokens if not t.is_stop} or {t.term for t in res.parsed.tokens}
    return main, set(res.expansion_terms)


def _items(res, start, end, main, extra):
    idx = STATE["index"]
    an = idx.analyzer
    items = []
    for rank in range(start, min(end, len(res.doc_ints))):
        d = int(res.doc_ints[rank])
        text = idx.doc_text(d)
        items.append(
            {
                "rank": rank + 1,
                "doc_id": idx.ext_id(d),
                "title": make_title(text),
                "snippet": make_snippet(text, an, main, extra),
                "score": float(res.scores[rank]),
            }
        )
    return items


@app.route("/")
def search_page():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", STATE["default_model"])
    if model not in STATE["presets"]:
        model = STATE["default_model"]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    rel = [r for r in request.args.getlist("rel") if r]
    ctx = {
        "q": q, "model": model, "presets": STATE["presets"], "info": _system_info(),
        "page": page, "rel": rel, "res": None, "items": [], "pages": 0, "error": None,
    }
    if q:
        idx = STATE["index"]
        label, params = STATE["presets"][model]
        fb_ints = [idx.int_id(r) for r in rel if idx.int_id(r) is not None]
        res = STATE["searcher"].search(q, params, k=MAX_RESULTS, feedback_docs=fb_ints or None)
        main, extra = _query_display_terms(res)
        n = len(res.doc_ints)
        ctx["pages"] = math.ceil(n / PER_PAGE)
        page = min(page, max(1, ctx["pages"]))
        ctx["page"] = page
        ctx["items"] = _items(res, (page - 1) * PER_PAGE, page * PER_PAGE, main, extra)
        ctx["res"] = res
        ctx["dym"] = None if res.corrections else STATE["searcher"].did_you_mean(q)
        ctx["model_desc"] = params.describe() + (" + Relevance Feedback" if fb_ints else "")
        ctx["page_range"] = _page_range(page, ctx["pages"])
    return render_template("index.html", **ctx)


def _page_range(page, pages, width=10):
    lo = max(1, page - width // 2)
    hi = min(pages, lo + width - 1)
    lo = max(1, hi - width + 1)
    return list(range(lo, hi + 1))


@app.route("/doc/<doc_id>")
def doc_page(doc_id):
    idx = STATE["index"]
    d = idx.int_id(doc_id)
    if d is None:
        abort(404)
    q = request.args.get("q", "")
    an = idx.analyzer
    from irsearch.query_parser import parse_query

    pq = parse_query(q, an) if q else None
    terms = {t.term for t in pq.tokens if not t.is_stop} if pq else set()
    text = idx.doc_text(d)
    return render_template(
        "doc.html", doc_id=doc_id, title=make_title(text), body=highlight(text, an, terms),
        q=q, n_tokens=int(idx.doclen[d]), info=_system_info(),
    )


@app.route("/similar/<doc_id>")
def similar_page(doc_id):
    idx = STATE["index"]
    d = idx.int_id(doc_id)
    if d is None:
        abort(404)
    _, params = STATE["presets"]["bm25"]
    res = STATE["searcher"].more_like_this(d, k=PER_PAGE, params=params)
    terms = set(res.query_terms)
    items = _items(res, 0, PER_PAGE, terms, set())
    src = idx.doc_text(d)
    return render_template(
        "similar.html", doc_id=doc_id, src_title=make_title(src), items=items, res=res, info=_system_info()
    )


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", STATE["default_model"])
    if model not in STATE["presets"]:
        return jsonify({"error": f"unknown model {model}", "models": list(STATE["presets"])}), 400
    k = min(int(request.args.get("k", 10)), MAX_RESULTS)
    idx = STATE["index"]
    _, params = STATE["presets"][model]
    res = STATE["searcher"].search(q, params, k=k)
    main, extra = _query_display_terms(res)
    return jsonify(
        {
            "query": q, "model": params.describe(), "n_docs_indexed": idx.N, "n_matched": res.n_matched,
            "elapsed_ms": round(res.elapsed_ms, 2), "query_terms": res.query_terms,
            "expansion_terms": res.expansion_terms, "corrections": res.corrections,
            "results": _items(res, 0, k, main, extra),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "index", "wikir_en1k"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model", default="bm25", help="default ranking preset")
    args = ap.parse_args()
    init(args.index, args.model)
    print(f"index loaded in {STATE['load_seconds']:.1f}s: {STATE['index'].N:,} docs -> http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
