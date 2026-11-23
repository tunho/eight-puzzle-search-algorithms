"""Grid-search ranking parameters on the TRAINING split (never on test).

Usage::

    python scripts/tune.py                 # bm25 -> ql -> proximity -> rm3
    python scripts/tune.py --what bm25 --metric map

Writes every evaluated configuration to ``results/tuning/<what>.json`` and the
best parameters to ``results/tuning/best_params.json``.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irsearch.evaluation import evaluate, load_qrels, load_queries, to_run  # noqa: E402
from irsearch.index import InvertedIndex  # noqa: E402
from irsearch.runner import run_queries  # noqa: E402
from irsearch.searcher import Searcher, SearchParams  # noqa: E402

OUT = os.path.join("results", "tuning")


def evaluate_params(searcher, queries, qrels, p):
    ranked, ms = run_queries(searcher, queries, p, k=1000)
    mean, _ = evaluate(qrels, to_run(ranked))
    return mean, ms


def grid(searcher, queries, qrels, name, configs, metric, log):
    rows = []
    t0 = time.time()
    for i, p in enumerate(configs, 1):
        mean, ms = evaluate_params(searcher, queries, qrels, p)
        rows.append({"params": vars(p).copy(), "metrics": mean, "ms_per_query": ms})
        log(f"[{name} {i}/{len(configs)}] {p.describe():70s} {metric}={mean[metric]:.4f} "
            f"nDCG@10={mean['ndcg_cut_10']:.4f} ({time.time() - t0:.0f}s)")
    rows.sort(key=lambda r: -r["metrics"][metric])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"metric": metric, "n_queries": len(queries), "rows": rows}, f, indent=1)
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index/wikir_en1k")
    ap.add_argument("--split", default="training")
    ap.add_argument("--what", default="all", choices=["bm25", "ql", "prox", "rm3", "rm3df", "all"])
    ap.add_argument("--metric", default="map")
    args = ap.parse_args()

    idx = InvertedIndex(args.index)
    s = Searcher(idx)
    queries = load_queries(args.split)
    qrels = load_qrels(args.split)
    print(f"tuning on {args.split}: {len(queries)} queries, metric={args.metric}", flush=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    best_path = os.path.join(OUT, "best_params.json")
    best = {}
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)

    def save():
        os.makedirs(OUT, exist_ok=True)
        with open(best_path, "w") as f:
            json.dump(best, f, indent=1)

    whats = ["bm25", "ql", "prox", "rm3"] if args.what == "all" else [args.what]
    if "bm25" in whats:
        k1s = [0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2]
        bs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cfg = [SearchParams(k1=k1, b=b) for k1, b in itertools.product(k1s, bs)]
        r = grid(s, queries, qrels, "bm25", cfg, args.metric, log)
        best["bm25"] = {"k1": r["params"]["k1"], "b": r["params"]["b"], args.metric: r["metrics"][args.metric]}
        save()
    k1, b = best.get("bm25", {}).get("k1", 0.9), best.get("bm25", {}).get("b", 0.4)
    if "ql" in whats:
        cfg = [SearchParams(model="ql", mu=mu) for mu in [100, 200, 300, 500, 800, 1000, 1500, 2000, 3000]]
        r = grid(s, queries, qrels, "ql", cfg, args.metric, log)
        best["ql"] = {"mu": r["params"]["mu"], args.metric: r["metrics"][args.metric]}
        save()
    if "prox" in whats:
        cfg = [SearchParams(k1=k1, b=b, prox_weight=w) for w in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]]
        r = grid(s, queries, qrels, "prox", cfg, args.metric, log)
        best["prox"] = {"k1": k1, "b": b, "prox_weight": r["params"]["prox_weight"], args.metric: r["metrics"][args.metric]}
        save()
    if "rm3" in whats:
        cfg = [
            SearchParams(k1=k1, b=b, rm3=True, fb_docs=d, fb_terms=t, fb_orig_weight=w)
            for d, t, w in itertools.product([5, 10, 20], [10, 20, 40], [0.3, 0.5, 0.7])
        ]
        r = grid(s, queries, qrels, "rm3", cfg, args.metric, log)
        best["rm3"] = {k: r["params"][k] for k in ("k1", "b", "fb_docs", "fb_terms", "fb_orig_weight")}
        best["rm3"]["fb_max_df"] = 1.0
        best["rm3"][args.metric] = r["metrics"][args.metric]
        save()
    if "rm3df" in whats:
        # second stage: filter generic feedback terms by document frequency
        r0 = best["rm3"]
        cfg = [
            SearchParams(k1=k1, b=b, rm3=True, fb_docs=r0["fb_docs"], fb_terms=t, fb_orig_weight=w, fb_max_df=m)
            for m, t, w in itertools.product([0.3, 0.1, 0.05], [10, 20, 40], [0.3, 0.5, 0.7])
        ]
        r = grid(s, queries, qrels, "rm3df", cfg, args.metric, log)
        if r["metrics"][args.metric] > r0[args.metric]:
            best["rm3"] = {k: r["params"][k] for k in ("k1", "b", "fb_docs", "fb_terms", "fb_orig_weight", "fb_max_df")}
            best["rm3"][args.metric] = r["metrics"][args.metric]
        save()
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
