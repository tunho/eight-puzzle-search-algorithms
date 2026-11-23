"""Final experiments: evaluate named configurations on validation and test.

Parameters are fixed beforehand on the training split (scripts/tune.py);
the test split is only used here, for reporting.

Usage::

    python scripts/run_experiments.py                       # all configs, validation + test
    python scripts/run_experiments.py --configs bm25,bm25_rm3 --splits test

Outputs (results/)::

    runs/<split>/<config>.run.gz              TREC run files (top 1000 per query)
    metrics/<split>/<config>.json             mean + per-query metrics
    metrics/<split>/<config>.trec_eval.txt    same, in `trec_eval -q` layout
    experiments.md                            result tables + significance tests
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irsearch.evaluation import (  # noqa: E402
    MEASURES, evaluate, load_qrels, load_queries, paired_randomization_test,
    relevance_level_analysis, save_metrics, to_run, write_run, write_trec_eval_style,
)
from irsearch.index import InvertedIndex  # noqa: E402
from irsearch.presets import load_best  # noqa: E402
from irsearch.runner import run_queries  # noqa: E402
from irsearch.searcher import Searcher, SearchParams  # noqa: E402

RES = "results"


def configs(best):
    bm = best["bm25"]
    k1, b = bm["k1"], bm["b"]
    pw = best["prox"]["prox_weight"]
    r = best["rm3"]
    fb = dict(fb_docs=r["fb_docs"], fb_terms=r["fb_terms"], fb_orig_weight=r["fb_orig_weight"],
              fb_max_df=r.get("fb_max_df", 1.0))
    # name -> (description, index, params)
    return {
        "tfidf": ("TF-IDF cosine (ltc.ltc)", "stem", SearchParams(model="tfidf")),
        "ql": (f"QL Dirichlet (mu={best['ql']['mu']:g})", "stem", SearchParams(model="ql", mu=best["ql"]["mu"])),
        "bm25_default": ("BM25 default (k1=1.2, b=0.75)", "stem", SearchParams(k1=1.2, b=0.75)),
        "bm25": (f"BM25 tuned (k1={k1:g}, b={b:g})", "stem", SearchParams(k1=k1, b=b)),
        "bm25_nostop": ("BM25 tuned, no stopword removal", "stem", SearchParams(k1=k1, b=b, remove_stopwords=False)),
        "bm25_nostem": ("BM25 tuned, no stemming", "nostem", SearchParams(k1=k1, b=b)),
        "bm25_prox": (f"BM25 + proximity (w={pw:g})", "stem", SearchParams(k1=k1, b=b, prox_weight=pw)),
        "bm25_rm3": (
            f"BM25 + RM3 ({fb['fb_docs']} docs, {fb['fb_terms']} terms, orig={fb['fb_orig_weight']:g}, "
            f"maxdf={fb['fb_max_df']:g})",
            "stem", SearchParams(k1=k1, b=b, rm3=True, **fb),
        ),
        "bm25_prox_rm3": ("BM25 + proximity + RM3", "stem", SearchParams(k1=k1, b=b, prox_weight=pw, rm3=True, **fb)),
    }


def fmt(v):
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None, help="comma separated (default: all)")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--index", default="index/wikir_en1k")
    ap.add_argument("--index-nostem", default="index/wikir_en1k_nostem")
    args = ap.parse_args()

    best = load_best()
    if not best:
        sys.exit("results/tuning/best_params.json missing - run scripts/tune.py first")
    cfgs = configs(best)
    names = args.configs.split(",") if args.configs else list(cfgs)
    splits = args.splits.split(",")
    searchers = {"stem": Searcher(InvertedIndex(args.index))}
    if any(cfgs[n][1] == "nostem" for n in names):
        if os.path.exists(os.path.join(args.index_nostem, "meta.json")):
            searchers["nostem"] = Searcher(InvertedIndex(args.index_nostem))
        else:
            print("no-stem index missing -> skipping bm25_nostem (build with --no-stem)")
            names = [n for n in names if cfgs[n][1] != "nostem"]

    summary = {}
    for split in splits:
        queries, qrels = load_queries(split), load_qrels(split)
        summary[split] = {}
        for n in names:
            desc, which, p = cfgs[n]
            t0 = time.time()
            ranked, ms = run_queries(searchers[which], queries, p, k=1000)
            run = to_run(ranked)
            mean, perq = evaluate(qrels, run)
            extra = relevance_level_analysis(qrels, run)
            info = {"config": n, "description": desc, "params": vars(p), "index": which, "split": split,
                    "n_queries": len(queries), "ms_per_query": ms}
            tmp = os.path.join(RES, "tmp", f"{split}_{n}.run")
            write_run(tmp, ranked, tag=n)
            os.makedirs(os.path.join(RES, "runs", split), exist_ok=True)
            with open(tmp, "rb") as fi, gzip.open(os.path.join(RES, "runs", split, f"{n}.run.gz"), "wb") as fo:
                shutil.copyfileobj(fi, fo)
            save_metrics(os.path.join(RES, "metrics", split, f"{n}.json"), {**mean, **extra}, perq, info)
            write_trec_eval_style(os.path.join(RES, "metrics", split, f"{n}.trec_eval.txt"), mean, perq, n)
            summary[split][n] = {"desc": desc, "mean": mean, "extra": extra, "perq": perq, "ms": ms}
            print(f"[{split}] {n:14s} " + " ".join(f"{MEASURES[m]}={mean[m]:.4f}" for m in MEASURES)
                  + f" rel2_MRR={extra['rel2_MRR']:.4f} rel1_R@1000={extra['rel1_R@1000']:.4f}"
                  + f" ({ms:.1f} ms/q, {time.time() - t0:.0f}s)", flush=True)

    write_report(summary, names, args.baseline, best)


def write_report(summary, names, baseline, best):
    lines = ["# Experiment results (wikir/en1k)", ""]
    lines.append("Parameters were tuned on the **training** split (1,444 queries, objective: MAP) "
                 "with `scripts/tune.py`; validation and test were not used for tuning.")
    lines.append("")
    lines.append("Tuned parameters: `" + json.dumps(best) + "`")
    lines.append("")
    lines.append(f"Significance: paired randomization test (20,000 sign flips, two-sided) against `{baseline}`; "
                 "† p < 0.05, ‡ p < 0.01.")
    lines.append("")
    cols = list(MEASURES)
    for split, res in summary.items():
        lines.append(f"## {split} ({len(next(iter(res.values()))['perq'])} queries)")
        lines.append("")
        lines.append("| config | " + " | ".join(MEASURES[c] for c in cols)
                     + " | rel2 MRR | rel2 S@1 | rel1 R@1000 | ms/query |")
        lines.append("|---|" + "---|" * (len(cols) + 4))
        base = res.get(baseline)
        for n in names:
            if n not in res:
                continue
            r = res[n]
            cells = []
            for c in cols:
                mark = ""
                if base is not None and n != baseline:
                    qs = list(r["perq"])
                    pv = paired_randomization_test([r["perq"][q][c] for q in qs], [base["perq"][q][c] for q in qs])
                    mark = "‡" if pv < 0.01 else ("†" if pv < 0.05 else "")
                best_val = max(res[m]["mean"][c] for m in res)
                val = fmt(r["mean"][c]) + mark
                cells.append(f"**{val}**" if abs(r["mean"][c] - best_val) < 1e-12 else val)
            e = r["extra"]
            lines.append(f"| `{n}` {r['desc']} | " + " | ".join(cells)
                         + f" | {fmt(e['rel2_MRR'])} | {fmt(e['rel2_S@1'])} | {fmt(e['rel1_R@1000'])} | {r['ms']:.1f} |")
        lines.append("")
    with open(os.path.join(RES, "experiments.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/experiments.md")


if __name__ == "__main__":
    main()
