"""TREC-style evaluation helpers.

* TREC run files:  ``qid Q0 docid rank score tag``
* TREC qrels:      ``qid 0 docid relevance``
* Metrics are computed with pytrec_eval (the trec_eval C code).  A small
  independent implementation (``own_metrics``) is used in the tests to
  cross-check the pipeline.
* Means are taken over **all queries in the qrels** (queries without results
  count as 0, like ``trec_eval -c``).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]

# measure name used by pytrec_eval -> short label used in tables
MEASURES: Dict[str, str] = {
    "map": "MAP",
    "recip_rank": "MRR",
    "P_10": "P@10",
    "ndcg_cut_10": "nDCG@10",
    "ndcg": "nDCG",
    "recall_100": "R@100",
    "recall_1000": "R@1000",
}
PYTREC_MEASURES = {"map", "recip_rank", "P", "ndcg_cut", "ndcg", "recall"}


# ---------------------------------------------------------------- data I/O
def load_queries(split: str, dataset: str = "wikir/en1k") -> List[Tuple[str, str]]:
    import ir_datasets

    return [(q.query_id, q.text) for q in ir_datasets.load(f"{dataset}/{split}").queries_iter()]


def load_qrels(split: str, dataset: str = "wikir/en1k") -> Qrels:
    import ir_datasets

    qrels: Qrels = {}
    for q in ir_datasets.load(f"{dataset}/{split}").qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = int(q.relevance)
    return qrels


def write_qrels(path: str, qrels: Qrels) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels, key=_qid_key):
            for did, rel in sorted(qrels[qid].items()):
                f.write(f"{qid} 0 {did} {rel}\n")


def _qid_key(q: str):
    return (0, int(q)) if q.isdigit() else (1, q)


def write_run(path: str, ranked: Mapping[str, Sequence[Tuple[str, float]]], tag: str) -> None:
    """``ranked[qid]`` must already be sorted best-first."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(ranked, key=_qid_key):
            for rank, (did, score) in enumerate(ranked[qid], start=1):
                f.write(f"{qid} Q0 {did} {rank} {score:.10f} {tag}\n")


def read_run(path: str) -> Run:
    run: Run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, did, _, score, _ = line.split()
            run.setdefault(qid, {})[did] = float(score)
    return run


def to_run(ranked: Mapping[str, Sequence[Tuple[str, float]]]) -> Run:
    return {q: {d: float(s) for d, s in lst} for q, lst in ranked.items()}


# ---------------------------------------------------------------- metrics
def evaluate(qrels: Qrels, run: Run, measures: Optional[Iterable[str]] = None) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Returns (mean over all qrels queries, per-query values)."""
    import pytrec_eval

    names = list(measures or MEASURES)
    ev = pytrec_eval.RelevanceEvaluator(qrels, PYTREC_MEASURES)
    # pytrec_eval skips queries absent from the run -> fill them with zeros
    per_q = ev.evaluate({q: run.get(q, {}) for q in qrels if run.get(q)})
    full: Dict[str, Dict[str, float]] = {}
    for q in qrels:
        full[q] = {m: float(per_q.get(q, {}).get(m, 0.0)) for m in names}
    mean = {m: float(np.mean([full[q][m] for q in qrels])) for m in names}
    return mean, full


def own_metrics(qrels: Qrels, run: Run) -> Dict[str, Dict[str, float]]:
    """Independent re-implementation of MAP, MRR, P@10, nDCG@10, nDCG, R@k.

    Ranking order follows trec_eval: score descending, doc id descending.
    nDCG uses linear gains (gain = relevance) and log2 discounts, as trec_eval.
    """
    out: Dict[str, Dict[str, float]] = {}
    for q, rels in qrels.items():
        docs = run.get(q, {})
        ranked = sorted(docs.items(), key=lambda x: (x[1], x[0]), reverse=True)
        ranked = [d for d, _ in ranked]
        n_rel = sum(1 for r in rels.values() if r > 0)
        hits, ap, rr = 0, 0.0, 0.0
        for i, d in enumerate(ranked, start=1):
            if rels.get(d, 0) > 0:
                hits += 1
                ap += hits / i
                if rr == 0.0:
                    rr = 1.0 / i
        ap = ap / n_rel if n_rel else 0.0

        def dcg(gains):
            return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

        gains = [rels.get(d, 0) for d in ranked]
        ideal = sorted((r for r in rels.values() if r > 0), reverse=True)
        ndcg10 = dcg(gains[:10]) / dcg(ideal[:10]) if ideal else 0.0
        ndcg = dcg(gains) / dcg(ideal) if ideal else 0.0
        out[q] = {
            "map": ap,
            "recip_rank": rr,
            "P_10": sum(1 for d in ranked[:10] if rels.get(d, 0) > 0) / 10.0,
            "ndcg_cut_10": ndcg10,
            "ndcg": ndcg,
            "recall_100": (sum(1 for d in ranked[:100] if rels.get(d, 0) > 0) / n_rel) if n_rel else 0.0,
            "recall_1000": (sum(1 for d in ranked[:1000] if rels.get(d, 0) > 0) / n_rel) if n_rel else 0.0,
        }
    return out


def relevance_level_analysis(qrels: Qrels, run: Run) -> Dict[str, float]:
    """WikIR-specific breakdown.

    rel=2 is the article whose title is the query; rel=1 are articles that
    link to it in their (removed) first sentence.
    """
    rr2, at1, at10, rec1 = [], [], [], []
    for q, rels in qrels.items():
        ranked = [d for d, _ in sorted(run.get(q, {}).items(), key=lambda x: (x[1], x[0]), reverse=True)]
        pos = {d: i for i, d in enumerate(ranked, start=1)}
        r2 = [d for d, r in rels.items() if r == 2]
        r1 = [d for d, r in rels.items() if r == 1]
        if r2:
            best = min((pos.get(d, 10**9) for d in r2))
            rr2.append(1.0 / best if best < 10**9 else 0.0)
            at1.append(1.0 if best <= 1 else 0.0)
            at10.append(1.0 if best <= 10 else 0.0)
        if r1:
            rec1.append(sum(1 for d in r1 if pos.get(d, 10**9) <= 1000) / len(r1))
    return {
        "rel2_MRR": float(np.mean(rr2)) if rr2 else 0.0,
        "rel2_S@1": float(np.mean(at1)) if at1 else 0.0,
        "rel2_S@10": float(np.mean(at10)) if at10 else 0.0,
        "rel1_R@1000": float(np.mean(rec1)) if rec1 else 0.0,
    }


# ------------------------------------------------------- significance test
def paired_randomization_test(a: Sequence[float], b: Sequence[float], n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired randomization (sign-flip permutation) test p-value
    for the difference in means of per-query scores ``a`` vs ``b``."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def save_metrics(path: str, mean: Dict[str, float], per_query: Dict[str, Dict[str, float]], info: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"info": info or {}, "mean": mean, "per_query": per_query}, f, indent=1)


def write_trec_eval_style(path: str, mean: Dict[str, float], per_query: Dict[str, Dict[str, float]], run_name: str) -> None:
    """Text file in the same layout as ``trec_eval -q`` output."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"runid\tall\t{run_name}\n")
        f.write(f"num_q\tall\t{len(per_query)}\n")
        for q in sorted(per_query, key=_qid_key):
            for m, v in per_query[q].items():
                f.write(f"{m}\t{q}\t{v:.4f}\n")
        for m, v in mean.items():
            f.write(f"{m}\tall\t{v:.4f}\n")
