"""Named ranking configurations shared by the web UI, the CLI and the experiments.

Parameters come from ``results/tuning/best_params.json`` (tuned on the
training split by ``scripts/tune.py``); defaults are used if it is missing.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from .searcher import SearchParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_PARAMS_PATH = os.path.join(ROOT, "results", "tuning", "best_params.json")


def load_best(path: str = BEST_PARAMS_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def presets(best: dict = None) -> Dict[str, Tuple[str, SearchParams]]:
    best = load_best() if best is None else best
    bm = best.get("bm25", {})
    k1, b = bm.get("k1", 0.9), bm.get("b", 0.4)
    mu = best.get("ql", {}).get("mu", 1000.0)
    pw = best.get("prox", {}).get("prox_weight", 0.1)
    r = best.get("rm3", {})
    fb = dict(
        fb_docs=r.get("fb_docs", 10), fb_terms=r.get("fb_terms", 10), fb_orig_weight=r.get("fb_orig_weight", 0.5),
        fb_max_df=r.get("fb_max_df", 1.0),
    )
    return {
        "bm25": ("BM25", SearchParams(k1=k1, b=b)),
        "bm25_prox": ("BM25 + 근접도(positional)", SearchParams(k1=k1, b=b, prox_weight=pw)),
        "bm25_rm3": ("BM25 + RM3 질의 확장", SearchParams(k1=k1, b=b, rm3=True, **fb)),
        "bm25_prox_rm3": ("BM25 + 근접도 + RM3", SearchParams(k1=k1, b=b, prox_weight=pw, rm3=True, **fb)),
        "tfidf": ("TF-IDF (cosine)", SearchParams(model="tfidf")),
        "ql": ("Query Likelihood (Dirichlet)", SearchParams(model="ql", mu=mu)),
    }
