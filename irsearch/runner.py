"""Run a whole query set through the searcher."""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from .searcher import Searcher, SearchParams


def run_queries(
    searcher: Searcher,
    queries: Sequence[Tuple[str, str]],
    params: SearchParams,
    k: int = 1000,
) -> Tuple[Dict[str, List[Tuple[str, float]]], float]:
    """Returns ({qid: [(docid, score), ...]}, mean milliseconds per query).

    Only the query *text* is given to the searcher.
    """
    ids = searcher.index.doc_ids
    ranked: Dict[str, List[Tuple[str, float]]] = {}
    t0 = time.perf_counter()
    for qid, text in queries:
        res = searcher.search(text, params, k=k)
        ranked[qid] = [(ids[d], float(s)) for d, s in zip(res.doc_ints.tolist(), res.scores.tolist())]
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(queries))
    return ranked, ms
