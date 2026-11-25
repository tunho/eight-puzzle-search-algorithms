"""Query processing and ranking (all scoring implemented here with numpy).

Ranking models
--------------
* ``bm25``  - Okapi BM25, Lucene-style idf  ln(1 + (N - df + 0.5) / (df + 0.5))
* ``tfidf`` - vector-space model, ltc.ltc cosine similarity
              w = (1 + ln tf) * ln(N / df), documents and query L2-normalised
* ``ql``    - query likelihood with Dirichlet smoothing (rank-equivalent form)

Extensions
----------
* positional proximity (BM25 only): every pair of consecutive query words
  ``(a, b)`` with query gap ``g`` becomes a pseudo-term "b occurs exactly
  ``g`` positions after a"; it is scored with the BM25 formula using its own
  document frequency and added with weight ``prox_weight``.
* RM3 pseudo relevance feedback (query expansion).
* interactive relevance feedback (user-selected relevant documents).
* "more like this" - find documents similar to a given document.
* filters: exact phrases, required terms, excluded terms/phrases.

Evaluation never uses query ids: a query is processed from its text only.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .index import InvertedIndex
from .query_parser import ParsedQuery, QueryToken, parse_query

MODELS = ("bm25", "tfidf", "ql")


@dataclass
class SearchParams:
    model: str = "bm25"
    k1: float = 0.9
    b: float = 0.4
    mu: float = 1000.0
    remove_stopwords: bool = True
    prox_weight: float = 0.0  # 0 disables proximity scoring (BM25 only)
    rm3: bool = False
    fb_docs: int = 10
    fb_terms: int = 10
    fb_orig_weight: float = 0.5
    fb_max_df: float = 1.0  # feedback terms with df/N above this are ignored (1.0 = no filter)
    spell: bool = True  # replace out-of-vocabulary query terms via the character n-gram index

    def describe(self) -> str:
        if self.model == "bm25":
            s = f"BM25(k1={self.k1:g}, b={self.b:g})"
            if self.prox_weight > 0:
                s += f" + proximity(w={self.prox_weight:g})"
        elif self.model == "tfidf":
            s = "TF-IDF cosine (ltc.ltc)"
        else:
            s = f"QL-Dirichlet(mu={self.mu:g})"
        if self.rm3:
            s += f" + RM3(docs={self.fb_docs}, terms={self.fb_terms}, orig={self.fb_orig_weight:g}"
            s += f", maxdf={self.fb_max_df:g})" if self.fb_max_df < 1.0 else ")"
        return s


@dataclass
class SearchResult:
    doc_ints: np.ndarray  # ranked internal doc ids
    scores: np.ndarray  # matching scores
    n_matched: int  # number of documents that matched the (filtered) query
    elapsed_ms: float
    parsed: Optional[ParsedQuery] = None
    query_terms: Dict[str, float] = field(default_factory=dict)  # term -> weight actually used
    expansion_terms: Dict[str, float] = field(default_factory=dict)  # RM3 / RF added terms
    params: Optional[SearchParams] = None
    corrections: Dict[str, str] = field(default_factory=dict)  # OOV query word -> index term used


class Searcher:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.analyzer = index.analyzer
        self.N = index.N
        self.avgdl = index.avgdl
        self._norm_cache: Dict[Tuple[float, float], np.ndarray] = {}
        self._max_doclen = int(index.meta.get("max_doclen", 0)) or int(index.doclen.max())
        self._suggester = None
        self._suggest_cache: Dict[Tuple[str, str], Optional[Tuple[str, int, int]]] = {}

    # ------------------------------------------------- spelling correction
    @property
    def suggester(self):
        if self._suggester is None:
            from .ngram import NgramSuggester

            self._suggester = NgramSuggester(self.index.index_dir)
        return self._suggester

    def _cached(self, kind: str, word: str):
        key = (kind, word)
        if key not in self._suggest_cache:
            fn = self.suggester.best_match if kind == "fix" else self.suggester.did_you_mean
            if len(self._suggest_cache) > 20_000:
                self._suggest_cache.clear()
            self._suggest_cache[key] = fn(word)
        return self._suggest_cache[key]

    def _apply_corrections(self, pq: ParsedQuery) -> Dict[str, str]:
        """Replace out-of-vocabulary query words by the closest corpus word."""
        seen, fixed = set(), {}
        groups = [pq.tokens, pq.exclude] + list(pq.exclude_phrases)
        for grp in groups:
            for tok in grp:
                if id(tok) in seen:
                    continue
                seen.add(id(tok))
                if tok.is_stop or self.index.term_id(tok.term) is not None:
                    continue
                m = self._cached("fix", tok.raw)
                if m is not None:
                    fixed[tok.raw] = m[0]
                    tok.term = self.index.terms[m[1]]
        return fixed

    def did_you_mean(self, query: str) -> Optional[str]:
        """'Did you mean' query for rare (but indexed) words; the query itself is not changed."""
        pq = parse_query(query, self.analyzer)
        repl = {}
        for tok in pq.tokens + pq.exclude:
            if tok.is_stop or tok.raw in repl:
                continue
            m = self._cached("dym", tok.raw)
            if m is not None:
                repl[tok.raw] = m[0]
        if not repl:
            return None
        import re

        return re.sub(r"[A-Za-z0-9]+", lambda mo: repl.get(mo.group().lower(), mo.group()), query)

    # ----------------------------------------------------------- statistics
    def bm25_idf(self, df) -> np.ndarray:
        df = np.asarray(df, dtype=np.float64)
        return np.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def _bm25_norm(self, k1: float, b: float) -> np.ndarray:
        key = (float(k1), float(b))
        arr = self._norm_cache.get(key)
        if arr is None:
            arr = k1 * (1.0 - b + b * self.index.doclen / self.avgdl)
            if len(self._norm_cache) > 64:
                self._norm_cache.clear()
            self._norm_cache[key] = arr
        return arr

    # ------------------------------------------------------- query weights
    def query_weights(self, tokens: Sequence[QueryToken], remove_stopwords: bool = True) -> Dict[int, float]:
        """Raw query-term counts (term id -> qtf).  OOV terms are dropped.

        Stopwords are removed unless the query consists only of stopwords
        (e.g. "the who"), in which case they are kept.
        """
        use = [t for t in tokens if not (remove_stopwords and t.is_stop)]
        if not use:
            use = list(tokens)
        w: Dict[int, float] = defaultdict(float)
        for t in use:
            tid = self.index.term_id(t.term)
            if tid is not None:
                w[tid] += 1.0
        return dict(w)

    def _proximity_tokens(self, tokens: Sequence[QueryToken], weights: Dict[int, float]) -> List[Tuple[int, int]]:
        """(term id, query position) of the tokens that take part in ranking."""
        out = []
        for t in tokens:
            tid = self.index.term_id(t.term)
            if tid is not None and tid in weights:
                out.append((tid, t.pos))
        return out

    # ------------------------------------------------------ scoring models
    def score_bm25(self, weights: Dict[int, float], k1: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.index
        scores = np.zeros(self.N, dtype=np.float64)
        matched = np.zeros(self.N, dtype=bool)
        norm = self._bm25_norm(k1, b)
        for tid, wq in weights.items():
            docs, tf = idx.postings(tid)
            if len(docs) == 0:
                continue
            tf = tf.astype(np.float64)
            idf = float(self.bm25_idf(len(docs)))
            scores[docs] += wq * idf * tf * (k1 + 1.0) / (tf + norm[docs])
            matched[docs] = True
        return scores, matched

    def score_tfidf(self, weights: Dict[int, float], log_qtf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.index
        scores = np.zeros(self.N, dtype=np.float64)
        matched = np.zeros(self.N, dtype=bool)
        qvec = {}
        for tid, wq in weights.items():
            df = idx.df(tid)
            idf = math.log(self.N / df)
            qvec[tid] = ((1.0 + math.log(wq)) if log_qtf else wq) * idf
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        for tid, wq in qvec.items():
            docs, tf = idx.postings(tid)
            df = len(docs)
            idf = math.log(self.N / df)
            scores[docs] += wq * (1.0 + np.log(tf.astype(np.float64))) * idf
            matched[docs] = True
        dn = idx.doc_norm_ltc
        nz = matched & (dn > 0)
        scores[nz] /= dn[nz] * qnorm
        return scores, matched

    def score_ql(self, weights: Dict[int, float], mu: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.index
        scores = np.zeros(self.N, dtype=np.float64)
        matched = np.zeros(self.N, dtype=bool)
        total = float(idx.total_tokens)
        for tid, wq in weights.items():
            docs, tf = idx.postings(tid)
            p_c = float(idx.cf[tid]) / total
            scores[docs] += wq * np.log1p(tf.astype(np.float64) / (mu * p_c))
            matched[docs] = True
        qlen = sum(weights.values())
        dl = idx.doclen[matched]
        scores[matched] += qlen * np.log(mu / (dl + mu))
        return scores, matched

    # --------------------------------------------------- positional search
    def phrase_matches(
        self, toks: Sequence[Tuple[int, int]], restrict: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Documents containing the token sequence at the given relative positions.

        ``toks`` is a list of ``(term_id, relative_position)``.  Returns
        ``(doc ids, number of occurrences)``.
        """
        idx = self.index
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        if not toks:
            return empty
        r0 = min(r for _, r in toks)
        toks = [(t, r - r0) for t, r in toks]
        R = max(r for _, r in toks)
        order = sorted(toks, key=lambda x: idx.df(x[0]))
        common = np.asarray(idx.postings(order[0][0])[0])
        for tid, _ in order[1:]:
            common = np.intersect1d(common, idx.postings(tid)[0], assume_unique=True)
            if common.size == 0:
                return empty
        if restrict is not None:
            common = np.intersect1d(common, restrict, assume_unique=True)
        if common.size == 0:
            return empty
        M = self._max_doclen + R + 1
        keys = None
        for tid, r in toks:
            docs = idx.postings(tid)[0]
            rel = np.searchsorted(docs, common)
            owner, pos = idx.positions_of(tid, rel)
            k = owner * M + (pos - r + R)
            keys = k if keys is None else np.intersect1d(keys, k, assume_unique=True)
            if keys.size == 0:
                return empty
        doc_idx, counts = np.unique(keys // M, return_counts=True)
        return common[doc_idx].astype(np.int64), counts.astype(np.int64)

    def score_proximity(self, prox_toks: List[Tuple[int, int]], k1: float, b: float) -> np.ndarray:
        """BM25-scored ordered "word pair at query gap" pseudo-terms."""
        scores = np.zeros(self.N, dtype=np.float64)
        toks = sorted(prox_toks, key=lambda x: x[1])
        if len(toks) < 2:
            return scores
        norm = self._bm25_norm(k1, b)
        for (ta, pa), (tb, pb) in zip(toks, toks[1:]):
            docs, cnt = self.phrase_matches([(ta, pa), (tb, pb)])
            if docs.size == 0:
                continue
            tf = cnt.astype(np.float64)
            idf = float(self.bm25_idf(docs.size))
            scores[docs] += idf * tf * (k1 + 1.0) / (tf + norm[docs])
        return scores

    # ------------------------------------------------------------- filters
    def _filter_mask(self, pq: ParsedQuery) -> Optional[np.ndarray]:
        if not pq.has_filters:
            return None
        idx = self.index
        mask = np.ones(self.N, dtype=bool)

        def docs_of_term(tok: QueryToken) -> np.ndarray:
            tid = idx.term_id(tok.term)
            return np.zeros(0, np.int64) if tid is None else np.asarray(idx.postings(tid)[0])

        def docs_of_phrase(toks: List[QueryToken]) -> np.ndarray:
            seq = []
            for t in toks:
                tid = idx.term_id(t.term)
                if tid is None:
                    return np.zeros(0, np.int64)
                seq.append((tid, t.pos))
            return self.phrase_matches(seq)[0]

        for tok in pq.must:
            m = np.zeros(self.N, dtype=bool)
            m[docs_of_term(tok)] = True
            mask &= m
        for ph in pq.phrases:
            m = np.zeros(self.N, dtype=bool)
            m[docs_of_phrase(ph)] = True
            mask &= m
        for tok in pq.exclude:
            mask[docs_of_term(tok)] = False
        for ph in pq.exclude_phrases:
            mask[docs_of_phrase(ph)] = False
        return mask

    # --------------------------------------------------------------- top-k
    def top_k(self, scores: np.ndarray, matched: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k matched docs; ties broken like trec_eval (doc id string, descending)."""
        cand = np.flatnonzero(matched)
        if cand.size == 0 or k <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float64)
        if cand.size > k:
            part = np.argpartition(-scores[cand], k - 1)[:k]
            cand = cand[part]
        ids = self.index.doc_ids
        cand = sorted(cand.tolist(), key=lambda i: ids[i], reverse=True)
        cand.sort(key=lambda i: -scores[i])  # stable
        cand = np.asarray(cand, dtype=np.int64)
        return cand, scores[cand]

    # ------------------------------------------------------- feedback / RM3
    def _doc_term_distribution(self, doc_int: int) -> Counter:
        pairs = self.analyzer.analyze_with_tokens(self.index.doc_text(doc_int))
        return Counter(
            t for raw, t in pairs if not self.analyzer.is_stopword(raw) and len(raw) > 1 and not raw.isdigit()
        )

    def feedback_model(
        self, doc_ints: Sequence[int], doc_weights: Sequence[float], n_terms: int, max_df: float = 1.0
    ) -> Dict[int, float]:
        """Relevance model P(w|R) = sum_D P(D) * tf(w,D)/|D|, top ``n_terms`` renormalised.

        Terms whose document frequency exceeds ``max_df * N`` are skipped as
        candidates (too generic to help, e.g. "also", "year").
        """
        acc: Dict[str, float] = defaultdict(float)
        wsum = float(sum(doc_weights))
        if wsum <= 0:
            return {}
        for d, wd in zip(doc_ints, doc_weights):
            c = self._doc_term_distribution(int(d))
            L = sum(c.values())
            if L == 0:
                continue
            for t, tf in c.items():
                acc[t] += (wd / wsum) * tf / L
        limit = max_df * self.N
        cand = []
        for t, v in acc.items():
            tid = self.index.term_id(t)
            if tid is not None and (max_df >= 1.0 or self.index.df(tid) <= limit):
                cand.append((t, tid, v))
        top = sorted(cand, key=lambda x: (-x[2], x[0]))[:n_terms]
        z = sum(v for _, _, v in top)
        return {tid: v / z for _, tid, v in top} if z > 0 else {}

    @staticmethod
    def interpolate(orig: Dict[int, float], fb: Dict[int, float], orig_weight: float) -> Dict[int, float]:
        qz = sum(orig.values()) or 1.0
        new: Dict[int, float] = defaultdict(float)
        for t, w in orig.items():
            new[t] += orig_weight * w / qz
        for t, w in fb.items():
            new[t] += (1.0 - orig_weight) * w
        return dict(new)

    # -------------------------------------------------------------- search
    def _score(self, weights: Dict[int, float], p: SearchParams, log_qtf: bool) -> Tuple[np.ndarray, np.ndarray]:
        if p.model == "bm25":
            return self.score_bm25(weights, p.k1, p.b)
        if p.model == "tfidf":
            return self.score_tfidf(weights, log_qtf=log_qtf)
        if p.model == "ql":
            return self.score_ql(weights, p.mu)
        raise ValueError(f"unknown model {p.model!r}; choose from {MODELS}")

    def _feedback_doc_weights(self, scores: np.ndarray, model: str) -> np.ndarray:
        if model == "ql":  # log-likelihood scale -> probabilities
            e = np.exp(scores - scores.max())
            return e / e.sum()
        s = np.clip(scores, 0.0, None)
        return s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s))

    def search(
        self,
        query: str,
        params: Optional[SearchParams] = None,
        k: int = 1000,
        feedback_docs: Optional[Sequence[int]] = None,
        exclude_docs: Optional[Iterable[int]] = None,
    ) -> SearchResult:
        """Rank documents for ``query``.

        ``feedback_docs``: internal ids the user marked as relevant
        (interactive relevance feedback); they expand the query like RM3
        with uniform document weights.
        """
        p = params or SearchParams()
        t0 = time.perf_counter()
        pq = parse_query(query, self.analyzer)
        corrections = self._apply_corrections(pq) if p.spell else {}
        weights = self.query_weights(pq.tokens, p.remove_stopwords)
        res = SearchResult(np.zeros(0, np.int64), np.zeros(0), 0, 0.0, pq, {}, {}, p, corrections)
        if not weights:
            res.elapsed_ms = (time.perf_counter() - t0) * 1000
            return res
        prox_toks = self._proximity_tokens(pq.tokens, weights) if p.prox_weight > 0 and p.model == "bm25" else []
        mask = self._filter_mask(pq)
        excl = np.fromiter(exclude_docs, dtype=np.int64) if exclude_docs is not None else None

        def run(w: Dict[int, float], log_qtf: bool):
            s, m = self._score(w, p, log_qtf)
            if prox_toks:
                s = s + p.prox_weight * self.score_proximity(prox_toks, p.k1, p.b)
            if mask is not None:
                m &= mask
            if excl is not None and excl.size:
                m[excl] = False
            return s, m

        scores, matched = run(weights, True)
        final_w = weights
        fb: Dict[int, float] = {}
        if feedback_docs:
            fb = self.feedback_model(feedback_docs, [1.0] * len(feedback_docs), max(p.fb_terms, 20), p.fb_max_df)
        elif p.rm3:
            top, top_s = self.top_k(scores, matched, p.fb_docs)
            if top.size:
                fb = self.feedback_model(top, self._feedback_doc_weights(top_s, p.model), p.fb_terms, p.fb_max_df)
        if fb:
            final_w = self.interpolate(weights, fb, p.fb_orig_weight)
            scores, matched = run(final_w, False)

        docs, sc = self.top_k(scores, matched, k)
        terms = self.index.terms
        res.doc_ints, res.scores = docs, sc
        res.n_matched = int(matched.sum())
        res.query_terms = {terms[t]: round(w, 4) for t, w in sorted(final_w.items(), key=lambda x: -x[1])}
        res.expansion_terms = {terms[t]: round(w, 4) for t, w in sorted(fb.items(), key=lambda x: -x[1]) if t not in weights}
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res

    def more_like_this(self, doc_int: int, k: int = 10, n_terms: int = 25, params: Optional[SearchParams] = None) -> SearchResult:
        """Documents similar to ``doc_int``: its top tf*idf terms as a weighted BM25 query."""
        p = params or SearchParams()
        t0 = time.perf_counter()
        c = self._doc_term_distribution(doc_int)
        cand = []
        for t, tf in c.items():
            tid = self.index.term_id(t)
            if tid is not None:
                cand.append((tf * float(self.bm25_idf(self.index.df(tid))), tid))
        cand.sort(reverse=True)
        w = {tid: s for s, tid in cand[:n_terms]}
        z = sum(w.values()) or 1.0
        w = {t: v / z for t, v in w.items()}
        scores, matched = self.score_bm25(w, p.k1, p.b)
        matched[doc_int] = False
        docs, sc = self.top_k(scores, matched, k)
        terms = self.index.terms
        return SearchResult(
            docs, sc, int(matched.sum()), (time.perf_counter() - t0) * 1000, None,
            {terms[t]: round(v, 4) for t, v in sorted(w.items(), key=lambda x: -x[1])}, {}, p,
        )
