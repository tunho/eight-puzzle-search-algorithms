"""Read-only access to an index built by :mod:`irsearch.index_builder`."""
from __future__ import annotations

import json
import mmap
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analyzer import Analyzer


class InvertedIndex:
    """Memory-mapped positional inverted index + document store."""

    def __init__(self, index_dir: str, mmap_mode: Optional[str] = "r"):
        self.index_dir = index_dir
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta: dict = json.load(f)

        def load(name):
            return np.load(os.path.join(index_dir, name), mmap_mode=mmap_mode)

        self.offsets = load("offsets.npy")
        self.post_docs = load("post_docs.npy")
        self.post_tf = load("post_tf.npy")
        self.pos_offsets = load("pos_offsets.npy")
        self.positions = load("positions.npy")
        self.cf = load("cf.npy")
        # small per-document arrays are kept in RAM
        self.doclen = np.load(os.path.join(index_dir, "doclen.npy")).astype(np.float64)
        self.doc_norm_ltc = np.load(os.path.join(index_dir, "doc_norm_ltc.npy"))
        self.doc_offsets = np.load(os.path.join(index_dir, "doc_offsets.npy"))

        with open(os.path.join(index_dir, "vocab.json"), encoding="utf-8") as f:
            terms: List[str] = json.load(f)
        self.terms = terms
        self.term2id: Dict[str, int] = {t: i for i, t in enumerate(terms)}
        with open(os.path.join(index_dir, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids: List[str] = json.load(f)
        self.docid2int: Dict[str, int] = {d: i for i, d in enumerate(self.doc_ids)}

        self._docs_file = open(os.path.join(index_dir, "docs.bin"), "rb")
        size = os.path.getsize(os.path.join(index_dir, "docs.bin"))
        self._docs_mm = mmap.mmap(self._docs_file.fileno(), 0, access=mmap.ACCESS_READ) if size else b""

        self.N: int = int(self.meta["n_docs"])
        self.V: int = int(self.meta["n_terms"])
        self.total_tokens: int = int(self.meta["total_tokens"])
        self.avgdl: float = float(self.meta["avgdl"])
        self.analyzer = Analyzer(stem=bool(self.meta["stem"]))

    # ------------------------------------------------------------------ terms
    def term_id(self, term: str) -> Optional[int]:
        return self.term2id.get(term)

    def df(self, tid: int) -> int:
        return int(self.offsets[tid + 1] - self.offsets[tid])

    def postings(self, tid: int) -> Tuple[np.ndarray, np.ndarray]:
        """(doc ids ascending, term frequencies) of term ``tid``."""
        s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
        return self.post_docs[s:e], self.post_tf[s:e]

    def positions_of(self, tid: int, rel_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Positions for selected postings of ``tid``.

        ``rel_idx`` are posting indices relative to the start of the term's
        posting list.  Returns ``(owner, pos)`` where ``owner[k]`` is the index
        into ``rel_idx`` the position ``pos[k]`` belongs to.
        """
        s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
        tf = self.post_tf[s:e].astype(np.int64)
        start_in_term = np.zeros(len(tf), dtype=np.int64)
        if len(tf) > 1:
            np.cumsum(tf[:-1], out=start_in_term[1:])
        base = int(self.pos_offsets[tid])
        lens = tf[rel_idx]
        starts = base + start_in_term[rel_idx]
        total = int(lens.sum())
        excl = np.zeros(len(lens), dtype=np.int64)
        if len(lens) > 1:
            np.cumsum(lens[:-1], out=excl[1:])
        gather = np.repeat(starts - excl, lens) + np.arange(total, dtype=np.int64)
        owner = np.repeat(np.arange(len(rel_idx), dtype=np.int64), lens)
        return owner, self.positions[gather].astype(np.int64)

    # -------------------------------------------------------------- documents
    def doc_text(self, i: int) -> str:
        s, e = int(self.doc_offsets[i]), int(self.doc_offsets[i + 1])
        return self._docs_mm[s:e].decode("utf-8")

    def surface(self, term: str) -> str:
        """Most frequent surface word for an index term (display only, e.g. 'presid' -> 'president')."""
        if "_display" not in self.__dict__:
            path = os.path.join(self.index_dir, "term_display.json")
            self._display = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    self._display = json.load(f)
        tid = self.term_id(term)
        if self._display is not None and tid is not None and self._display[tid]:
            return self._display[tid]
        # fallback for indexes built without the surface vocabulary
        cache = self.__dict__.setdefault("_surface_cache", {})
        if term in cache:
            return cache[term]
        best = term
        if tid is not None and self.meta.get("stem"):
            from collections import Counter

            docs = self.postings(tid)[0][:5]
            c = Counter()
            for d in docs.tolist():
                for raw, t in self.analyzer.analyze_with_tokens(self.doc_text(d)):
                    if t == term:
                        c[raw] += 1
            if c:
                best = min(c.items(), key=lambda x: (-x[1], len(x[0]), x[0]))[0]
        if len(cache) > 100_000:
            cache.clear()
        cache[term] = best
        return best

    def ext_id(self, i: int) -> str:
        return self.doc_ids[i]

    def int_id(self, doc_id: str) -> Optional[int]:
        return self.docid2int.get(doc_id)

    def close(self) -> None:
        try:
            if isinstance(self._docs_mm, mmap.mmap):
                self._docs_mm.close()
        finally:
            self._docs_file.close()
