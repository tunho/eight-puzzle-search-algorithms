"""Character n-gram index over the surface vocabulary (spelling correction).

At indexing time every distinct surface word of the corpus is stored with its
corpus frequency and the index term (stem) it maps to.  Each word ``w`` is
padded as ``$w$`` and split into character trigrams, and an inverted index
maps each trigram to the words containing it.

For a query word that is not in the vocabulary (a typo such as ``irsh`` or a
word that never occurs in the corpus) we

1. collect candidate words sharing trigrams (ranked by the Dice coefficient),
2. compute the optimal-string-alignment (Damerau-Levenshtein) distance to the
   best candidates (``chlie`` -> ``chile`` is one transposition),
3. pick the closest candidate; ties are broken by corpus frequency.

Files (in the index directory)::

    surface_words.json      surface words, most frequent first
    surface_cf.npy          int64[W]  corpus frequency of each word
    surface_tid.npy         int32[W]  index term id of each word's stem
    term_display.json       most frequent surface word of every index term (display)
    ngram_list.json         trigram strings (list index == trigram id)
    ngram_offsets.npy       int64[G+1]
    ngram_words.npy         int32[...]  surface word ids per trigram
    word_ngram_count.npy    uint16[W]   number of distinct trigrams of each word
"""
from __future__ import annotations

import json
import os
from array import array
from typing import Dict, List, Optional, Tuple

import numpy as np

N = 3


def trigrams(word: str) -> List[str]:
    s = f"${word}$"
    return sorted({s[i : i + N] for i in range(len(s) - N + 1)})


def save_surface_vocab(index_dir: str, surface_cf: Dict[str, int], analyzer, term2id: Dict[str, int], n_terms: int) -> None:
    words = sorted(surface_cf, key=lambda w: (-surface_cf[w], w))
    cf = np.array([surface_cf[w] for w in words], dtype=np.int64)
    stems = analyzer.normalize(words)
    tid = np.array([term2id.get(t, -1) for t in stems], dtype=np.int32)
    display: List[Optional[str]] = [None] * n_terms
    for w, t in zip(words, tid.tolist()):  # most frequent word first
        if t >= 0 and display[t] is None:
            display[t] = w
    with open(os.path.join(index_dir, "surface_words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f)
    np.save(os.path.join(index_dir, "surface_cf.npy"), cf)
    np.save(os.path.join(index_dir, "surface_tid.npy"), tid)
    with open(os.path.join(index_dir, "term_display.json"), "w", encoding="utf-8") as f:
        json.dump(display, f)


def build_ngram_index(index_dir: str, log=print) -> None:
    with open(os.path.join(index_dir, "surface_words.json"), encoding="utf-8") as f:
        words: List[str] = json.load(f)
    gid: Dict[str, int] = {}
    g_ids = array("i")
    w_ids = array("i")
    counts = np.zeros(len(words), dtype=np.uint16)
    for wid, w in enumerate(words):
        grams = trigrams(w)
        counts[wid] = min(len(grams), 65535)
        for g in grams:
            i = gid.get(g)
            if i is None:
                i = gid[g] = len(gid)
            g_ids.append(i)
            w_ids.append(wid)
    g = np.frombuffer(g_ids, dtype=np.int32)
    w = np.frombuffer(w_ids, dtype=np.int32)
    order = np.argsort(g, kind="stable")
    offsets = np.zeros(len(gid) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g, minlength=len(gid)), out=offsets[1:])
    np.save(os.path.join(index_dir, "ngram_offsets.npy"), offsets)
    np.save(os.path.join(index_dir, "ngram_words.npy"), w[order])
    np.save(os.path.join(index_dir, "word_ngram_count.npy"), counts)
    grams_by_id = [None] * len(gid)
    for k, i in gid.items():
        grams_by_id[i] = k
    with open(os.path.join(index_dir, "ngram_list.json"), "w", encoding="utf-8") as f:
        json.dump(grams_by_id, f)
    log(f"[ngram] {len(words):,} surface words, {len(gid):,} trigrams, {len(w):,} entries")


def osa_distance(a: str, b: str, limit: int = 1 << 30) -> int:
    """Optimal string alignment distance (insert, delete, substitute, adjacent
    transposition = 1).  Returns ``limit + 1`` as soon as it must exceed ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: Optional[List[int]] = None
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class NgramSuggester:
    def __init__(self, index_dir: str):
        d = index_dir
        if not os.path.exists(os.path.join(d, "surface_words.json")):
            raise FileNotFoundError("surface vocabulary missing - rebuild the index with scripts/build_index.py")
        if not os.path.exists(os.path.join(d, "ngram_offsets.npy")):
            build_ngram_index(d, log=lambda *a: None)
        with open(os.path.join(d, "surface_words.json"), encoding="utf-8") as f:
            self.words: List[str] = json.load(f)
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.cf = np.load(os.path.join(d, "surface_cf.npy"))
        self.tid = np.load(os.path.join(d, "surface_tid.npy"))
        self.offsets = np.load(os.path.join(d, "ngram_offsets.npy"))
        self.gram_words = np.load(os.path.join(d, "ngram_words.npy"), mmap_mode="r")
        self.counts = np.load(os.path.join(d, "word_ngram_count.npy")).astype(np.float64)
        with open(os.path.join(d, "ngram_list.json"), encoding="utf-8") as f:
            self.gram2id = {g: i for i, g in enumerate(json.load(f))}

    @staticmethod
    def max_distance(word: str) -> int:
        n = len(word)
        return 1 if n <= 4 else (2 if n <= 10 else 3)

    def frequency(self, word: str) -> int:
        i = self.word2id.get(word)
        return 0 if i is None else int(self.cf[i])

    def _candidates(self, word: str, n_candidates: int) -> np.ndarray:
        grams = [self.gram2id[g] for g in trigrams(word) if g in self.gram2id]
        if not grams:
            return np.zeros(0, np.int64)
        lists = [self.gram_words[self.offsets[g] : self.offsets[g + 1]] for g in grams]
        common = np.bincount(np.concatenate(lists), minlength=len(self.words))
        n_q = len(trigrams(word))
        dice = 2.0 * common / (n_q + self.counts)
        cand = np.flatnonzero(common > 0)
        if cand.size > n_candidates:
            cand = cand[np.argpartition(-dice[cand], n_candidates - 1)[:n_candidates]]
        return cand

    def _edits1(self, word: str) -> List[int]:
        """Vocabulary words at edit distance 1 (delete / transpose / replace / insert).

        Trigram overlap misses transpositions of short words (``chlie`` and
        ``chile`` share a single trigram), so distance-1 neighbours are also
        generated explicitly."""
        letters = "abcdefghijklmnopqrstuvwxyz0123456789"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        cands = set()
        for a, b in splits:
            if b:
                cands.add(a + b[1:])
                for c in letters:
                    cands.add(a + c + b[1:])
            if len(b) > 1:
                cands.add(a + b[1] + b[0] + b[2:])
            for c in letters:
                cands.add(a + c + b)
        return [self.word2id[c] for c in cands if c in self.word2id]

    def best_match(self, word: str, min_cf: int = 2, n_candidates: int = 200) -> Optional[Tuple[str, int, int]]:
        """Closest frequent surface word -> (word, index term id, distance) or None."""
        limit = self.max_distance(word)
        best = None
        cands = set(self._candidates(word, n_candidates).tolist()) | set(self._edits1(word))
        for wid in sorted(cands):
            w = self.words[wid]
            cf = int(self.cf[wid])
            if w == word or cf < min_cf or self.tid[wid] < 0:
                continue
            dist = osa_distance(word, w, limit)
            if dist > limit:
                continue
            key = (dist, -cf, w)
            if best is None or key < best[0]:
                best = (key, w, int(self.tid[wid]), dist)
        return None if best is None else (best[1], best[2], best[3])

    def did_you_mean(self, word: str, ratio: float = 100.0, max_own_cf: int = 3) -> Optional[Tuple[str, int, int]]:
        """Suggestion for a *rare* in-vocabulary word: a close word >= ``ratio`` times more frequent."""
        own = self.frequency(word)
        if own > max_own_cf:
            return None
        m = self.best_match(word, min_cf=max(2, int(ratio * max(own, 1))))
        return m
