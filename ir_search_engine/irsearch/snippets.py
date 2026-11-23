"""Snippet extraction and query-term highlighting for result display."""
from __future__ import annotations

import html
import re
from typing import Iterable, List, Set, Tuple

from .analyzer import Analyzer

_TOKEN_SPAN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str, analyzer: Analyzer) -> List[Tuple[int, int, str]]:
    spans = [(m.start(), m.end(), m.group().lower()) for m in _TOKEN_SPAN_RE.finditer(text)]
    stems = analyzer.normalize([t for _, _, t in spans])
    return [(s, e, st) for (s, e, _), st in zip(spans, stems)]


def _render(text: str, toks, lo: int, hi: int, main: Set[str], extra: Set[str]) -> str:
    """HTML for tokens [lo, hi) with <mark> around matching terms."""
    if lo >= hi:
        return ""
    out = []
    cur = toks[lo][0]
    for s, e, st in toks[lo:hi]:
        out.append(html.escape(text[cur:s]))
        word = html.escape(text[s:e])
        if st in main:
            out.append(f"<mark>{word}</mark>")
        elif st in extra:
            out.append(f'<mark class="exp">{word}</mark>')
        else:
            out.append(word)
        cur = e
    return "".join(out)


def make_snippet(
    text: str,
    analyzer: Analyzer,
    query_terms: Iterable[str],
    expansion_terms: Iterable[str] = (),
    window: int = 40,
) -> str:
    """Best ``window``-token passage (most distinct query terms, then most hits), as HTML."""
    main, extra = set(query_terms), set(expansion_terms) - set(query_terms)
    toks = _tokens(text, analyzer)
    n = len(toks)
    if n == 0:
        return ""
    if n <= window:
        return _render(text, toks, 0, n, main, extra)
    hit = [st in main for _, _, st in toks]
    best, best_key = 0, (-1, -1)
    for start in range(0, n - window + 1):
        seg = toks[start : start + window]
        distinct = len({st for (_, _, st), h in zip(seg, hit[start : start + window]) if h})
        key = (distinct, sum(hit[start : start + window]))
        if key > best_key:
            best, best_key = start, key
    # start the window a few tokens before the first hit for context
    first = next((i for i in range(best, best + window) if hit[i]), best)
    start = max(0, min(first - 5, n - window))
    body = _render(text, toks, start, start + window, main, extra)
    return ("… " if start > 0 else "") + body + (" …" if start + window < n else "")


def highlight(text: str, analyzer: Analyzer, query_terms: Iterable[str], expansion_terms: Iterable[str] = ()) -> str:
    main, extra = set(query_terms), set(expansion_terms) - set(query_terms)
    toks = _tokens(text, analyzer)
    return _render(text, toks, 0, len(toks), main, extra) if toks else html.escape(text)


def make_title(text: str, n_words: int = 9) -> str:
    """wikir documents have no title field: use the first words of the text."""
    words = text.split()
    t = " ".join(words[:n_words])
    return t + (" …" if len(words) > n_words else "")
