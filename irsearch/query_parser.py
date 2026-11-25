"""User query syntax.

Supported syntax (all optional; plain keyword queries work as usual)::

    irish sea               free-text keywords (ranked)
    "president of chile"    exact phrase: results must contain the phrase
    +ireland                term that must appear
    -scotland               term that must NOT appear
    -"isle of man"          phrase that must NOT appear

Each token keeps its position in the query so that adjacent query words can
be used for positional (proximity) scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .analyzer import Analyzer

_PART_RE = re.compile(r'([+-]?)"([^"]*)"|([+-]?)(\S+)')


@dataclass
class QueryToken:
    raw: str  # lower-cased surface token
    term: str  # index term (stem)
    pos: int  # position in the query token stream
    is_stop: bool


@dataclass
class ParsedQuery:
    text: str
    tokens: List[QueryToken] = field(default_factory=list)  # tokens used for ranking
    phrases: List[List[QueryToken]] = field(default_factory=list)  # required phrases
    must: List[QueryToken] = field(default_factory=list)  # required terms
    exclude: List[QueryToken] = field(default_factory=list)  # forbidden terms
    exclude_phrases: List[List[QueryToken]] = field(default_factory=list)

    @property
    def has_filters(self) -> bool:
        return bool(self.phrases or self.must or self.exclude or self.exclude_phrases)


def parse_query(text: str, analyzer: Analyzer) -> ParsedQuery:
    pq = ParsedQuery(text=text)
    if text.count('"') % 2 == 1:  # unbalanced quote -> ignore quotes
        text = text.replace('"', " ")
    pos = 0
    for m in _PART_RE.finditer(text):
        if m.group(2) is not None:  # quoted phrase
            sign, body, is_phrase = m.group(1), m.group(2), True
        else:
            sign, body, is_phrase = m.group(3), m.group(4), False
        pairs = analyzer.analyze_with_tokens(body)
        if not pairs:
            continue
        toks = []
        for raw, term in pairs:
            toks.append(QueryToken(raw, term, pos, analyzer.is_stopword(raw)))
            pos += 1
        if sign == "-":
            if is_phrase and len(toks) > 1:
                pq.exclude_phrases.append(toks)
            else:
                pq.exclude.extend(toks)
            continue
        pq.tokens.extend(toks)
        if is_phrase and len(toks) > 1:
            pq.phrases.append(toks)
        elif sign == "+" or (is_phrase and len(toks) == 1):
            pq.must.extend(toks)
    return pq
