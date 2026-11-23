"""Text analysis: tokenization, stopword detection and stemming.

The same analyzer is used for documents (at indexing time) and queries
(at search time) so that the produced terms always match.

Design notes
------------
* wikir/en1k documents are already lower-cased and contain only the
  characters ``[a-z0-9 ]``.  We still lower-case and split on any
  non-alphanumeric character so that free-form user queries
  (punctuation, upper case, quotes ...) are handled the same way.
* **All** tokens, stopwords included, are indexed together with their
  positions.  Stopword removal is a *query-time* decision.  This keeps
  exact phrase search (``"president of chile"``) possible and lets us
  run the stopword ablation without re-indexing.
* Stemming uses the Snowball (Porter2) English stemmer from PyStemmer.
  Stemming is a text-normalisation step, not a search algorithm.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")

# English stopword list (NLTK list; forms with apostrophes are dropped because
# the corpus has no apostrophes - "don't" appears as "don t").
STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are was
    were be been being have has had having do does did doing a an the and but
    if or because as until while of at by for with about against between into
    through during before after above below to from up down in out on off over
    under again further then once here there when where why how all any both
    each few more most other some such no nor not only own same so than too
    very s t can will just don should now d ll m o re ve y ain aren couldn
    didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
    weren won wouldn
    """.split()
)


class Analyzer:
    """Tokenizer + optional stemmer.

    Parameters
    ----------
    stem:
        Apply the Snowball English stemmer when True.
    """

    def __init__(self, stem: bool = True):
        self.stem = stem
        self._stemmer = None
        if stem:
            import Stemmer  # PyStemmer

            self._stemmer = Stemmer.Stemmer("english")
            self._stemmer.maxCacheSize = 500_000

    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lower-case and split into alphanumeric tokens."""
        return TOKEN_RE.findall(text.lower())

    def normalize(self, tokens: List[str]) -> List[str]:
        """Map raw tokens to index terms (stems)."""
        if self._stemmer is None:
            return list(tokens)
        return self._stemmer.stemWords(tokens)

    def analyze(self, text: str) -> List[str]:
        """Text -> list of index terms; list position == token position."""
        return self.normalize(self.tokenize(text))

    def analyze_with_tokens(self, text: str) -> List[Tuple[str, str]]:
        """Text -> list of (raw_token, term) pairs (position = list index)."""
        toks = self.tokenize(text)
        return list(zip(toks, self.normalize(toks)))

    def term(self, token: str) -> Optional[str]:
        """Normalize a single raw token."""
        toks = self.tokenize(token)
        if not toks:
            return None
        return self.normalize(toks[:1])[0]

    @staticmethod
    def is_stopword(raw_token: str) -> bool:
        return raw_token in STOPWORDS
