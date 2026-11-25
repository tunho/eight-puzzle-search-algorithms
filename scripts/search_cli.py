"""Terminal interface.

Usage::

    python scripts/search_cli.py                      # interactive
    python scripts/search_cli.py -q "irish sea"       # one query, then exit

Interactive commands::

    <query>              search (supports "phrase", +must, -exclude)
    :model <name>        switch ranking preset (bm25, bm25_prox, bm25_rm3, bm25_prox_rm3, tfidf, ql)
    :next / :prev        next / previous result page
    :doc <doc_id>        show a whole document
    :similar <doc_id>    more-like-this
    :rf <id> [<id> ...]  relevance feedback with the given relevant doc ids
    :help, :quit
"""
import argparse
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irsearch.index import InvertedIndex  # noqa: E402
from irsearch.presets import presets  # noqa: E402
from irsearch.searcher import Searcher  # noqa: E402
from irsearch.snippets import highlight, make_snippet, make_title  # noqa: E402

PER_PAGE = 10
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def term_html(s: str) -> str:
    """Convert snippet HTML (<mark>) to terminal text."""
    on, on2, off = ("\033[1;33m", "\033[1;32m", "\033[0m") if USE_COLOR else ("[", "[", "]")
    s = s.replace('<mark class="exp">', on2).replace("<mark>", on).replace("</mark>", off)
    return html.unescape(re.sub(r"<[^>]+>", "", s))


class CLI:
    def __init__(self, index_dir: str, model: str, preload: bool = True):
        self.idx = InvertedIndex(index_dir, mmap_mode=None if preload else "r")
        self.s = Searcher(self.idx)
        self.s.suggester  # load spelling-correction vocabulary up front
        self.presets = presets()
        self.model = model if model in self.presets else "bm25"
        self.res = None
        self.page = 1
        self.last_q = ""

    def info(self):
        m = self.idx.meta
        print(f"색인: {m['dataset']} | 문서 {self.idx.N:,}개 | 용어 {self.idx.V:,}개 | 포스팅 {m['n_postings']:,}개 | "
              f"모델: {self.presets[self.model][0]}")

    def show(self):
        res = self.res
        if res is None:
            return
        main = {t.term for t in res.parsed.tokens if not t.is_stop} if res.parsed else set(res.query_terms)
        extra = set(res.expansion_terms)
        n = len(res.doc_ints)
        pages = max(1, (n + PER_PAGE - 1) // PER_PAGE)
        if getattr(res, "corrections", None):
            print("오타 교정:", ", ".join(f"{a} -> {b}" for a, b in res.corrections.items()))
        print(f"\n검색된 문서 수: {res.n_matched:,} | 수행 시간: {res.elapsed_ms:.1f} ms | {res.params.describe() if res.params else ''}")
        if res.expansion_terms:
            print("확장 용어:", ", ".join(f"{self.idx.surface(t)}({w:.3g})" for t, w in res.expansion_terms.items()))
        start = (self.page - 1) * PER_PAGE
        for r in range(start, min(n, start + PER_PAGE)):
            d = int(res.doc_ints[r])
            text = self.idx.doc_text(d)
            print(f"\n{r + 1:3d}. [{self.idx.ext_id(d)}] {make_title(text)}   (score {res.scores[r]:.4f})")
            print("     " + term_html(make_snippet(text, self.idx.analyzer, main, extra)))
        print(f"\n-- 페이지 {self.page}/{pages} (:next / :prev) --")

    def search(self, q, feedback=None):
        _, p = self.presets[self.model]
        fb = [self.idx.int_id(x) for x in feedback or [] if self.idx.int_id(x) is not None]
        self.res = self.s.search(q, p, k=1000, feedback_docs=fb or None)
        self.page = 1
        self.last_q = q
        self.show()
        dym = None if self.res.corrections else self.s.did_you_mean(q)
        if dym:
            print(f"이것을 찾으셨나요? {dym}")

    def handle(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        if not line.startswith(":"):
            self.search(line)
            return True
        cmd, *args = line[1:].split()
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "help":
            print(__doc__)
        elif cmd == "model":
            if args and args[0] in self.presets:
                self.model = args[0]
                self.info()
            else:
                print("사용 가능:", ", ".join(self.presets))
        elif cmd in ("next", "prev") and self.res is not None:
            pages = max(1, (len(self.res.doc_ints) + PER_PAGE - 1) // PER_PAGE)
            self.page = min(pages, self.page + 1) if cmd == "next" else max(1, self.page - 1)
            self.show()
        elif cmd == "doc" and args:
            d = self.idx.int_id(args[0])
            if d is None:
                print("없는 문서 ID")
            else:
                terms = {t.term for t in self.res.parsed.tokens} if self.res and self.res.parsed else set()
                print(term_html(highlight(self.idx.doc_text(d), self.idx.analyzer, terms)))
        elif cmd == "similar" and args:
            d = self.idx.int_id(args[0])
            if d is None:
                print("없는 문서 ID")
            else:
                self.res = self.s.more_like_this(d, k=1000, params=self.presets["bm25"][1])
                self.page = 1
                print("사용된 용어:", ", ".join(self.idx.surface(t) for t in self.res.query_terms))
                self.show()
        elif cmd == "rf" and args and self.last_q:
            self.search(self.last_q, feedback=args)
        else:
            print("알 수 없는 명령입니다. :help")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join("index", "wikir_en1k"))
    ap.add_argument("--model", default="bm25")
    ap.add_argument("-q", "--query", default=None)
    ap.add_argument("--mmap", action="store_true", help="memory-map the index instead of loading it into RAM")
    args = ap.parse_args()
    cli = CLI(args.index, args.model, preload=not args.mmap)
    cli.info()
    if args.query:
        cli.search(args.query)
        return
    print("질의를 입력하세요 (:help 도움말, :quit 종료)")
    while True:
        try:
            line = input("\n검색> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cli.handle(line):
            break


if __name__ == "__main__":
    main()
