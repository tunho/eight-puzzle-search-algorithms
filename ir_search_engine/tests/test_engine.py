"""Correctness tests on a small synthetic corpus.

Every score produced by the engine is compared with a direct (brute force)
computation of the textbook formula.
"""
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irsearch.analyzer import Analyzer  # noqa: E402
from irsearch.evaluation import evaluate, own_metrics  # noqa: E402
from irsearch.index import InvertedIndex  # noqa: E402
from irsearch.index_builder import build_index  # noqa: E402
from irsearch.query_parser import parse_query  # noqa: E402
from irsearch.searcher import Searcher, SearchParams  # noqa: E402

WORDS = (
    "the of and irish sea island ireland wales england scotland president chile university kentucky "
    "school academy new york city state united states river bridge music band film festival venice "
    "running runs ran house representatives pennsylvania"
).split()


def make_corpus(n=400, seed=7):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        L = rng.randint(3, 60)
        docs.append((f"d{i:04d}", " ".join(rng.choice(WORDS) for _ in range(L))))
    docs.append(("dphrase1", "the president of chile visited the irish sea"))
    docs.append(("dphrase2", "chile of president irish the sea"))
    return docs


@pytest.fixture(scope="module", params=[True, False], ids=["stem", "nostem"])
def engine(request, tmp_path_factory):
    out = tmp_path_factory.mktemp("idx")
    docs = make_corpus()
    # workers=2 + small batches exercises the multi-process merge path
    build_index(docs, str(out), stem=request.param, workers=2, batch_size=50, log=lambda *a: None)
    idx = InvertedIndex(str(out))
    yield idx, Searcher(idx), dict(docs)
    idx.close()


def brute_terms(an, text):
    return an.analyze(text)


def test_index_matches_documents(engine):
    idx, _, docs = engine
    an = idx.analyzer
    for i, did in enumerate(idx.doc_ids):
        terms = brute_terms(an, docs[did])
        assert idx.doc_text(i) == docs[did]
        assert idx.doclen[i] == len(terms)
        for t, tf in Counter(terms).items():
            dl, tfl = idx.postings(idx.term_id(t))
            k = np.searchsorted(dl, i)
            assert dl[k] == i and tfl[k] == tf
            _, pos = idx.positions_of(idx.term_id(t), np.array([k]))
            assert list(pos) == [p for p, x in enumerate(terms) if x == t]
    # df sums
    assert sum(idx.df(t) for t in range(idx.V)) == len(idx.post_docs)


def _brute_stats(idx, docs):
    an = idx.analyzer
    tfs = {i: Counter(brute_terms(an, docs[did])) for i, did in enumerate(idx.doc_ids)}
    df = Counter()
    for c in tfs.values():
        df.update(c.keys())
    return tfs, df


@pytest.mark.parametrize("query", ["irish sea", "president of chile", "university of kentucky state", "running bridge bridge"])
@pytest.mark.parametrize("k1,b", [(0.9, 0.4), (1.2, 0.75), (2.0, 0.0)])
def test_bm25_brute_force(engine, query, k1, b):
    idx, s, docs = engine
    tfs, df = _brute_stats(idx, docs)
    N, avgdl = idx.N, idx.avgdl
    q = [t for raw, t in idx.analyzer.analyze_with_tokens(query) if not Analyzer.is_stopword(raw)]
    res = s.search(query, SearchParams(k1=k1, b=b), k=N)
    got = dict(zip(res.doc_ints.tolist(), res.scores.tolist()))
    for i in range(N):
        dl = sum(tfs[i].values())
        exp = 0.0
        for t in q:
            if t in tfs[i]:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                tf = tfs[i][t]
                exp += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        if exp > 0:
            assert got[i] == pytest.approx(exp, rel=1e-9)
        else:
            assert i not in got


@pytest.mark.parametrize("query", ["irish sea", "president chile chile", "venice film festival"])
def test_tfidf_brute_force(engine, query):
    idx, s, docs = engine
    tfs, df = _brute_stats(idx, docs)
    N = idx.N
    q = Counter(t for raw, t in idx.analyzer.analyze_with_tokens(query) if not Analyzer.is_stopword(raw))
    qv = {t: (1 + math.log(c)) * math.log(N / df[t]) for t, c in q.items()}
    qn = math.sqrt(sum(v * v for v in qv.values()))
    res = s.search(query, SearchParams(model="tfidf"), k=N)
    got = dict(zip(res.doc_ints.tolist(), res.scores.tolist()))
    for i in range(N):
        dv = {t: (1 + math.log(c)) * math.log(N / df[t]) for t, c in tfs[i].items()}
        dn = math.sqrt(sum(v * v for v in dv.values()))
        dot = sum(qv[t] * dv.get(t, 0.0) for t in qv)
        if any(t in tfs[i] for t in qv):
            assert got[i] == pytest.approx(dot / (dn * qn), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("mu", [100.0, 1000.0])
def test_ql_brute_force(engine, mu):
    idx, s, docs = engine
    tfs, _ = _brute_stats(idx, docs)
    total = sum(sum(c.values()) for c in tfs.values())
    cf = Counter()
    for c in tfs.values():
        cf.update(c)
    query = "irish sea island"
    q = [t for raw, t in idx.analyzer.analyze_with_tokens(query) if not Analyzer.is_stopword(raw)]
    res = s.search(query, SearchParams(model="ql", mu=mu), k=idx.N)
    got = dict(zip(res.doc_ints.tolist(), res.scores.tolist()))
    # full Dirichlet log-likelihood differs from the engine score by a
    # document-independent constant -> compare score differences
    full = {}
    for i in range(idx.N):
        if not any(t in tfs[i] for t in q):
            continue
        dl = sum(tfs[i].values())
        full[i] = sum(math.log((tfs[i].get(t, 0) + mu * cf[t] / total) / (dl + mu)) for t in q)
    const = {round(got[i] - full[i], 9) for i in full}
    assert len(const) == 1
    assert set(got) == set(full)


def test_phrase_matches_brute_force(engine):
    idx, s, docs = engine
    an = idx.analyzer
    for phrase in ["president of chile", "irish sea", "the irish sea", "new york city"]:
        pt = an.analyze(phrase)
        seq = [(idx.term_id(t), p) for p, t in enumerate(pt)]
        got_docs, got_cnt = s.phrase_matches(seq)
        got = dict(zip(got_docs.tolist(), got_cnt.tolist()))
        for i, did in enumerate(idx.doc_ids):
            terms = brute_terms(an, docs[did])
            cnt = sum(1 for p in range(len(terms) - len(pt) + 1) if terms[p : p + len(pt)] == pt)
            assert got.get(i, 0) == cnt, (phrase, did)


def test_filters(engine):
    idx, s, docs = engine
    res = s.search('"president of chile"', k=idx.N)
    found = {idx.ext_id(i) for i in res.doc_ints.tolist()}
    assert "dphrase1" in found and "dphrase2" not in found
    res = s.search("irish -sea", k=idx.N)
    for i in res.doc_ints.tolist():
        assert "sea" not in docs[idx.ext_id(i)].split()
    res = s.search("+wales sea", k=idx.N)
    for i in res.doc_ints.tolist():
        assert "wales" in docs[idx.ext_id(i)].split()


def test_query_parser():
    an = Analyzer(stem=True)
    pq = parse_query('Irish "President of Chile" +Wales -sea -"isle of man"', an)
    assert [t.raw for t in pq.tokens] == ["irish", "president", "of", "chile", "wales"]
    assert [t.pos for t in pq.tokens] == [0, 1, 2, 3, 4]
    assert [[t.raw for t in p] for p in pq.phrases] == [["president", "of", "chile"]]
    assert [t.raw for t in pq.must] == ["wales"]
    assert [t.raw for t in pq.exclude] == ["sea"]
    assert [[t.raw for t in p] for p in pq.exclude_phrases] == [["isle", "of", "man"]]
    # unbalanced quotes are ignored
    assert [t.raw for t in parse_query('the "who', an).tokens] == ["the", "who"]


def test_stopword_only_query_kept(engine):
    idx, s, _ = engine
    res = s.search("the of", k=10)
    assert res.n_matched > 0


def test_rm3_and_feedback_run(engine):
    idx, s, _ = engine
    res = s.search("irish sea", SearchParams(rm3=True, fb_docs=5, fb_terms=5), k=20)
    assert len(res.doc_ints) > 0
    w = sum(res.query_terms.values())
    assert w == pytest.approx(1.0, abs=1e-3)
    res2 = s.search("irish sea", feedback_docs=res.doc_ints[:2].tolist(), k=20)
    assert len(res2.doc_ints) > 0
    mlt = s.more_like_this(int(res.doc_ints[0]), k=5)
    assert int(res.doc_ints[0]) not in mlt.doc_ints.tolist()


def test_metrics_cross_check():
    rng = random.Random(3)
    qrels, run = {}, {}
    for q in range(30):
        qid = str(q)
        qrels[qid] = {f"d{rng.randint(0, 60)}": rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 12))}
        run[qid] = {f"d{d}": round(rng.random(), 2) for d in rng.sample(range(80), 50)}  # rounding -> ties
    run["31"] = {"d1": 1.0}  # query not in qrels is ignored
    qrels["32"] = {"d5": 1}  # query without results counts as zero
    mean, perq = evaluate(qrels, run)
    own = own_metrics(qrels, run)
    for q in qrels:
        for m, v in perq[q].items():
            assert own[q][m] == pytest.approx(v, abs=1e-9), (q, m)
    assert perq["32"]["map"] == 0.0
    assert mean["map"] == pytest.approx(sum(perq[q]["map"] for q in qrels) / len(qrels))


def test_top_k_tie_break(engine):
    idx, s, _ = engine
    scores = np.zeros(idx.N)
    matched = np.zeros(idx.N, dtype=bool)
    matched[[0, 1, 2, 3]] = True
    scores[[0, 1, 2, 3]] = [1.0, 2.0, 1.0, 1.0]
    docs, _ = s.top_k(scores, matched, 10)
    ids = [idx.ext_id(i) for i in docs.tolist()]
    assert ids[0] == idx.ext_id(1)
    assert ids[1:] == sorted(ids[1:], reverse=True)
