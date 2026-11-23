"""Build a positional inverted index (written from scratch, numpy-based).

On-disk layout (directory ``out_dir``)::

    meta.json            collection statistics + build settings
    vocab.json           list of terms; list index == term id
    offsets.npy          int64[V+1]  postings of term t are [offsets[t], offsets[t+1])
    post_docs.npy        int32[P]    internal doc ids, ascending within a term
    post_tf.npy          uint16[P]   term frequency of each posting
    pos_offsets.npy      int64[V+1]  positions of term t are [pos_offsets[t], pos_offsets[t+1])
    positions.npy        uint16[T]   token positions, grouped by posting (same order as post_docs)
    cf.npy               int64[V]    collection frequency of each term
    doclen.npy           int32[N]    number of tokens per document
    doc_norm_ltc.npy     float64[N]  L2 norm of the ltc (log-tf * idf) document vector
    doc_ids.json         list of external doc ids; list index == internal doc id
    docs.bin             UTF-8 document texts, concatenated (document store for snippets)
    doc_offsets.npy      int64[N+1]  byte offsets of each document in docs.bin

The construction is a classic "sort-based" inversion:

1. Documents are analyzed in parallel worker processes.  Each worker turns
   a batch of documents into (local_term_id, doc, tf, positions) records
   with a batch-local vocabulary.
2. The main process maps the batch vocabularies to global term ids and
   concatenates all records (they are in doc order).
3. A stable sort by term id yields, for every term, its postings in
   ascending doc order.  Positions are re-ordered the same way.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from array import array
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .analyzer import Analyzer

INDEX_FORMAT_VERSION = 1

_WORKER_ANALYZER: Optional[Analyzer] = None


def _init_worker(stem: bool) -> None:
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = Analyzer(stem=stem)


def _process_batch(args: Tuple[int, Sequence[str]]):
    """Invert one batch of documents (runs inside a worker process)."""
    start, texts = args
    an = _WORKER_ANALYZER
    assert an is not None
    vocab: dict = {}
    t_ids = array("i")
    d_off = array("i")
    tfs = array("H")
    pos = array("H")
    dls = array("i")
    for i, text in enumerate(texts):
        terms = an.analyze(text)
        n = len(terms)
        if n > 65535:  # positions are stored as uint16
            terms = terms[:65535]
            n = 65535
        dls.append(n)
        occ: dict = {}
        for p, t in enumerate(terms):
            lst = occ.get(t)
            if lst is None:
                occ[t] = [p]
            else:
                lst.append(p)
        for t, plist in occ.items():
            tid = vocab.get(t)
            if tid is None:
                tid = len(vocab)
                vocab[t] = tid
            t_ids.append(tid)
            d_off.append(i)
            tfs.append(len(plist))
            pos.extend(plist)
    return (
        start,
        list(vocab),
        np.frombuffer(t_ids, dtype=np.int32).copy(),
        np.frombuffer(d_off, dtype=np.int32).copy(),
        np.frombuffer(tfs, dtype=np.uint16).copy(),
        np.frombuffer(pos, dtype=np.uint16).copy(),
        np.frombuffer(dls, dtype=np.int32).copy(),
    )


def _batches(texts: Sequence[str], batch_size: int) -> Iterator[Tuple[int, Sequence[str]]]:
    for s in range(0, len(texts), batch_size):
        yield s, texts[s : s + batch_size]


def iter_ir_dataset_docs(dataset: str) -> Iterator[Tuple[str, str]]:
    import ir_datasets

    for d in ir_datasets.load(dataset).docs_iter():
        yield d.doc_id, d.text


def build_index(
    docs: Iterable[Tuple[str, str]],
    out_dir: str,
    stem: bool = True,
    workers: Optional[int] = None,
    batch_size: int = 4000,
    dataset_name: str = "custom",
    log=print,
) -> dict:
    """Build the index for ``docs`` (iterable of ``(doc_id, text)``) into ``out_dir``."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. read corpus, write the document store -------------------------
    doc_ids: List[str] = []
    texts: List[str] = []
    doc_offsets = array("q", [0])
    with open(os.path.join(out_dir, "docs.bin"), "wb") as fout:
        pos = 0
        for did, text in docs:
            doc_ids.append(did)
            texts.append(text)
            b = text.encode("utf-8")
            fout.write(b)
            pos += len(b)
            doc_offsets.append(pos)
    n_docs = len(doc_ids)
    if len(set(doc_ids)) != n_docs:
        raise ValueError("duplicate doc ids in corpus")
    log(f"[index] read {n_docs:,} documents ({time.time() - t0:.1f}s)")

    # ---- 2. parallel analysis + per-batch inversion -------------------------
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 1)))
    vocab: dict = {}
    parts_t, parts_d, parts_tf, parts_pos = [], [], [], []
    doclen = np.zeros(n_docs, dtype=np.int32)

    def consume(result):
        start, terms, t, d, tf, ps, dl = result
        gids = np.fromiter(
            (vocab.setdefault(x, len(vocab)) for x in terms), dtype=np.int32, count=len(terms)
        )
        parts_t.append(gids[t])
        parts_d.append(d + start)
        parts_tf.append(tf)
        parts_pos.append(ps)
        doclen[start : start + len(dl)] = dl

    done = 0
    if workers > 1 and n_docs > batch_size:
        ctx = mp.get_context("spawn")  # portable (Windows/macOS/Linux)
        with ctx.Pool(workers, initializer=_init_worker, initargs=(stem,)) as pool:
            for res in pool.imap(_process_batch, _batches(texts, batch_size), chunksize=1):
                consume(res)
                done += len(res[6])
                if done % (batch_size * 10) < batch_size or done == n_docs:
                    log(f"[index] analyzed {done:,}/{n_docs:,} docs ({time.time() - t0:.1f}s)")
    else:
        _init_worker(stem)
        for b in _batches(texts, batch_size):
            res = _process_batch(b)
            consume(res)
            done += len(res[6])
        log(f"[index] analyzed {done:,}/{n_docs:,} docs ({time.time() - t0:.1f}s)")
    del texts

    term_ids = np.concatenate(parts_t) if parts_t else np.zeros(0, np.int32)
    post_docs_unsorted = np.concatenate(parts_d) if parts_d else np.zeros(0, np.int32)
    tfs = np.concatenate(parts_tf) if parts_tf else np.zeros(0, np.uint16)
    positions = np.concatenate(parts_pos) if parts_pos else np.zeros(0, np.uint16)
    del parts_t, parts_d, parts_tf, parts_pos
    n_terms = len(vocab)
    n_post = len(term_ids)
    log(f"[index] vocabulary {n_terms:,} terms, {n_post:,} postings, {len(positions):,} positions")

    # ---- 3. sort postings by term (stable => doc order kept within a term) -
    order = np.argsort(term_ids, kind="stable")
    df = np.bincount(term_ids, minlength=n_terms).astype(np.int64)
    offsets = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(df, out=offsets[1:])
    post_docs = post_docs_unsorted[order]
    post_tf = tfs[order]
    del post_docs_unsorted

    # positions: gather in the new posting order, chunked to bound memory
    orig_start = np.zeros(n_post, dtype=np.int64)
    if n_post > 1:
        np.cumsum(tfs[:-1], dtype=np.int64, out=orig_start[1:])
    del tfs
    new_positions = np.empty_like(positions)
    chunk = 4_000_000
    out_ptr = 0
    for cs in range(0, n_post, chunk):
        sel = order[cs : cs + chunk]
        starts = orig_start[sel]
        lens = post_tf[cs : cs + chunk].astype(np.int64)
        total = int(lens.sum())
        if total == 0:
            continue
        excl = np.zeros(len(lens), dtype=np.int64)
        np.cumsum(lens[:-1], out=excl[1:])
        gather = np.repeat(starts - excl, lens) + np.arange(total, dtype=np.int64)
        new_positions[out_ptr : out_ptr + total] = positions[gather]
        out_ptr += total
    assert out_ptr == len(positions)
    del positions, orig_start, order

    # per-term position offsets + collection frequency
    cf = np.add.reduceat(post_tf, offsets[:-1], dtype=np.int64) if n_post else np.zeros(n_terms, np.int64)
    pos_offsets = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(cf, out=pos_offsets[1:])

    # ltc document norms: sqrt(sum_t ((1+ln tf) * ln(N/df))^2)
    idf = np.log(n_docs / df.astype(np.float64))
    w = (1.0 + np.log(post_tf.astype(np.float64))) * np.repeat(idf, df)
    doc_norm = np.sqrt(np.bincount(post_docs, weights=w * w, minlength=n_docs))
    del w

    # ---- 4. write ----------------------------------------------------------
    np.save(os.path.join(out_dir, "offsets.npy"), offsets)
    np.save(os.path.join(out_dir, "post_docs.npy"), post_docs)
    np.save(os.path.join(out_dir, "post_tf.npy"), post_tf)
    np.save(os.path.join(out_dir, "pos_offsets.npy"), pos_offsets)
    np.save(os.path.join(out_dir, "positions.npy"), new_positions)
    np.save(os.path.join(out_dir, "cf.npy"), cf)
    np.save(os.path.join(out_dir, "doclen.npy"), doclen)
    np.save(os.path.join(out_dir, "doc_norm_ltc.npy"), doc_norm)
    np.save(os.path.join(out_dir, "doc_offsets.npy"), np.frombuffer(doc_offsets, dtype=np.int64))
    terms_by_id = [None] * n_terms
    for t, i in vocab.items():
        terms_by_id[i] = t
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(terms_by_id, f)
    with open(os.path.join(out_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    total_tokens = int(doclen.sum())
    meta = {
        "format_version": INDEX_FORMAT_VERSION,
        "dataset": dataset_name,
        "stem": stem,
        "stemmer": "snowball-english" if stem else None,
        "n_docs": n_docs,
        "n_terms": n_terms,
        "n_postings": int(n_post),
        "total_tokens": total_tokens,
        "avgdl": total_tokens / max(1, n_docs),
        "max_doclen": int(doclen.max()) if n_docs else 0,
        "build_seconds": round(time.time() - t0, 1),
        "workers": workers,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"[index] done in {meta['build_seconds']}s -> {out_dir}")
    return meta
