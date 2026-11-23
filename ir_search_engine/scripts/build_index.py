"""Build the inverted index for wikir/en1k.

Usage (from the ``ir_search_engine`` directory)::

    python scripts/build_index.py                     # stemmed index -> index/wikir_en1k
    python scripts/build_index.py --no-stem --out index/wikir_en1k_nostem
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irsearch.index_builder import build_index, iter_ir_dataset_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wikir/en1k")
    ap.add_argument("--out", default=None, help="output directory (default: index/wikir_en1k[_nostem])")
    ap.add_argument("--no-stem", action="store_true", help="disable stemming")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: #cpus, max 8)")
    ap.add_argument("--batch-size", type=int, default=4000)
    args = ap.parse_args()

    stem = not args.no_stem
    out = args.out or os.path.join("index", "wikir_en1k" + ("" if stem else "_nostem"))
    build_index(
        iter_ir_dataset_docs(args.dataset),
        out,
        stem=stem,
        workers=args.workers,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
