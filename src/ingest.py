"""
Build-time ingest — the only thing in this project that writes the index.

    python3 -m src.ingest                 # embed everything, write the index
    python3 -m src.ingest --dry-run       # chunk and report, ZERO API calls
    python3 -m src.ingest --check         # exit 1 if the index is stale
    python3 -m src.ingest --embedder offline --out data/index/offline_index.jsonl

Separating ingest from query is what makes the running cost of this system one
embedding call and one generation call per question, rather than 277 embeddings
every time someone asks for a song. The index is committed to the repository so
a fresh clone can run real Gemini retrieval without spending anything.

`--dry-run` exists so chunk sizes can be tuned for free: it does everything
except call the API.
"""

import argparse
import os
import sys
from typing import List

from src.config import DEFAULT_CATALOG, DEFAULT_INDEX_PATH, DEFAULT_KB_DIR
from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import EmbeddingError, EmbeddingUnavailable, make_embedder
from src.guardrails import CatalogError
from src.logging_setup import configure_logging, get_logger
from src.models import KIND_PROSE, KIND_SONG, Chunk
from src.vector_store import IndexUnavailableError, VectorStore

logger = get_logger("ingest")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.ingest",
        description="Chunk and embed the corpus into a vector index.",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG,
                        help=f"catalog CSV (default: {DEFAULT_CATALOG})")
    parser.add_argument("--kb-dir", default=DEFAULT_KB_DIR,
                        help=f"prose knowledge base directory (default: {DEFAULT_KB_DIR})")
    parser.add_argument("--out", default=DEFAULT_INDEX_PATH,
                        help=f"index path to write (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--embedder", default="auto", choices=["auto", "gemini", "offline"],
                        help="which embedding backend to use (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and report statistics without calling the API")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the index does not match the current corpus")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def report_chunks(chunks: List[Chunk]) -> None:
    """Prints the size distribution a `--dry-run` is for."""
    songs = [c for c in chunks if c.kind == KIND_SONG]
    prose = [c for c in chunks if c.kind == KIND_PROSE]
    lengths = sorted(len(c.text) for c in prose)

    print(f"\n  song cards   : {len(songs)}")
    print(f"  prose chunks : {len(prose)}")
    print(f"  total        : {len(chunks)}")

    if lengths:
        print(f"\n  prose chunk chars: min {lengths[0]} · "
              f"median {lengths[len(lengths) // 2]} · max {lengths[-1]}")
        sources = sorted({c.source for c in prose})
        for source in sources:
            count = sum(1 for c in prose if c.source == source)
            print(f"    {source:34} {count:3} chunk(s)")
    print()


def main(argv: List[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(console_level=args.log_level)

    try:
        chunks = build_corpus(args.catalog, args.kb_dir)
    except CatalogError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    fingerprint = corpus_fingerprint(chunks)

    # --check: compare the committed index against the live corpus.
    if args.check:
        try:
            store = VectorStore.load(args.out)
        except IndexUnavailableError as exc:
            print(f"STALE: {exc}", file=sys.stderr)
            return 1

        # Constructing an embedder costs no API calls, so --check can also
        # verify that the index was built by the backend this environment
        # would use — the mismatch a fingerprint comparison cannot see.
        try:
            expected = make_embedder(args.embedder)
        except EmbeddingUnavailable:
            expected = None

        reason = store.mismatch_reason(fingerprint, expected)
        if reason is None:
            print(f"OK: {args.out} matches the corpus "
                  f"({len(store)} vectors, {store.embedder_name})")
            return 0

        print(
            f"STALE: {args.out} is unusable — {reason}.\n"
            f"  index : {len(store)} vectors, {store.dimensions} dims, "
            f"{store.embedder_name}\n"
            f"  corpus: {len(chunks)} chunks, fingerprint {fingerprint[:16]}...\n"
            f"  Rebuild with: python3 -m src.ingest",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(f"\nDry run — no API calls. Corpus from {args.catalog} and {args.kb_dir}:")
        report_chunks(chunks)
        print(f"  fingerprint  : {fingerprint[:32]}...")
        print(f"  would write  : {args.out}\n")
        return 0

    try:
        embedder = make_embedder(args.embedder)
    except EmbeddingUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nEmbedding {len(chunks)} chunk(s) with {embedder.name}...")
    try:
        store = VectorStore.build(chunks, embedder, fingerprint)
    except EmbeddingError as exc:
        print(f"Error: embedding failed - {exc}", file=sys.stderr)
        return 1

    store.save(args.out)

    counts = store.counts()
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(
        f"\n  wrote {args.out}\n"
        f"  {len(store)} vectors ({counts.get(KIND_SONG, 0)} song · "
        f"{counts.get(KIND_PROSE, 0)} prose) · {store.dimensions} dims · "
        f"{size_mb:.1f} MB\n"
        f"  embedder    : {embedder.name}\n"
        f"  fingerprint : {fingerprint[:32]}...\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
