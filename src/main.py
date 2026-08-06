"""
Command line runner for the Music Recommender Simulation.

Wires the pipeline from diagrams/architecture.mmd together. Two modes:

    rag      natural-language query -> embed -> top-k chunks -> rerank ->
             inject into a prompt -> generate grounded prose     (default)
    classic  taste profile -> weighted scoring -> template explanation
             (the pre-RAG pipeline, kept for comparison)

Run from the repo root:

    python3 -m src.main                            # all seven demo profiles
    python3 -m src.main --profile lofi -k 3
    python3 -m src.main --query "something for a late night drive"
    python3 -m src.main --mode classic --profile lofi
    python3 -m src.main --log-level INFO           # show the pipeline's logging

Runs without any API key: retrieval falls back to a deterministic offline
embedder and generation to the template explainer. Set GEMINI_API_KEY in .env
for real Gemini embeddings and generation.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# Run as a script (`python3 src/main.py`) as well as a module
# (`python3 -m src.main`): put the repo root on the path when there is no
# package context, so the absolute `src.` imports below resolve either way.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GEMINI_API_KEY is loaded from .env by src.config, which every entry point
# imports — see the note there on why it does not live in main.py any more.
from src.answerer import AnswerAgent
from src.config import (
    DEFAULT_CONTEXT_K,
    DEFAULT_INDEX_PATH,
    DEFAULT_KB_DIR,
    RERANK_CANDIDATES,
    api_key,
)
from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import Embedder, EmbeddingError, EmbeddingUnavailable, make_embedder
from src.explainer import ExplanationAgent
from src.guardrails import CatalogError, ProfileError, QueryError
from src.llm_client import GenerationUnavailable, TemplateGenerator, make_generator
from src.logging_setup import configure_logging, get_logger
from src.models import BAND_LOW
from src.retriever import DEFAULT_CATALOG, Retriever, SemanticRetriever
from src.vector_store import IndexUnavailableError, VectorStore

logger = get_logger("main")

# Demo profiles. Each carries both representations: `query` is what a listener
# would actually type (used by rag mode), `prefs` is the structured profile the
# weighted scorer takes (used by classic mode). A test asserts that
# extract_prefs() recovers `prefs` from `query`, so the two cannot drift apart.
PROFILES: Dict[str, Dict[str, Any]] = {
    "pop": {
        "label": "High-energy pop",
        "query": "upbeat happy pop songs to start the morning",
        "prefs": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8},
    },
    "lofi": {
        "label": "Chill lofi",
        "query": "mellow lofi beats to study to, nothing too energetic",
        "prefs": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3},
    },
    "rock": {
        "label": "Intense rock",
        "query": "loud intense rock for a hard workout",
        "prefs": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.7},
    },
    "metal": {
        "label": "Angry metal",
        "query": "angry metal, as heavy and high-energy as you have",
        "prefs": {"favorite_genre": "metal", "favorite_mood": "angry", "target_energy": 0.9},
    },
    "city pop": {
        "label": "Driveable city pop",
        "query": "nostalgic city pop for a late night drive",
        "prefs": {"favorite_genre": "city pop", "favorite_mood": "nostalgic", "target_energy": 0.8},
    },
    "jazz": {
        "label": "High-energy jazz",
        "query": "energetic jazz with real swing, not background music",
        "prefs": {"favorite_genre": "jazz", "favorite_mood": "energetic", "target_energy": 0.7},
    },
    "funk": {
        "label": "Confident funk",
        "query": "confident funk with a groove I can strut to",
        "prefs": {"favorite_genre": "funk", "favorite_mood": "confident", "target_energy": 0.9},
    },
}

WIDTH = 68


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.main",
        description="Retrieval-augmented music recommender (CLI demo).",
    )
    parser.add_argument(
        "--mode", choices=["rag", "classic"], default="rag",
        help="rag: embeddings + generation (default). classic: the pre-RAG pipeline.",
    )
    parser.add_argument(
        "--profile", choices=[*PROFILES, "all"], default="all",
        help="which demo profile to run (default: all of them)",
    )
    parser.add_argument("--query", default=None,
                        help="ask one natural-language question instead of a profile")
    parser.add_argument("-k", type=int, default=5, help="how many songs to recommend (default: 5)")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help=f"catalog CSV (default: {DEFAULT_CATALOG})")
    parser.add_argument("--kb-dir", default=DEFAULT_KB_DIR, help=f"prose knowledge base (default: {DEFAULT_KB_DIR})")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help=f"vector index (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--context-k", type=int, default=DEFAULT_CONTEXT_K,
                        help=f"prose chunks injected into the prompt (default: {DEFAULT_CONTEXT_K})")
    parser.add_argument("--embedder", choices=["auto", "gemini", "offline"], default="auto",
                        help="embedding backend (default: auto)")
    parser.add_argument("--generator", choices=["auto", "gemini", "template", "off"], default="auto",
                        help="generation backend; 'off' retrieves but does not generate (default: auto)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the retrieved chunk ids and similarities")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level; the full trail always goes to logs/run.log",
    )
    return parser.parse_args(argv)


def build_semantic_stack(args, retriever: Retriever):
    """
    Assembles the retrieval side of rag mode.

    Prefers the committed index, but only when it matches the current corpus
    *and* the embedder this run will use. A mismatch on either is a loud warning
    and a fall back to the offline embedder — never a silent rebuild, because
    triggering paid API calls as a side effect of editing a CSV is a worse
    failure than a warning.

    The embedder is resolved before the index is judged: which embedder we ended
    up with is half of what makes an index usable.
    """
    chunks = build_corpus(args.catalog, args.kb_dir)
    fingerprint = corpus_fingerprint(chunks)

    embedder: Optional[Embedder] = None
    store: Optional[VectorStore] = None
    index_note = ""

    if args.embedder != "offline":
        try:
            store = VectorStore.load(args.index)
        except IndexUnavailableError as exc:
            logger.info("No usable index (%s); using the offline embedder", exc)
            store = None

        if store is not None:
            embedder = make_embedder(args.embedder)
            reason = store.mismatch_reason(fingerprint, embedder)
            if reason is None:
                index_note = f"index: {len(store)} chunks, fresh"
            else:
                print(
                    f"\n  Warning: {args.index} is unusable for this run —\n"
                    f"  {reason}.\n"
                    f"  Rebuild with: python3 -m src.ingest\n"
                    f"  Falling back to the offline embedder for this run.\n",
                    file=sys.stderr,
                )
                logger.warning("Unusable index at %s (%s); falling back offline",
                               args.index, reason)
                store = None
                embedder = None

    if store is None or embedder is None:
        embedder = make_embedder("offline")
        store = VectorStore.build(chunks, embedder, fingerprint)
        index_note = f"built in memory, {len(store)} chunks"

    return SemanticRetriever(retriever, store, embedder), embedder, index_note


def main(argv: List[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(console_level=args.log_level)

    if args.k <= 0:
        print("Error: -k must be a positive integer.", file=sys.stderr)
        return 1

    # --- Retriever: build the knowledge base -------------------------------
    try:
        retriever = Retriever.from_csv(args.catalog)
    except CatalogError as exc:
        logger.error("Catalog load failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(retriever.songs)} songs from {args.catalog}")

    if args.mode == "classic":
        return run_classic(args, retriever)
    return run_rag(args, retriever)


# --- classic mode -----------------------------------------------------------


def run_classic(args, retriever: Retriever) -> int:
    """The pre-RAG pipeline: structured profile in, template explanation out."""
    agent = ExplanationAgent()
    selected = list(PROFILES) if args.profile == "all" else [args.profile]

    run_confidences: List[float] = []
    low_confidence_total = 0

    for name in selected:
        profile = PROFILES[name]
        try:
            results = retriever.retrieve(profile["prefs"], k=args.k)
        except ProfileError as exc:
            logger.error("Profile '%s' rejected: %s", name, exc)
            print(f"Skipping profile '{name}': {exc}", file=sys.stderr)
            continue

        explained = [(record, agent.explain(profile["prefs"], record)) for record in results]
        run_confidences.extend(record.confidence for record in results)
        low_confidence_total += sum(1 for record in results if record.band == BAND_LOW)

        print_recommendations(profile, explained, retriever.average_confidence(results))

    print_run_report(retriever, agent, run_confidences, low_confidence_total, mode="classic")
    return 0


# --- rag mode ---------------------------------------------------------------


def run_rag(args, retriever: Retriever) -> int:
    """Natural-language query in, retrieved-and-grounded explanation out."""
    try:
        semantic, embedder, index_note = build_semantic_stack(args, retriever)
    except (CatalogError, EmbeddingUnavailable, EmbeddingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        generator = make_generator(args.generator)
    except GenerationUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    offline_embedder = embedder.name.startswith("offline")
    offline_generator = generator is None or isinstance(generator, TemplateGenerator)
    # Gated on the key actually being absent. Running offline now has other
    # causes - an unusable index, an explicit --embedder/--generator flag - and
    # blaming a key that is sitting right there sends the reader to the wrong
    # problem. The run report names both backends either way.
    if offline_embedder and offline_generator and not api_key():
        print(
            "\n  Warning: GEMINI_API_KEY not found - running fully offline.\n"
            f"    embeddings : {embedder.name} (deterministic, no network)\n"
            "    generation : template explainer (deterministic, no network)\n"
            "  Set GEMINI_API_KEY in .env to enable Gemini embeddings and generation.\n"
            "  The full retrieval pipeline still runs.",
            file=sys.stderr,
        )

    agent = AnswerAgent(generator)
    catalog_titles = [song["title"] for song in retriever.songs]

    if args.query:
        queries = [("Your query", args.query)]
    else:
        selected = list(PROFILES) if args.profile == "all" else [args.profile]
        queries = [(PROFILES[n]["label"], PROFILES[n]["query"]) for n in selected]

    run_confidences: List[float] = []
    low_confidence_total = 0

    for label, query in queries:
        try:
            records, prose, prefs = semantic.retrieve(
                query, k=args.k, context_k=args.context_k, candidates=RERANK_CANDIDATES,
            )
        except (QueryError, ProfileError, EmbeddingError) as exc:
            logger.error("Query %r rejected: %s", query, exc)
            print(f"Skipping '{label}': {exc}", file=sys.stderr)
            continue

        run_confidences.extend(record.confidence for record in records)
        low_confidence_total += sum(1 for record in records if record.band == BAND_LOW)

        answer = agent.answer(query, prose, records, prefs, catalog_titles=catalog_titles)
        print_rag_result(label, query, prefs, records, prose, answer,
                         semantic.average_confidence(records), args.show_chunks)

    print_run_report(
        semantic, agent, run_confidences, low_confidence_total,
        mode="rag", embedder=embedder.name, index_note=index_note,
        generator=(generator.name if generator else "off"),
        average_similarity=semantic.average_similarity(),
    )
    return 0


# --- output -----------------------------------------------------------------


def print_recommendations(profile: Dict[str, Any], explained, avg_confidence: float) -> None:
    """Renders one profile's recommendations, each with its generated explanation."""
    prefs = profile["prefs"]

    print()
    print("=" * WIDTH)
    print(f"  {profile['label'].upper()}")
    print(
        f"  profile: genre={prefs['favorite_genre']} · mood={prefs['favorite_mood']} · "
        f"energy={prefs['target_energy']}"
    )
    print(f"  average confidence: {avg_confidence:.2f}")
    print("=" * WIDTH)

    if not explained:
        print("\n  No recommendations could be produced for this profile.")
        return

    for rank, (record, explanation) in enumerate(explained, start=1):
        print()
        print(f"  {rank}. {record.song['title']}  —  {record.song['artist']}")
        print(f"     score {record.score:.2f} · confidence {record.confidence:.2f} ({record.band})")
        print(f"     {explanation}")

    print()
    print("=" * WIDTH)


def print_rag_result(label, query, prefs, records, prose, answer,
                     avg_confidence: float, show_chunks: bool) -> None:
    """Renders one natural-language query, what it was understood to mean, and the answer."""
    print()
    print("=" * WIDTH)
    print(f"  {label.upper()}")
    print(f'  query: "{query}"')
    print(
        f"  understood as: genre={prefs['favorite_genre']} · "
        f"mood={prefs['favorite_mood']} · energy={prefs['target_energy']:.2f}"
    )
    print(f"  average confidence: {avg_confidence:.2f}")
    print("=" * WIDTH)

    if not records:
        print("\n  No recommendations could be produced for this query.")
        return

    if show_chunks:
        print("\n  retrieved context:")
        for chunk in prose:
            print(f"    {chunk.similarity:+.3f}  [{chunk.chunk.chunk_id}] {chunk.chunk.title}")

    for rank, record in enumerate(records, start=1):
        print()
        print(f"  {rank}. {record.song['title']}  —  {record.song['artist']}")
        similarity = f" · similarity {record.similarity:.3f}" if record.similarity is not None else ""
        print(f"     score {record.score:.2f} · confidence {record.confidence:.2f} "
              f"({record.band}){similarity}")

    print()
    print(f"  ANSWER ({answer.backend}{' · fell back' if answer.fell_back else ''}):")
    for line in _wrap(answer.text, WIDTH - 4):
        print(f"    {line}")

    print()
    print("=" * WIDTH)


def _wrap(text: str, width: int) -> List[str]:
    """Minimal greedy word wrap, so output stays readable in a terminal."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def print_run_report(retriever, agent, confidences: List[float], low_confidence_total: int,
                     mode: str = "rag", embedder: str = "", index_note: str = "",
                     generator: str = "", average_similarity: float = 0.0) -> None:
    """
    Prints the reliability summary for the run.

    This is the measured part: how confident the system was, how often it fell
    back to a weak match, and whether every printed claim was grounded.
    """
    total = len(confidences)
    average = round(sum(confidences) / total, 3) if total else 0.0

    print()
    print("-" * WIDTH)
    print("  RUN REPORT")
    print("-" * WIDTH)
    print(f"  mode                  : {mode}")

    if mode == "rag":
        print(f"  embedder              : {embedder}" + (f"  ({index_note})" if index_note else ""))
        print(f"  generator             : {generator}")

    print(f"  queries served        : {retriever.stats['queries']}")
    print(f"  recommendations made  : {total}")
    print(f"  average confidence    : {average:.2f}")
    print(f"  low-confidence picks  : {low_confidence_total}/{total}")

    if mode == "rag":
        chunks = retriever.stats.get("chunks_retrieved", 0)
        prose = retriever.stats.get("prose_retrieved", 0)
        print(f"  chunks retrieved      : {chunks} ({chunks - prose} song · {prose} prose)")
        print(f"  avg chunk similarity  : {average_similarity:.3f}")
        print(f"  cited sentences       : {agent.stats['sentences_cited']}/"
              f"{agent.stats['sentences_total']}")
        print(f"  fabricated citations  : {agent.stats['fabricated_citations']}")
        print(f"  substituted songs     : {agent.stats['substituted_titles']}")
        print(f"  answers withheld      : {agent.stats['withheld']}")
        print(f"  fell back to template : {agent.stats['fell_back']}")
    else:
        print(f"  claims made / dropped : {agent.stats['claims_made']} / {agent.stats['claims_dropped']}")
        print(f"  explanations withheld : {agent.stats['withheld']}")

    print(f"  grounding rate        : {agent.grounding_rate():.2f}")
    print(f"  full log              : logs/run.log")
    print("-" * WIDTH)
    print()

    logger.info(
        "Run complete: %d recommendation(s), avg confidence %.3f, grounding rate %.3f",
        total, average, agent.grounding_rate(),
    )


if __name__ == "__main__":
    sys.exit(main())
