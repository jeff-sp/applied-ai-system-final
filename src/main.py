"""
Command line runner for the Music Recommender Simulation.

Wires the pipeline from diagrams/architecture.mmd together:

    profile -> Retriever (csv -> score -> rank) -> ExplanationAgent -> output

Run from the repo root:

    python3 -m src.main                      # all seven demo profiles
    python3 -m src.main --profile lofi -k 3
    python3 -m src.main --log-level INFO     # show the pipeline's own logging
"""

import argparse
import os
import sys
from typing import Any, Dict, List

# Run as a script (`python3 src/main.py`) as well as a module
# (`python3 -m src.main`): put the repo root on the path when there is no
# package context, so the absolute `src.` imports below resolve either way.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainer import ExplanationAgent
from src.guardrails import CatalogError, ProfileError
from src.logging_setup import configure_logging, get_logger
from src.models import BAND_LOW
from src.retriever import DEFAULT_CATALOG, Retriever

logger = get_logger("main")

# Demo profiles. A profile needs exactly these three keys.
PROFILES: Dict[str, Dict[str, Any]] = {
    "pop": {
        "label": "High-energy pop",
        "prefs": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8},
    },
    "lofi": {
        "label": "Chill lofi",
        "prefs": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3},
    },
    "rock": {
        "label": "Intense rock",
        "prefs": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.7},
    },
    "metal": {
        "label": "Angry metal",
        "prefs": {"favorite_genre": "metal", "favorite_mood": "angry", "target_energy": 0.9},
    },
    "city pop": {
        "label": "Driveable city pop",
        "prefs": {"favorite_genre": "city pop", "favorite_mood": "nostalgic", "target_energy": 0.8},
    },
    "jazz": {
        "label": "High-energy jazz",
        "prefs": {"favorite_genre": "jazz", "favorite_mood": "energetic", "target_energy": 0.7}
    },
    "funk": {
        "label": "Confident funk",
        "prefs": {"favorite_genre": "funk", "favorite_mood": "confident", "target_energy": 0.9}
    }
}

WIDTH = 68


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.main",
        description="Retrieval-augmented music recommender (CLI demo).",
    )
    parser.add_argument(
        "--profile", choices=[*PROFILES, "all"], default="all",
        help="which demo profile to run (default: all of them)",
    )
    parser.add_argument("-k", type=int, default=5, help="how many songs to recommend (default: 5)")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help=f"catalog CSV (default: {DEFAULT_CATALOG})")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level; the full trail always goes to logs/run.log",
    )
    return parser.parse_args(argv)


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

    agent = ExplanationAgent()
    selected = list(PROFILES) if args.profile == "all" else [args.profile]

    print(f"\nLoaded {len(retriever.songs)} songs from {args.catalog}")

    run_confidences: List[float] = []
    low_confidence_total = 0

    for name in selected:
        profile = PROFILES[name]

        # --- Retrieval, then augmented generation --------------------------
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

    print_run_report(retriever, agent, run_confidences, low_confidence_total)
    return 0


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


def print_run_report(retriever: Retriever, agent: ExplanationAgent,
                     confidences: List[float], low_confidence_total: int) -> None:
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
    print(f"  queries served        : {retriever.stats['queries']}")
    print(f"  recommendations made  : {total}")
    print(f"  average confidence    : {average:.2f}")
    print(f"  low-confidence picks  : {low_confidence_total}/{total}")
    print(f"  claims made / dropped : {agent.stats['claims_made']} / {agent.stats['claims_dropped']}")
    print(f"  grounding rate        : {agent.grounding_rate():.2f}")
    print(f"  explanations withheld : {agent.stats['withheld']}")
    print(f"  full log              : logs/run.log")
    print("-" * WIDTH)
    print()

    logger.info(
        "Run complete: %d recommendation(s), avg confidence %.3f, grounding rate %.3f",
        total, average, agent.grounding_rate(),
    )


if __name__ == "__main__":
    sys.exit(main())
