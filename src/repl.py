"""
Interactive session runner — ask, look at the answer, ask again.

`src/main.py` answers one question per process. That is the right shape for a
scripted demo, but it means every follow-up question pays the startup cost
again: building the corpus, loading 277 vectors, and — when the committed index
cannot be used — embedding every chunk from scratch.

This module builds that stack exactly once and then loops. Per turn the cost is
what the design docs claim the system costs: one embedding call and one
generation call.

    python3 -m src.repl
    python3 -m src.repl --embedder offline --generator template -k 3

Two commands beyond asking questions:

    why <n>     the full scoring evidence for recommendation n
    quit        leave (Ctrl-D and Ctrl-C also work)

`why` prints numbers that already exist. Every recommendation is scored through
`evaluate_song()`, which records a SignalMatch per signal — hits and misses
both — and the normal output shows almost none of it. Nothing here re-derives a
value; if it is on screen, the scorer put it on the record.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO

# Run as a script as well as a module, matching src/main.py.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.answerer import AnswerAgent
from src.config import (
    DEFAULT_CONTEXT_K,
    DEFAULT_INDEX_PATH,
    DEFAULT_KB_DIR,
    RERANK_CANDIDATES,
)
from src.embeddings import EmbeddingError, EmbeddingUnavailable
from src.guardrails import CatalogError, ProfileError, QueryError
from src.llm_client import GenerationUnavailable, make_generator
from src.logging_setup import configure_logging, get_logger
from src.main import WIDTH, build_semantic_stack, print_rag_result, print_why
from src.retriever import DEFAULT_CATALOG, Retriever
from src.vector_store import DimensionMismatchError

logger = get_logger("repl")

PROMPT = "\nask> "
QUIT_WORDS = {"quit", "exit", ":q"}
HELP_WORDS = {"help", "?", ":h"}

BANNER = """
  Ask for music in your own words. Two commands:

    why <n>    show the full scoring evidence for recommendation n
    help       show this again
    quit       leave (Ctrl-D and Ctrl-C also work)
"""


def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    The subset of src/main.py's flags that mean anything in a session.

    Deliberately no --mode, --profile, or --query: this runs the rag pipeline
    on whatever you type, and the demo profiles belong to the one-shot CLI.
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m src.repl",
        description="Interactive music recommender session.",
    )
    parser.add_argument("-k", type=int, default=5, help="how many songs per answer (default: 5)")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help=f"catalog CSV (default: {DEFAULT_CATALOG})")
    parser.add_argument("--kb-dir", default=DEFAULT_KB_DIR, help=f"prose knowledge base (default: {DEFAULT_KB_DIR})")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help=f"vector index (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--context-k", type=int, default=DEFAULT_CONTEXT_K,
                        help=f"prose chunks injected into the prompt (default: {DEFAULT_CONTEXT_K})")
    parser.add_argument("--embedder", choices=["auto", "gemini", "offline"], default="auto",
                        help="embedding backend (default: auto)")
    parser.add_argument("--generator", choices=["auto", "gemini", "template", "off"], default="auto",
                        help="generation backend (default: auto)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the retrieved chunk ids and similarities")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="console log level; the full trail always goes to logs/run.log")
    return parser.parse_args(argv)


class Session:
    """
    One interactive session over a warm retrieval stack.

    Everything expensive lives here as an attribute and is built once in
    `start()`. The loop below only ever calls `retrieve` and `answer`.

    `last_records` is the only piece of conversational state: it is what `why`
    indexes into. It holds the same RetrievedSong objects the ranker produced,
    so an explanation of a pick cannot drift from the pick itself.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.retriever: Optional[Retriever] = None
        self.semantic = None
        self.embedder = None
        self.generator = None
        self.agent: Optional[AnswerAgent] = None
        self.catalog_titles: List[str] = []
        self.last_records: List[Any] = []
        self.last_query: str = ""

    def start(self) -> None:
        """Builds the stack. Raises the same errors src/main.py handles at startup."""
        self.retriever = Retriever.from_csv(self.args.catalog)
        self.semantic, self.embedder, index_note = build_semantic_stack(self.args, self.retriever)
        self.generator = make_generator(self.args.generator)
        self.agent = AnswerAgent(self.generator)
        self.catalog_titles = [song["title"] for song in self.retriever.songs]

        print(f"\nLoaded {len(self.retriever.songs)} songs from {self.args.catalog}")
        print(f"  embedder: {self.embedder.name}" + (f" ({index_note})" if index_note else ""))
        print(f"  generator: {self.generator.name if self.generator else 'off'}")

    def ask(self, query: str) -> None:
        """
        Answers one question and remembers the result for `why`.

        The body is what src/main.py's run_rag does per query, minus the setup
        it repeats on every invocation.
        """
        records, prose, prefs = self.semantic.retrieve(
            query,
            k=self.args.k,
            context_k=self.args.context_k,
            candidates=RERANK_CANDIDATES,
        )
        answer = self.agent.answer(query, prose, records, prefs, catalog_titles=self.catalog_titles)

        print_rag_result(
            "Your query", query, prefs, records, prose, answer,
            self.semantic.average_confidence(records), self.args.show_chunks,
        )

        self.last_records = list(records)
        self.last_query = query

    def why(self, argument: str) -> None:
        """Prints the scoring evidence for one of the last answer's picks."""
        if not self.last_records:
            print("  Nothing to explain yet — ask for some music first.", file=sys.stderr)
            return

        highest = len(self.last_records)
        try:
            rank = int(argument)
        except ValueError:
            print(f"  'why' takes a rank number, 1 to {highest}. Try: why 1", file=sys.stderr)
            return

        if not 1 <= rank <= highest:
            print(
                f"  There is no #{rank} — the last answer had {highest} "
                f"recommendation{'s' if highest != 1 else ''}.",
                file=sys.stderr,
            )
            return

        print_why(self.last_records[rank - 1], rank, self.embedder)


def main(argv: List[str] = None, stdin: TextIO = None) -> int:
    """
    Runs an interactive session until the user leaves or input runs out.

    `stdin` is a parameter so the loop can be driven by a scripted input source
    in tests, the same way main(argv) makes the one-shot CLI testable.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(console_level=args.log_level)
    source = sys.stdin if stdin is None else stdin

    if args.k <= 0:
        print("Error: -k must be a positive integer.", file=sys.stderr)
        return 1

    session = Session(args)
    try:
        session.start()
    except (CatalogError, EmbeddingUnavailable, EmbeddingError, GenerationUnavailable) as exc:
        logger.error("Session startup failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(BANNER)

    while True:
        line = _read_line(source)
        if line is None:
            break

        line = line.strip()
        if not line:
            continue
        if line.lower() in QUIT_WORDS:
            break
        if line.lower() in HELP_WORDS:
            print(BANNER)
            continue

        command, _, argument = line.partition(" ")
        if command.lower() == "why":
            session.why(argument.strip())
            continue

        # Anything else is a question. A bad one ends this turn, not the session:
        # a session that dies on a typo is worse than one that says what was wrong.
        try:
            session.ask(line)
        except (QueryError, ProfileError, EmbeddingError, DimensionMismatchError) as exc:
            logger.error("Query %r rejected: %s", line, exc)
            print(f"  Skipped: {exc}", file=sys.stderr)

    print("\n" + "-" * WIDTH)
    print("  Session over. Full log: logs/run.log")
    print("-" * WIDTH + "\n")
    return 0


def _read_line(source: TextIO) -> Optional[str]:
    """
    Reads one line, returning None when the session should end.

    Ctrl-D (EOF) and Ctrl-C both end the session rather than raising: leaving is
    not an error. The prompt is only drawn for a real terminal, so scripted
    input and captured test output stay clean.
    """
    try:
        if source is sys.stdin and source.isatty():
            return input(PROMPT)
        line = source.readline()
        return None if line == "" else line
    except (EOFError, KeyboardInterrupt):
        return None


if __name__ == "__main__":
    sys.exit(main())
