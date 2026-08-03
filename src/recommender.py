"""
Scoring and ranking — the core of the Retriever in diagrams/architecture.mmd.

`evaluate_song()` is the single source of truth: it computes the score and
collects the evidence for that score in one pass. `score_song()` is the
thin (score, reasons) view of it that the CLI and tests use, and
`recommend_songs()` is the sort-and-cut ranking policy layered on top.

Nothing here generates prose. Explanations live in src/explainer.py, and they
are built from the evidence this module returns — never re-derived.
"""

import csv
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from src.guardrails import (
    REQUIRED_COLUMNS,
    CatalogError,
    RowError,
    summarize_skips,
    validate_song_row,
    validate_user_prefs,
)
from src.logging_setup import get_logger
from src.models import (
    MAX_SCORE,
    RetrievedSong,
    ScoreBreakdown,
    SignalMatch,
    Song,
    UserProfile,
    compute_confidence,
    confidence_band,
)

logger = get_logger("recommender")

# Scoring weights from the Phase 2 Algorithm Recipe (weight shift: energy-first).
# Genre +1.5, mood +0.5 (flat categorical bonuses), energy x2.0 linear closeness.
# Genre outranks mood: a listener asking for rock would rather hear a confident
# rock track than an intense funk one, so a genre hit must never be outbid by a
# mood hit alone. A perfect song caps at MAX_SCORE (4.0). See docs/algorithm_recipe.md.
W_GENRE = 1.5
W_MOOD = 0.5
W_ENERGY = 2.0

# Energy points at or above this count as a "close match" worth mentioning.
ENERGY_REASON_THRESHOLD = 1.6

NO_MATCH_REASON = "No strong matches, but worth a listen"


def evaluate_song(user_prefs: Dict[str, Any], song: Dict[str, Any]) -> ScoreBreakdown:
    """
    Scores one song and records *why*, in a single pass.

    Returns a ScoreBreakdown carrying the total score, a SignalMatch per
    scoring signal (matched or not — misses are evidence too, and the
    explanation agent uses them to state caveats honestly), and the
    human-readable reason strings.
    """
    score = 0.0
    reasons: List[str] = []
    signals: List[SignalMatch] = []

    # Genre - categorical, all-or-nothing.
    genre_matched = song["genre"] == user_prefs["favorite_genre"]
    genre_points = W_GENRE if genre_matched else 0.0
    score += genre_points
    if genre_matched:
        reasons.append(f"Matched your favorite genre ({song['genre']})")
    signals.append(SignalMatch(
        name="genre", field_name="genre", matched=genre_matched,
        points=genre_points, max_points=W_GENRE,
        song_value=song["genre"], target_value=user_prefs["favorite_genre"],
    ))

    # Mood - categorical, all-or-nothing.
    mood_matched = song["mood"] == user_prefs["favorite_mood"]
    mood_points = W_MOOD if mood_matched else 0.0
    score += mood_points
    if mood_matched:
        reasons.append(f"mood also matched ({song['mood']})")
    signals.append(SignalMatch(
        name="mood", field_name="mood", matched=mood_matched,
        points=mood_points, max_points=W_MOOD,
        song_value=song["mood"], target_value=user_prefs["favorite_mood"],
    ))

    # Energy - continuous, linear closeness on the 0-1 scale (partial credit).
    energy_points = W_ENERGY * (1 - abs(song["energy"] - user_prefs["target_energy"]))
    score += energy_points
    energy_matched = energy_points >= ENERGY_REASON_THRESHOLD
    if energy_matched:
        reasons.append("energy was a close match")
    signals.append(SignalMatch(
        name="energy", field_name="energy", matched=energy_matched,
        points=energy_points, max_points=W_ENERGY,
        song_value=song["energy"], target_value=user_prefs["target_energy"],
    ))

    return ScoreBreakdown(score=score, signals=signals, reasons=reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into the knowledge base the retriever searches.

    Guardrails: a missing file, a header without the required columns, or a
    file with no usable rows raises CatalogError. Individual malformed rows
    are skipped and logged rather than aborting the run, so one bad line in a
    catalog of hundreds does not take the system down.
    """
    logger.info("Loading catalog from %s", csv_path)

    songs: List[Dict] = []
    skipped: List[str] = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise CatalogError(f"{csv_path} is empty - no CSV header found")

            missing_columns = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing_columns:
                raise CatalogError(
                    f"{csv_path} is missing required column(s): {', '.join(missing_columns)}"
                )

            # start=2 so the reported line number matches the file (line 1 is the header).
            for line_no, row in enumerate(reader, start=2):
                try:
                    songs.append(validate_song_row(row, line_no))
                except RowError as exc:
                    skipped.append(str(exc))
                    logger.debug("Skipping row: %s", exc)
    except FileNotFoundError as exc:
        raise CatalogError(f"Catalog not found at {csv_path} - run from the repo root") from exc
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{csv_path} is not valid UTF-8 text: {exc}") from exc

    summarize_skips(skipped, csv_path)

    if not songs:
        raise CatalogError(f"{csv_path} contained no valid songs ({len(skipped)} row(s) rejected)")

    logger.info("Loaded %d song(s), skipped %d", len(songs), len(skipped))
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.

    The (score, reasons) view of evaluate_song(). Reasons are collected in the
    same pass as the score, so an explanation can never claim a match the
    score did not award.
    """
    breakdown = evaluate_song(user_prefs, song)
    return (breakdown.score, breakdown.reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """
    Ranking policy: score every song, sort descending, return the top k.

    Returns (song, score, reasons) tuples. Kept separate from scoring so the
    ranking rule can change without touching the math.
    """
    return [
        (record.song, record.score, record.reasons)
        for record in rank_songs(user_prefs, songs, k)
    ]


def build_record(user_prefs: Dict[str, Any], song: Dict[str, Any]) -> RetrievedSong:
    """
    Evaluates one song and wraps it as a RetrievedSong: the record plus the
    evidence and confidence the explanation agent needs. The empty-reasons
    fallback fires here so no recommendation is ever shown without a reason.
    """
    breakdown = evaluate_song(user_prefs, song)
    confidence = compute_confidence(breakdown, MAX_SCORE)
    return RetrievedSong(
        song=song,
        score=breakdown.score,
        reasons=breakdown.reasons if breakdown.reasons else [NO_MATCH_REASON],
        breakdown=breakdown,
        confidence=confidence,
        band=confidence_band(confidence),
    )


def rank_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[RetrievedSong]:
    """
    Scores, ranks, and cuts to top-k, returning full RetrievedSong records
    (evidence + confidence included). `recommend_songs()` is the tuple view
    of this; the Retriever uses the records directly.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    scored = [build_record(user_prefs, song) for song in songs]
    # Sort by score, highest first. Ties keep catalog order (Python sort is stable).
    scored.sort(key=lambda record: record.score, reverse=True)
    return scored[:k]


class Recommender:
    """
    Object-oriented facade over the functional pipeline.

    Kept because diagrams/uml.mmd documents it, but it no longer duplicates
    the logic: it converts dataclasses to the dict form the core works in and
    delegates. One implementation, two entry points.
    """

    def __init__(self, songs: List[Song]):
        """Stores the song catalog this recommender ranks against."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs for the given user profile."""
        prefs = validate_user_prefs(asdict(user))
        by_id = {song.id: song for song in self.songs}
        ranked = rank_songs(prefs, [asdict(song) for song in self.songs], k)
        return [by_id[record.song["id"]] for record in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable, evidence-grounded reason this song fits."""
        from src.explainer import ExplanationAgent  # local import: avoids an import cycle

        prefs = validate_user_prefs(asdict(user))
        record = build_record(prefs, asdict(song))
        return ExplanationAgent().explain(prefs, record)
