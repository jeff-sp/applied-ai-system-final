"""
The Retriever — the "R" in retrieval-augmented generation.

It owns the knowledge base (data/songs.csv), indexes it, and answers a taste
profile with the top-k candidates *plus the evidence for each one*. The
Explanation Agent downstream is given only what this returns, so anything the
system says about a song traces back to a retrieved record.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from src.guardrails import validate_user_prefs
from src.logging_setup import get_logger
from src.models import BAND_LOW, RetrievedSong
from src.recommender import load_songs, rank_songs

logger = get_logger("retriever")

DEFAULT_CATALOG = "data/songs.csv"


class Retriever:
    """
    Holds the song catalog and retrieves scored, evidence-carrying candidates.

    Also keeps per-instance counters so a run can report retrieval health
    (queries served, average confidence, how often it fell back to weak
    energy-only matches).
    """

    def __init__(self, songs: List[Dict[str, Any]], source: str = "<memory>"):
        if not songs:
            raise ValueError("Retriever needs a non-empty catalog")

        self.songs = songs
        self.source = source
        # Lightweight indexes over the knowledge base. Small catalogs do not
        # need them for speed; they exist so the retriever can tell the user
        # *before* ranking that a profile has no genre coverage at all.
        self.genre_index: Counter = Counter(song["genre"] for song in songs)
        self.mood_index: Counter = Counter(song["mood"] for song in songs)
        self.stats: Dict[str, Any] = {"queries": 0, "results": 0, "low_confidence": 0}

        logger.info(
            "Retriever ready: %d songs from %s across %d genre(s)",
            len(songs), source, len(self.genre_index),
        )

    @classmethod
    def from_csv(cls, csv_path: str = DEFAULT_CATALOG) -> "Retriever":
        """Builds a Retriever from a CSV catalog. Raises CatalogError if unusable."""
        return cls(load_songs(csv_path), source=csv_path)

    def retrieve(self, user_prefs: Dict[str, Any], k: int = 5) -> List[RetrievedSong]:
        """
        Answers a taste profile with the top-k candidates and their evidence.

        The profile is validated first (guardrail), then every song is scored
        and ranked. Coverage gaps and low-confidence results are logged, not
        hidden — a thin catalog is a reliability problem the user should hear
        about, not a silent one.
        """
        prefs = validate_user_prefs(user_prefs)
        logger.info(
            "Query: genre=%s mood=%s target_energy=%.2f k=%d",
            prefs["favorite_genre"], prefs["favorite_mood"], prefs["target_energy"], k,
        )

        genre_hits = self.genre_index.get(prefs["favorite_genre"], 0)
        if genre_hits == 0:
            logger.warning(
                "No song in %s has genre '%s' - results will lean on energy alone",
                self.source, prefs["favorite_genre"],
            )
        elif genre_hits < k:
            logger.info(
                "Only %d song(s) match genre '%s'; ranks past %d fall back to weaker signals",
                genre_hits, prefs["favorite_genre"], genre_hits,
            )

        results = rank_songs(prefs, self.songs, k)

        self.stats["queries"] += 1
        self.stats["results"] += len(results)
        self.stats["low_confidence"] += sum(1 for r in results if r.band == BAND_LOW)

        for rank, record in enumerate(results, start=1):
            logger.debug(
                "  #%d %s (%s) score=%.2f confidence=%.2f band=%s",
                rank, record.song["title"], record.song["genre"],
                record.score, record.confidence, record.band,
            )

        if results and all(r.band == BAND_LOW for r in results):
            logger.warning(
                "Every result for genre=%s mood=%s is low confidence - the catalog "
                "does not really serve this profile",
                prefs["favorite_genre"], prefs["favorite_mood"],
            )

        return results

    def average_confidence(self, results: Optional[List[RetrievedSong]] = None) -> float:
        """Mean confidence across a result set (0.0 for an empty set)."""
        if not results:
            return 0.0
        return round(sum(r.confidence for r in results) / len(results), 3)
