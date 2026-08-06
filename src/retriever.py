"""
The Retriever — the "R" in retrieval-augmented generation.

It owns the knowledge base (data/songs.csv), indexes it, and answers a taste
profile with the top-k candidates *plus the evidence for each one*. The
Explanation Agent downstream is given only what this returns, so anything the
system says about a song traces back to a retrieved record.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import (
    BLEND_ALPHA,
    DEFAULT_CONTEXT_K,
    RERANK_CANDIDATES,
)
from src.config import DEFAULT_CATALOG as _DEFAULT_CATALOG
from src.embeddings import Embedder
from src.guardrails import validate_query, validate_user_prefs
from src.logging_setup import get_logger
from src.models import (
    BAND_LOW,
    MAX_SCORE,
    RetrievedChunk,
    RetrievedSong,
    compute_confidence,
    confidence_band,
)
from src.recommender import build_record, load_songs, rank_songs
from src.vector_store import VectorStore

logger = get_logger("retriever")

DEFAULT_CATALOG = _DEFAULT_CATALOG

# Everyday words that point at a catalog mood without naming it. Longest match
# wins at lookup time, so "melancholy" beats a synonym mapping to it.
MOOD_SYNONYMS: Dict[str, str] = {
    "sad": "melancholy",
    "heartbreak": "melancholy",
    "heartbroken": "melancholy",
    "down": "melancholy",
    "pumped": "energetic",
    "hype": "energetic",
    "workout": "energetic",
    "gym": "energetic",
    "mellow": "chill",
    "calm": "relaxed",
    "unwind": "relaxed",
    "sleep": "relaxed",
    "study": "focused",
    "coding": "focused",
    "concentrate": "focused",
    "driving": "intense",
    "heavy": "intense",
    "retro": "nostalgic",
    "throwback": "nostalgic",
    "80s": "nostalgic",
    "swagger": "confident",
    "strut": "confident",
    "upbeat": "happy",
    "feel-good": "happy",
    "angry": "angry",
    "rage": "angry",
}

# Phrases that imply a target energy. Checked longest-first.
ENERGY_HINTS: Tuple[Tuple[str, float], ...] = (
    ("nothing too energetic", 0.25),
    ("hard workout", 0.9),
    ("high-energy", 0.9),
    ("high energy", 0.9),
    ("late night", 0.45),
    ("background", 0.35),
    ("workout", 0.85),
    ("party", 0.85),
    ("gym", 0.85),
    ("energetic", 0.85),
    ("upbeat", 0.8),
    ("intense", 0.8),
    ("heavy", 0.8),
    ("loud", 0.8),
    ("dance", 0.8),
    ("study", 0.3),
    ("sleep", 0.25),
    ("mellow", 0.3),
    ("chill", 0.3),
    ("quiet", 0.25),
    ("calm", 0.25),
    ("relax", 0.3),
    ("sad", 0.3),
)

DEFAULT_TARGET_ENERGY = 0.5


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


# --- Natural-language query handling ----------------------------------------


# Cues that invert the term following them. Without this, "nothing too
# energetic" matches the mood `energetic` and the system recommends exactly what
# the listener asked to avoid — measured on the lofi demo query.
NEGATION_CUES: Tuple[str, ...] = (
    "nothing too", "nothing", "not too", "not ", "no ", "without", "avoid", "less ",
)

# How far back to look for a negation cue before a matched term.
_NEGATION_WINDOW = 16


def _is_negated(lowered: str, term: str) -> bool:
    """True when the text immediately before `term` negates it."""
    position = lowered.find(term)
    if position <= 0:
        return False
    window = lowered[max(0, position - _NEGATION_WINDOW):position]
    return any(cue in window for cue in NEGATION_CUES)


def _longest_match(query: str, vocabulary: Sequence[str]) -> Optional[str]:
    """
    Finds the longest non-negated vocabulary term appearing in the query.

    Longest-first matters: "indie pop" and "pop" are both catalog genres, and a
    shortest-match rule would file every indie pop request under pop.

    Negated terms are skipped rather than matched, so "nothing too energetic"
    and "not metal" describe what the listener does not want.
    """
    lowered = query.lower()
    matches = [
        term for term in vocabulary
        if term in lowered and not _is_negated(lowered, term)
    ]
    if not matches:
        return None
    return max(matches, key=len)


def extract_prefs(
    query: str,
    songs: Sequence[Dict[str, Any]],
    fallback_song: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Derives a scoring profile from a natural-language query.

    Deterministic and vocabulary-driven — no model call. It reads the genres and
    moods that actually exist in the catalog, matches the longest one present in
    the query, and maps a handful of everyday synonyms onto catalog labels.

    Anything it cannot find is taken from `fallback_song`: the top-1 result that
    embedding retrieval already returned. That keeps the reranker working from
    what retrieval actually found rather than from a guess, and means a failure
    here degrades ranking slightly instead of breaking the request.
    """
    lowered = validate_query(query).lower()

    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    genre = _longest_match(lowered, genres)

    mood = _longest_match(lowered, moods)
    if mood is None:
        synonym = _longest_match(lowered, list(MOOD_SYNONYMS))
        if synonym:
            mood = MOOD_SYNONYMS[synonym]

    energy: Optional[float] = None
    for phrase, value in ENERGY_HINTS:
        if phrase in lowered:
            energy = value
            break

    inferred = []
    if genre is None and fallback_song:
        genre = fallback_song["genre"]
        inferred.append("genre")
    if mood is None and fallback_song:
        mood = fallback_song["mood"]
        inferred.append("mood")
    if energy is None:
        if fallback_song:
            energy = float(fallback_song["energy"])
            inferred.append("energy")
        else:
            energy = DEFAULT_TARGET_ENERGY

    if inferred:
        logger.info(
            "Query %r: inferred %s from the top retrieved song",
            query, ", ".join(inferred),
        )

    return validate_user_prefs({
        "favorite_genre": genre or (songs[0]["genre"] if songs else "pop"),
        "favorite_mood": mood or (songs[0]["mood"] if songs else "happy"),
        "target_energy": energy,
    })


def blended_rank(similarity: float, record: RetrievedSong) -> float:
    """
    The sort key for reranking: semantic similarity plus weighted score.

    Kept as a free function, and deliberately never written back onto the
    record — `RetrievedSong.score` continues to mean exactly what it has always
    meant, the weighted score out of MAX_SCORE. The blend is a private ordering
    decision, not a new public number.
    """
    return BLEND_ALPHA * similarity + (1.0 - BLEND_ALPHA) * (record.score / MAX_SCORE)


class SemanticRetriever:
    """
    Two-stage retrieval: embeddings shortlist, the weighted scorer reranks.

    Composes a plain Retriever rather than replacing it, so the catalog
    coverage warnings, the stats counters, and average_confidence all keep
    working unchanged — and so the classic path stays available for comparison.

    Why two stages rather than pure embedding search: an embedding understands
    "something mellow for studying" but cannot honour a numeric target energy,
    and the hand-tuned weights are the part of this project whose behaviour is
    actually explainable. Measured on this corpus, recall@20 for the target
    genre is 19/19 across genres, so the shortlist reliably contains what the
    reranker needs.
    """

    def __init__(self, retriever: "Retriever", store: VectorStore, embedder: Embedder):
        self.retriever = retriever
        self.store = store
        self.embedder = embedder
        self.songs_by_id = {song["id"]: song for song in retriever.songs}
        self.stats: Dict[str, Any] = retriever.stats
        self.stats.setdefault("chunks_retrieved", 0)
        self.stats.setdefault("prose_retrieved", 0)
        self.stats.setdefault("similarity_sum", 0.0)

    @property
    def songs(self) -> List[Dict[str, Any]]:
        return self.retriever.songs

    def average_confidence(self, results: Optional[List[RetrievedSong]] = None) -> float:
        return self.retriever.average_confidence(results)

    def average_similarity(self) -> float:
        """Mean cosine across every chunk retrieved this run."""
        count = self.stats.get("chunks_retrieved", 0)
        if not count:
            return 0.0
        return round(self.stats.get("similarity_sum", 0.0) / count, 3)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        context_k: int = DEFAULT_CONTEXT_K,
        candidates: int = RERANK_CANDIDATES,
    ) -> Tuple[List[RetrievedSong], List[RetrievedChunk], Dict[str, Any]]:
        """
        Answers a natural-language query.

        Returns the reranked top-k songs, the prose chunks to inject as context,
        and the profile that was inferred from the query (so the caller can show
        the user what the system understood them to mean).
        """
        cleaned = validate_query(query)
        query_vector = self.embedder.embed_query(cleaned)

        song_hits, prose_hits = self.store.search_stratified(
            query_vector, k_songs=candidates, k_prose=context_k,
        )

        self.stats["chunks_retrieved"] += len(song_hits) + len(prose_hits)
        self.stats["prose_retrieved"] += len(prose_hits)
        self.stats["similarity_sum"] += sum(
            h.similarity for h in list(song_hits) + list(prose_hits)
        )

        if not song_hits:
            logger.warning("No song chunks retrieved for %r", cleaned)
            return [], list(prose_hits), extract_prefs(cleaned, self.songs)

        top_song = self.songs_by_id.get(song_hits[0].chunk.metadata.get("song_id"))
        prefs = extract_prefs(cleaned, self.songs, fallback_song=top_song)
        logger.info(
            "Query %r -> genre=%s mood=%s target_energy=%.2f",
            cleaned, prefs["favorite_genre"], prefs["favorite_mood"], prefs["target_energy"],
        )

        # Rerank the shortlist with the existing weighted scorer, folding the
        # retrieval similarity into confidence but never into `score`.
        records: List[Tuple[float, str, RetrievedSong]] = []
        for hit in song_hits:
            song = self.songs_by_id.get(hit.chunk.metadata.get("song_id"))
            if song is None:
                logger.warning(
                    "Index references song id %s which is not in the catalog - "
                    "the index is stale", hit.chunk.metadata.get("song_id"),
                )
                continue

            record = build_record(prefs, song)
            record.similarity = hit.similarity
            record.chunk_id = hit.chunk.chunk_id

            # Similarity is always recorded, but only folded into confidence
            # when the backend's scores are calibrated enough to justify it.
            if self.embedder.contributes_confidence:
                record.confidence = compute_confidence(
                    record.breakdown, MAX_SCORE,
                    similarity=hit.similarity,
                    similarity_floor=self.embedder.similarity_floor,
                    similarity_ceiling=self.embedder.similarity_ceiling,
                )
                record.band = confidence_band(record.confidence)
            records.append((blended_rank(hit.similarity, record), hit.chunk.chunk_id, record))

        records.sort(key=lambda item: (-item[0], item[1]))
        results = [item[2] for item in records[:k]]

        self.stats["queries"] += 1
        self.stats["results"] += len(results)
        self.stats["low_confidence"] += sum(1 for r in results if r.band == BAND_LOW)

        for rank, record in enumerate(results, start=1):
            logger.debug(
                "  #%d %s (%s) score=%.2f sim=%.3f confidence=%.2f band=%s",
                rank, record.song["title"], record.song["genre"],
                record.score, record.similarity or 0.0, record.confidence, record.band,
            )

        return results, list(prose_hits), prefs
