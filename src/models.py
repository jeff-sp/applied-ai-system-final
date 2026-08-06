"""
Shared data structures for the retrieval-augmented pipeline.

This module holds the vocabulary every other module speaks in, and nothing
else — it imports no project code except src/config.py, which itself imports
none, so it can never take part in an import cycle. See
diagrams/architecture.mmd for how the pieces connect.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import CONFIDENCE_SIMILARITY_WEIGHT

# A perfect song scores W_GENRE + W_MOOD + W_ENERGY (see recommender.score_song).
MAX_SCORE = 4.0

# Chunk kinds. A song card is one CSV row rendered as prose; a prose chunk is a
# window over the hand-written knowledge base in docs/kb/.
KIND_SONG = "song"
KIND_PROSE = "prose"

# Confidence bands. These are policy, not math: they decide when the
# explanation agent is allowed to sound sure of itself.
BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.45


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_valence: float
    target_danceability: float
    target_acousticness: float


@dataclass
class SignalMatch:
    """
    One scoring signal's verdict for one song.

    This is the unit of *evidence*: the explanation agent may only make a
    claim that traces back to one of these, and `field_name` names the song
    field the claim is checked against.
    """
    name: str           # "genre" | "mood" | "energy"
    field_name: str     # the key in the song dict this signal reads
    matched: bool
    points: float
    max_points: float
    song_value: Any     # the song's actual value for this field
    target_value: Any   # what the profile wanted


@dataclass
class ScoreBreakdown:
    """The full result of scoring one song — the score plus why."""
    score: float
    signals: List[SignalMatch] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of scoring signals that actually matched (0.0-1.0)."""
        if not self.signals:
            return 0.0
        return sum(1 for s in self.signals if s.matched) / len(self.signals)


@dataclass(frozen=True)
class Chunk:
    """
    One retrievable unit of text — the thing that actually gets embedded.

    Two kinds live side by side. A `song` chunk is one catalog row rendered as
    a sentence or two (never split; a card is already tiny). A `prose` chunk is
    an overlapping window over docs/kb/. Keeping both in one store lets a single
    query surface the songs *and* the background that explains why they fit.

    `chunk_id` is the citation handle: it is what the generation prompt asks the
    model to cite, and what guardrails.check_generated_answer verifies against.
    """
    chunk_id: str                  # "song:47" | "genres:city-pop:0"
    kind: str                      # KIND_SONG | KIND_PROSE
    title: str                     # used by the DOC_PREFIX template
    text: str
    source: str                    # "data/songs.csv" | "docs/kb/genres.md"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk plus how closely it matched the query (cosine, -1.0 to 1.0)."""
    chunk: Chunk
    similarity: float


@dataclass
class RetrievedSong:
    """
    One retrieved candidate, carrying its own evidence.

    This is what the Retriever hands to the Explanation Agent — the record
    *and* everything needed to justify it, so the agent never has to invent
    or re-derive a fact.

    `similarity` and `chunk_id` are populated only on the semantic path and
    default to None, so the classic scoring path and every test written against
    it are unaffected.
    """
    song: Dict[str, Any]
    score: float
    reasons: List[str]
    breakdown: ScoreBreakdown
    confidence: float
    band: str
    similarity: Optional[float] = None
    chunk_id: Optional[str] = None


def rescale_similarity(similarity: float, floor: float, ceiling: float) -> float:
    """
    Maps a raw cosine into [0, 1] using the embedder's own useful range.

    Necessary because cosine similarities do not span [0, 1] in practice — real
    Gemini scores cluster in roughly 0.25-0.85, and the hashing embedder runs
    lower still. Fed in raw, the term would sit near 0.6 for everything and add
    no information. `floor` and `ceiling` are per-embedder tuned policy, not
    math; see Embedder.similarity_floor.
    """
    if ceiling <= floor:
        return 0.0
    scaled = (similarity - floor) / (ceiling - floor)
    return min(1.0, max(0.0, scaled))


def compute_confidence(
    breakdown: ScoreBreakdown,
    max_score: float = MAX_SCORE,
    similarity: Optional[float] = None,
    similarity_floor: float = 0.0,
    similarity_ceiling: float = 1.0,
) -> float:
    """
    Rates how sure the system is that this song fits the profile, in [0, 1].

    Two ingredients, because either one alone lies:

    - `score_ratio` — how many of the available points the song earned.
      Alone it is fooled by energy: with W_ENERGY = 2.0 a song can bank half
      the points on a single signal and look like a solid match.
    - `coverage`    — how many *distinct* signals matched. Alone it ignores
      how good each match was.

    Weighting score slightly higher (0.6/0.4) keeps a near-perfect single-signal
    song from being punished too hard, while coverage still drags down the
    "recommended on energy alone" cases the README calls out.

    On the semantic path a third ingredient is available: how closely the song
    card matched the listener's actual words. It is folded in at
    CONFIDENCE_SIMILARITY_WEIGHT. When `similarity` is None — the classic path,
    and every pre-existing test — the result is byte-identical to before.
    """
    score_ratio = breakdown.score / max_score if max_score else 0.0
    score_ratio = min(1.0, max(0.0, score_ratio))
    base = 0.6 * score_ratio + 0.4 * breakdown.coverage

    if similarity is None:
        return round(base, 3)

    semantic = rescale_similarity(similarity, similarity_floor, similarity_ceiling)
    weight = CONFIDENCE_SIMILARITY_WEIGHT
    return round((1.0 - weight) * base + weight * semantic, 3)


def confidence_band(confidence: float) -> str:
    """Buckets a confidence score into high / medium / low."""
    if confidence >= HIGH_THRESHOLD:
        return BAND_HIGH
    if confidence >= MEDIUM_THRESHOLD:
        return BAND_MEDIUM
    return BAND_LOW
