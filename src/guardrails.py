"""
Guardrails: validation, safe failure, and grounding checks.

Three jobs, matching the "Guardrails & Logging" box in
diagrams/architecture.mmd:

1. Catalog guardrails  — a malformed CSV row is skipped and logged, never
   allowed to crash a run or silently poison the ranking with `None`s.
2. Profile guardrails  — a nonsense user profile is rejected (or clamped)
   before it reaches the scorer.
3. Grounding guardrails — a claim the explanation agent wants to make is
   checked against the retrieved record before it can be printed.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

from src.config import MAX_QUERY_CHARS, MIN_CITED_RATIO
from src.logging_setup import get_logger

logger = get_logger("guardrails")

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
)

# Fields the scoring math assumes live on a 0.0-1.0 scale. A value outside
# that range would break the `1 - abs(distance)` closeness formula (it can go
# negative), so such rows are rejected rather than quietly mis-scored.
UNIT_INTERVAL_FIELDS: Tuple[str, ...] = ("energy", "valence", "danceability", "acousticness")

REQUIRED_PROFILE_KEYS: Tuple[str, ...] = ("favorite_genre", "favorite_mood", "target_energy")

# Reasonable bounds for a tempo, used only to flag obviously corrupt rows.
MIN_TEMPO_BPM = 20
MAX_TEMPO_BPM = 400


class CatalogError(Exception):
    """The catalog as a whole is unusable (missing file, no columns, no valid rows)."""


class RowError(Exception):
    """A single catalog row is unusable. Recoverable: skip the row, keep the run."""


class ProfileError(Exception):
    """The user profile is unusable (missing or non-numeric fields)."""


class QueryError(Exception):
    """The natural-language query is unusable (empty, or absurdly long)."""


# A citation marker as the generation prompt asks for it: [song:47],
# [genres:city-pop:1]. Deliberately strict — a malformed marker is not a
# citation, and treating it as one would weaken the check it exists to enforce.
CITATION_RE = re.compile(r"\[([a-z0-9_]+:[a-z0-9\-]+(?::\d+)?)\]")

# Sentence splitter for the citation-density check. Same simple rule as the
# chunker: punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def validate_song_row(row: Dict[str, str], line_no: int) -> Dict[str, Any]:
    """
    Type-casts and range-checks one CSV row.

    Returns the typed song dict, or raises RowError with a message naming the
    line and the offending field. The caller decides whether to skip or abort;
    this function never returns partial data.
    """
    missing = [c for c in REQUIRED_COLUMNS if row.get(c) in (None, "")]
    if missing:
        raise RowError(f"line {line_no}: missing value(s) for {', '.join(missing)}")

    try:
        song: Dict[str, Any] = {
            "id": int(row["id"]),
            "title": row["title"].strip(),
            "artist": row["artist"].strip(),
            "genre": row["genre"].strip().lower(),
            "mood": row["mood"].strip().lower(),
            "energy": float(row["energy"]),
            "tempo_bpm": int(row["tempo_bpm"]),
            "valence": float(row["valence"]),
            "danceability": float(row["danceability"]),
            "acousticness": float(row["acousticness"]),
        }
    except (TypeError, ValueError) as exc:
        raise RowError(f"line {line_no}: could not parse numeric field ({exc})") from exc

    for numeric_field in UNIT_INTERVAL_FIELDS:
        value = song[numeric_field]
        if not 0.0 <= value <= 1.0:
            raise RowError(
                f"line {line_no}: {numeric_field}={value} is outside the 0.0-1.0 range"
            )

    if not MIN_TEMPO_BPM <= song["tempo_bpm"] <= MAX_TEMPO_BPM:
        raise RowError(f"line {line_no}: tempo_bpm={song['tempo_bpm']} is implausible")

    if not song["title"]:
        raise RowError(f"line {line_no}: title is blank")

    return song


def validate_user_prefs(user_prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks a user profile before it reaches the scorer.

    Missing or non-numeric fields are fatal (ProfileError) — guessing what a
    user meant would produce confident, wrong recommendations. An out-of-range
    `target_energy` is recoverable, so it is clamped to [0, 1] and logged.

    Returns a validated copy; the caller's dict is left untouched.
    """
    if not isinstance(user_prefs, dict):
        raise ProfileError(f"profile must be a dict, got {type(user_prefs).__name__}")

    missing = [key for key in REQUIRED_PROFILE_KEYS if key not in user_prefs]
    if missing:
        raise ProfileError(f"profile is missing required key(s): {', '.join(missing)}")

    validated = dict(user_prefs)

    for key in ("favorite_genre", "favorite_mood"):
        value = validated[key]
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"profile.{key} must be a non-empty string, got {value!r}")
        validated[key] = value.strip().lower()

    try:
        energy = float(validated["target_energy"])
    except (TypeError, ValueError) as exc:
        raise ProfileError(
            f"profile.target_energy must be numeric, got {validated['target_energy']!r}"
        ) from exc

    if not 0.0 <= energy <= 1.0:
        clamped = min(1.0, max(0.0, energy))
        logger.warning(
            "target_energy=%s is outside 0.0-1.0; clamping to %s", energy, clamped
        )
        energy = clamped
    validated["target_energy"] = energy

    return validated


def is_claim_grounded(field_name: str, asserted_value: Any, song: Dict[str, Any]) -> bool:
    """
    Verifies that a claim the explanation agent wants to make is actually
    supported by the retrieved record.

    This is the check that keeps generation *augmented* rather than invented:
    the agent may only assert a value that is literally present in the song
    it is explaining. Floats are compared with a tolerance; everything else
    is compared case-insensitively as text.
    """
    if field_name not in song:
        return False

    actual = song[field_name]
    if isinstance(actual, float) or isinstance(asserted_value, float):
        try:
            return abs(float(actual) - float(asserted_value)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(actual).strip().lower() == str(asserted_value).strip().lower()


def summarize_skips(skipped: List[str], source: str) -> None:
    """Logs a single summary line for rows dropped during a catalog load."""
    if skipped:
        logger.warning(
            "%s: skipped %d malformed row(s); first: %s",
            source, len(skipped), skipped[0],
        )


# --- Generation guardrails --------------------------------------------------
#
# Everything above this line guards inputs. Everything below guards output, and
# only matters once a language model is writing the prose. The template
# explainer in explainer.py cannot fabricate, because it only interpolates
# fields it was handed. A model can, so its output is checked before it is shown.


def validate_query(text: Any) -> str:
    """
    Checks a natural-language query before it is embedded.

    Empty queries are fatal: embedding one produces a meaningless vector that
    still returns a confident-looking top-k. Over-long input is fatal too,
    rather than truncated, because silently answering a different question than
    the one asked is worse than refusing.
    """
    if not isinstance(text, str):
        raise QueryError(f"query must be a string, got {type(text).__name__}")

    cleaned = text.strip()
    if not cleaned:
        raise QueryError("query is empty")

    if len(cleaned) > MAX_QUERY_CHARS:
        raise QueryError(
            f"query is {len(cleaned)} characters; the limit is {MAX_QUERY_CHARS}"
        )

    return cleaned


def extract_citations(text: str) -> List[str]:
    """Returns every citation marker in the text, in order, with duplicates."""
    return CITATION_RE.findall(text)


def split_sentences(text: str) -> List[str]:
    """Splits generated prose into sentences for the density check."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


@dataclass
class GroundingReport:
    """
    The verdict on one generated answer.

    `withheld` is the decision; the counters exist so the run report can show
    how often the model strayed and in which direction.
    """
    sentences_total: int = 0
    sentences_cited: int = 0
    fabricated_citations: List[str] = field(default_factory=list)
    substituted_titles: List[str] = field(default_factory=list)
    withheld: bool = False
    reason: str = ""

    @property
    def cited_ratio(self) -> float:
        """Share of sentences carrying at least one valid citation."""
        if not self.sentences_total:
            return 0.0
        return round(self.sentences_cited / self.sentences_total, 3)


def check_generated_answer(
    text: str,
    allowed_chunk_ids: Iterable[str],
    candidate_titles: Iterable[str],
    catalog_titles: Iterable[str],
    min_cited_ratio: float = MIN_CITED_RATIO,
) -> Tuple[str, GroundingReport]:
    """
    Verifies generated prose against the evidence it was given.

    Three failure modes, deliberately treated differently:

    1. **Fabricated citation** — the answer cites a chunk that was not
       retrieved. The offending sentence is dropped and counted; the rest of
       the answer survives, because one bad provenance marker does not
       necessarily poison the whole response.
    2. **Substituted song** — the answer names a catalog song that the ranker
       did not select. This withholds the *entire* answer. It is the dangerous
       failure: the deterministic ranker chose the recommendations, and a model
       quietly swapping one in is precisely the property this system promises
       cannot happen. It must not be salvageable by dropping a sentence.
    3. **Thin citation density** — too few sentences carry a citation. Catches
       an answer that narrates freely and sprinkles one marker to look grounded.

    Returns the (possibly reduced) text and the report. When the report says
    withheld, the caller falls back to the deterministic template explainer.
    """
    allowed: Set[str] = set(allowed_chunk_ids)
    candidates = {t.strip().lower() for t in candidate_titles}
    report = GroundingReport()

    # 2. Substitution check first: it is fatal, so there is no point repairing
    #    sentences in an answer that is about to be discarded wholesale.
    lowered = text.lower()
    for title in catalog_titles:
        normalized = title.strip().lower()
        if not normalized or normalized in candidates:
            continue
        # Word-boundary match so a short title cannot match inside another word.
        if re.search(rf"\b{re.escape(normalized)}\b", lowered):
            report.substituted_titles.append(title)

    sentences = split_sentences(text)
    report.sentences_total = len(sentences)

    if report.substituted_titles:
        report.withheld = True
        report.reason = (
            "answer named song(s) the ranker did not select: "
            + ", ".join(sorted(report.substituted_titles))
        )
        logger.error("Withholding generated answer: %s", report.reason)
        return text, report

    # 1. Citation validity, sentence by sentence.
    kept: List[str] = []
    for sentence in sentences:
        citations = extract_citations(sentence)
        bad = [c for c in citations if c not in allowed]
        if bad:
            report.fabricated_citations.extend(bad)
            logger.error(
                "Dropping sentence citing unretrieved chunk(s) %s: %r",
                ", ".join(bad), sentence[:80],
            )
            continue
        if citations:
            report.sentences_cited += 1
        kept.append(sentence)

    cleaned = " ".join(kept).strip()

    # 3. Density, measured against the sentences the model actually wrote.
    if not cleaned:
        report.withheld = True
        report.reason = "no sentence survived the citation check"
    elif report.cited_ratio < min_cited_ratio:
        report.withheld = True
        report.reason = (
            f"only {report.sentences_cited}/{report.sentences_total} sentences "
            f"were cited (minimum ratio {min_cited_ratio})"
        )

    if report.withheld:
        logger.error("Withholding generated answer: %s", report.reason)

    return cleaned, report
