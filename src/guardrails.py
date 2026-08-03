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

from typing import Any, Dict, List, Tuple

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
