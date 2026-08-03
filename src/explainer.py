"""
The Explanation Agent — the "augmented generation" half of the pipeline.

It never sees the catalog. It only sees a RetrievedSong handed over by the
Retriever, and it may only say things that record can support:

    retrieved evidence  ->  claims  ->  grounding check  ->  prose

A claim that fails the grounding check is dropped and logged, and if every
claim for a song is dropped the agent withholds the explanation rather than
inventing one. Generation is template-based and deterministic (no model call,
no network, no API key), so a given record always produces the same text —
which is what makes the output testable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from src.guardrails import is_claim_grounded
from src.logging_setup import get_logger
from src.models import BAND_HIGH, BAND_LOW, BAND_MEDIUM, RetrievedSong

logger = get_logger("explainer")

WITHHELD_EXPLANATION = (
    "No verifiable evidence for this pick, so no explanation is offered."
)

# How sure the wording is allowed to sound, per confidence band.
OPENINGS = {
    BAND_HIGH: "Strong match",
    BAND_MEDIUM: "Partial match",
    BAND_LOW: "Weak match",
}


@dataclass(frozen=True)
class Claim:
    """
    One assertion the agent wants to make, tied to the song field that
    licenses it. `field_name` + `value` are what the grounding check verifies;
    `text` is only ever rendered if that check passes.
    """
    field_name: str
    value: Any
    text: str
    supporting: bool


class ExplanationAgent:
    """
    Turns retrieved evidence into grounded prose.

    Tracks its own reliability counters (`stats`) so a run can report how many
    claims it made, how many it had to drop, and how many explanations it
    withheld entirely.
    """

    def __init__(self) -> None:
        self.stats: Dict[str, int] = {
            "explanations": 0,
            "claims_made": 0,
            "claims_dropped": 0,
            "withheld": 0,
        }

    def explain(self, user_prefs: Dict[str, Any], record: RetrievedSong) -> str:
        """Generates the grounded 'why this song' text for one retrieved record."""
        claims = self._build_claims(record)
        verified = self._verify(claims, record)

        self.stats["explanations"] += 1

        if not verified:
            self.stats["withheld"] += 1
            logger.error(
                "Withholding explanation for '%s': no claim survived grounding",
                record.song.get("title", "<unknown>"),
            )
            return WITHHELD_EXPLANATION

        supporting = [c.text for c in verified if c.supporting]
        contrasting = [c.text for c in verified if not c.supporting]

        opening = OPENINGS.get(record.band, "Match")
        parts = [f"{opening} (confidence {record.confidence:.2f})"]

        if supporting:
            parts.append(f": recommended because {_join(supporting)}")
        else:
            parts.append(": nothing in your profile lines up with this song")

        text = "".join(parts)

        if contrasting:
            # Semicolons here: the caveat fragments already contain commas.
            text += f". Caveat: {'; '.join(contrasting)}"

        if record.band == BAND_LOW:
            text += ". Treat this as a stretch pick rather than a real match"

        return text + "."

    # -- internals -----------------------------------------------------------

    def _build_claims(self, record: RetrievedSong) -> List[Claim]:
        """Turns each scoring signal — hit or miss — into a claim about the song."""
        claims: List[Claim] = []

        for signal in record.breakdown.signals:
            if signal.name == "genre":
                text = (
                    f"it's {signal.song_value}, your favorite genre"
                    if signal.matched
                    else f"it's {signal.song_value}, not the {signal.target_value} you asked for"
                )
            elif signal.name == "mood":
                text = (
                    f"the mood is {signal.song_value}, exactly what you wanted"
                    if signal.matched
                    else f"the mood is {signal.song_value}, not {signal.target_value}"
                )
            elif signal.name == "energy":
                text = (
                    f"its energy ({signal.song_value:.2f}) sits close to your "
                    f"target of {signal.target_value:.2f}"
                    if signal.matched
                    else f"its energy ({signal.song_value:.2f}) is off your "
                         f"target of {signal.target_value:.2f}"
                )
            else:  # pragma: no cover - guards against a signal added without wording
                logger.warning("No wording for signal '%s'; skipping", signal.name)
                continue

            claims.append(Claim(
                field_name=signal.field_name,
                value=signal.song_value,
                text=text,
                supporting=signal.matched,
            ))

        return claims

    def _verify(self, claims: List[Claim], record: RetrievedSong) -> List[Claim]:
        """
        Drops any claim not literally supported by the retrieved record.

        In normal operation nothing is dropped — the claims are built from the
        same record they are checked against. It fires when the record and the
        evidence have drifted apart (a stale breakdown, a mutated song dict, a
        future signal reading a field that isn't there), which is exactly the
        case where an ungrounded explanation would otherwise sound confident.
        """
        verified: List[Claim] = []
        for claim in claims:
            self.stats["claims_made"] += 1
            if is_claim_grounded(claim.field_name, claim.value, record.song):
                verified.append(claim)
            else:
                self.stats["claims_dropped"] += 1
                logger.error(
                    "Dropped ungrounded claim on '%s': %s=%r not supported by the record",
                    record.song.get("title", "<unknown>"), claim.field_name, claim.value,
                )
        return verified

    def grounding_rate(self) -> float:
        """Share of attempted claims that survived the grounding check (1.0 = all)."""
        made = self.stats["claims_made"]
        if not made:
            return 1.0
        return round((made - self.stats["claims_dropped"]) / made, 3)


def _join(fragments: List[str]) -> str:
    """Joins fragments as 'a', 'a and b', or 'a, b and c'."""
    if len(fragments) == 1:
        return fragments[0]
    return f"{', '.join(fragments[:-1])} and {fragments[-1]}"
