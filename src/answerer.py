"""
The Answer Agent — prompt construction, generation, and grounding enforcement.

This is the "augmented" half of retrieval-augmented generation, and the single
most important design decision in the system lives here:

    **The model does not choose the songs.**

Retrieval and the weighted reranker fix the list. The prompt hands that list to
the model and asks it only to write prose about it, citing the passage each
fact came from. Every recommendation therefore still traces back to a specific
weight in a specific line of recommender.py, exactly as it did before a model
was involved — what changed is who writes the sentences, not who picks.

Anything the model writes that is not supported by the retrieved passages is
dropped or withheld by guardrails.check_generated_answer, and a withheld answer
falls back to the deterministic template explainer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.explainer import ExplanationAgent
from src.guardrails import GroundingReport, check_generated_answer
from src.llm_client import (
    GenerationError,
    Generator,
    TemplateGenerator,
)
from src.logging_setup import get_logger
from src.models import RetrievedChunk, RetrievedSong

logger = get_logger("answerer")

REFUSAL = "I do not know based on the catalog I have."


@dataclass
class Answer:
    """One generated (or fallen-back) answer plus how it was produced."""
    text: str
    backend: str
    citations: List[str] = field(default_factory=list)
    grounding: Optional[GroundingReport] = None
    fell_back: bool = False


def build_prompt(
    query: str,
    chunks: Sequence[RetrievedChunk],
    records: Sequence[RetrievedSong],
) -> str:
    """
    Assembles the generation prompt: context passages, the request, the list.

    The passages carry their chunk ids because those ids are the citation
    handles the grounding check verifies against. The song list is numbered and
    explicitly fenced ("do NOT add, drop, or substitute") because the model's
    job is to describe a decision already made, not to make one.
    """
    passages = []
    for retrieved in chunks:
        chunk = retrieved.chunk
        passages.append(f"[{chunk.chunk_id}] title: {chunk.title}\n{chunk.text}")

    selected = []
    for i, record in enumerate(records, start=1):
        song = record.song
        selected.append(
            f'{i}. "{song["title"]}" — {song["artist"]} '
            f"(score {record.score:.2f}/4.0, confidence {record.confidence:.2f}, {record.band})"
        )

    return f"""You are HarmonyRanker's music explainer.

Context passages — the ONLY facts you may use:

{chr(10).join(passages)}

Listener's request:
{query}

Songs the ranker already selected (do NOT add, drop, or substitute any):
{chr(10).join(selected)}

Rules:
- Write about ONLY the numbered songs above, in that order.
- Never invent a title, artist, genre, mood, or number. Every attribute you
  state must appear verbatim in a context passage.
- End every factual sentence with the id of the passage it came from, in
  square brackets, e.g. [song:47].
- At most two sentences per song. No preamble, no closing summary.
- If the passages do not support a recommendation, reply with exactly:
  "{REFUSAL}"
"""


class AnswerAgent:
    """
    Chooses a generation backend, enforces grounding, and falls back safely.

    Keeps the same reliability counters the ExplanationAgent exposes, so the
    run report reads identically whichever backend produced the text.
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        fallback: Optional[ExplanationAgent] = None,
    ):
        self.generator = generator
        self.fallback = fallback or ExplanationAgent()
        self.stats: Dict[str, int] = {
            "answers": 0,
            "withheld": 0,
            "fell_back": 0,
            "sentences_total": 0,
            "sentences_cited": 0,
            "fabricated_citations": 0,
            "substituted_titles": 0,
        }

    def _template_answer(self, user_prefs: Dict[str, Any],
                         records: Sequence[RetrievedSong]) -> str:
        """Deterministic prose, one line per song, with citations attached."""
        lines = []
        for record in records:
            text = self.fallback.explain(user_prefs, record)
            lines.append(f"{record.song['title']} — {text} [song:{record.song['id']}]")
        return " ".join(lines)

    def answer(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        records: Sequence[RetrievedSong],
        user_prefs: Dict[str, Any],
        catalog_titles: Optional[Sequence[str]] = None,
    ) -> Answer:
        """
        Produces the explanation for one set of recommendations.

        Falls back to the template explainer when there is no model backend,
        when the call fails, or when the grounding check withholds the output.
        """
        self.stats["answers"] += 1

        if not records:
            return Answer(text=REFUSAL, backend="none")

        # No model, or the offline backend: go straight to deterministic prose.
        if self.generator is None or isinstance(self.generator, TemplateGenerator):
            return Answer(
                text=self._template_answer(user_prefs, records),
                backend="template",
            )

        prompt = build_prompt(query, chunks, records)

        try:
            raw = self.generator.generate(prompt)
        except GenerationError as exc:
            logger.error("Generation failed (%s); falling back to the template explainer", exc)
            self.stats["fell_back"] += 1
            return Answer(
                text=self._template_answer(user_prefs, records),
                backend="template",
                fell_back=True,
            )

        if raw.strip() == REFUSAL:
            return Answer(text=REFUSAL, backend=self.generator.name)

        allowed_ids = [c.chunk.chunk_id for c in chunks]
        candidate_titles = [r.song["title"] for r in records]
        cleaned, report = check_generated_answer(
            raw,
            allowed_chunk_ids=allowed_ids,
            candidate_titles=candidate_titles,
            catalog_titles=catalog_titles or candidate_titles,
        )

        self.stats["sentences_total"] += report.sentences_total
        self.stats["sentences_cited"] += report.sentences_cited
        self.stats["fabricated_citations"] += len(report.fabricated_citations)
        self.stats["substituted_titles"] += len(report.substituted_titles)

        if report.withheld:
            self.stats["withheld"] += 1
            self.stats["fell_back"] += 1
            return Answer(
                text=self._template_answer(user_prefs, records),
                backend="template",
                grounding=report,
                fell_back=True,
            )

        from src.guardrails import extract_citations
        return Answer(
            text=cleaned,
            backend=self.generator.name,
            citations=extract_citations(cleaned),
            grounding=report,
        )

    def grounding_rate(self) -> float:
        """
        Share of generated sentences that survived the grounding check.

        Same contract as ExplanationAgent.grounding_rate: 1.0 when nothing was
        dropped, including when no model ran at all.
        """
        total = self.stats["sentences_total"]
        if not total:
            return self.fallback.grounding_rate()
        dropped = total - self.stats["sentences_cited"]
        return round(max(0.0, (total - dropped)) / total, 3)
