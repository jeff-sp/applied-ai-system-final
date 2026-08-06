"""
Tests for prompt construction and the grounding guardrail.

These are the tests that justify letting a language model write user-facing
prose at all. The system's central claim is that the deterministic ranker
chooses the songs and the model only describes them; these prove that a model
which violates that claim gets caught rather than printed.
"""

import pytest

from src.answerer import REFUSAL, AnswerAgent, build_prompt
from src.guardrails import (
    GroundingReport,
    QueryError,
    check_generated_answer,
    extract_citations,
    validate_query,
)
from src.llm_client import BrokenGenerator, FakeGenerator, RaisingGenerator
from src.models import (
    KIND_PROSE,
    KIND_SONG,
    Chunk,
    RetrievedChunk,
    RetrievedSong,
    ScoreBreakdown,
    SignalMatch,
)

SONG = {
    "id": 42, "title": "Plastic Love", "artist": "Mariya Takeuchi",
    "genre": "city pop", "mood": "nostalgic", "energy": 0.72,
    "tempo_bpm": 103, "valence": 0.63, "danceability": 0.81, "acousticness": 0.12,
}
PREFS = {"favorite_genre": "city pop", "favorite_mood": "nostalgic", "target_energy": 0.7}


def record(song=SONG, score=3.5, confidence=0.8, band="high"):
    breakdown = ScoreBreakdown(score=score, signals=[
        SignalMatch(name="genre", field_name="genre", matched=True, points=1.5,
                    max_points=1.5, song_value=song["genre"], target_value=song["genre"]),
        SignalMatch(name="mood", field_name="mood", matched=True, points=0.5,
                    max_points=0.5, song_value=song["mood"], target_value=song["mood"]),
        SignalMatch(name="energy", field_name="energy", matched=True, points=1.8,
                    max_points=2.0, song_value=song["energy"], target_value=0.7),
    ], reasons=["matched"])
    return RetrievedSong(song=song, score=score, reasons=["matched"],
                         breakdown=breakdown, confidence=confidence, band=band)


def chunks():
    return [
        RetrievedChunk(chunk=Chunk(chunk_id="song:42", kind=KIND_SONG,
                                   title="Plastic Love — Mariya Takeuchi",
                                   text="A city pop song with a nostalgic mood.",
                                   source="data/songs.csv"), similarity=0.7),
        RetrievedChunk(chunk=Chunk(chunk_id="genres:city-pop:0", kind=KIND_PROSE,
                                   title="City pop", text="City pop is Japanese urban pop.",
                                   source="docs/kb/genres.md"), similarity=0.6),
    ]


# --- query validation -------------------------------------------------------

def test_a_normal_query_is_accepted_and_trimmed():
    assert validate_query("  something upbeat  ") == "something upbeat"


def test_an_empty_query_is_rejected():
    """Embedding an empty string still returns a confident-looking top-k."""
    with pytest.raises(QueryError, match="empty"):
        validate_query("   ")


def test_a_non_string_query_is_rejected():
    with pytest.raises(QueryError, match="string"):
        validate_query(None)


def test_an_overlong_query_is_rejected_not_truncated():
    """Silently answering a different question is worse than refusing."""
    with pytest.raises(QueryError, match="limit"):
        validate_query("x" * 5000)


# --- citation parsing -------------------------------------------------------

def test_citations_are_extracted_in_order():
    text = "First [song:1]. Second [genres:lofi:2]. Third [song:1]."
    assert extract_citations(text) == ["song:1", "genres:lofi:2", "song:1"]


def test_malformed_markers_are_not_citations():
    assert extract_citations("see [Song 42] and [42] and []") == []


# --- the three grounding layers ---------------------------------------------

def test_a_fully_cited_answer_survives_intact():
    text = "It is city pop [song:42]. The genre is Japanese urban pop [genres:city-pop:0]."
    cleaned, report = check_generated_answer(
        text, ["song:42", "genres:city-pop:0"], ["Plastic Love"], ["Plastic Love"],
    )

    assert not report.withheld
    assert report.sentences_cited == 2
    assert report.cited_ratio == 1.0
    assert cleaned == text


def test_a_fabricated_citation_drops_only_that_sentence():
    """One bad provenance marker does not necessarily poison the whole answer."""
    text = ("It is city pop [song:42]. "
            "It also won a Grammy [song:9999]. "
            "The genre is Japanese urban pop [genres:city-pop:0].")
    cleaned, report = check_generated_answer(
        text, ["song:42", "genres:city-pop:0"], ["Plastic Love"], ["Plastic Love"],
    )

    assert report.fabricated_citations == ["song:9999"]
    assert "Grammy" not in cleaned
    assert "city pop [song:42]" in cleaned
    assert not report.withheld


def test_a_substituted_song_withholds_the_whole_answer():
    """
    The dangerous failure: the ranker chose the songs, and a model quietly
    swapping one in is exactly what this system promises cannot happen. It must
    not be salvageable by dropping a sentence.
    """
    text = 'You should hear "Master of Puppets" instead [song:42].'
    cleaned, report = check_generated_answer(
        text, ["song:42"], ["Plastic Love"], ["Plastic Love", "Master of Puppets"],
    )

    assert report.withheld
    assert report.substituted_titles == ["Master of Puppets"]
    assert "did not select" in report.reason


def test_a_candidate_song_is_not_treated_as_a_substitution():
    text = 'The pick is "Plastic Love" [song:42].'
    _, report = check_generated_answer(
        text, ["song:42"], ["Plastic Love"], ["Plastic Love", "Master of Puppets"],
    )

    assert not report.withheld
    assert report.substituted_titles == []


def test_thin_citation_density_withholds():
    """Catches an answer that narrates freely and sprinkles one marker."""
    text = ("This is a wonderful song [song:42]. It will change your life. "
            "Everyone loves it. You will play it constantly. It is the best.")
    _, report = check_generated_answer(
        text, ["song:42"], ["Plastic Love"], ["Plastic Love"],
    )

    assert report.withheld
    assert report.cited_ratio < 0.6
    assert "sentences" in report.reason


def test_an_answer_where_nothing_survives_is_withheld():
    text = "Completely made up [song:9999]."
    _, report = check_generated_answer(
        text, ["song:42"], ["Plastic Love"], ["Plastic Love"],
    )

    assert report.withheld
    assert "no sentence survived" in report.reason


def test_an_empty_report_has_a_zero_ratio_rather_than_dividing_by_zero():
    assert GroundingReport().cited_ratio == 0.0


# --- prompt construction ----------------------------------------------------

def test_the_prompt_carries_chunk_ids_as_citation_handles():
    prompt = build_prompt("something nostalgic", chunks(), [record()])

    assert "[song:42]" in prompt
    assert "[genres:city-pop:0]" in prompt


def test_the_prompt_fences_the_song_list():
    """The model describes a decision already made; it does not make one."""
    prompt = build_prompt("something nostalgic", chunks(), [record()])

    assert "do NOT add, drop, or substitute" in prompt
    assert '"Plastic Love" — Mariya Takeuchi' in prompt
    assert REFUSAL in prompt


# --- AnswerAgent end to end -------------------------------------------------

def test_a_clean_model_answer_is_used():
    good = "It is city pop with a nostalgic mood [song:42]."
    agent = AnswerAgent(FakeGenerator([good]))
    answer = agent.answer("q", chunks(), [record()], PREFS, catalog_titles=["Plastic Love"])

    assert answer.backend == "fake"
    assert not answer.fell_back
    assert answer.citations == ["song:42"]


def test_a_fabricating_model_falls_back_to_the_template():
    agent = AnswerAgent(BrokenGenerator())
    answer = agent.answer("q", chunks(), [record()], PREFS, catalog_titles=["Plastic Love"])

    assert answer.fell_back
    assert answer.backend == "template"
    assert agent.stats["fabricated_citations"] >= 1
    # The fallback text is the deterministic explainer's, and it is grounded.
    assert "Plastic Love" in answer.text


def test_a_substituting_model_is_withheld_and_falls_back():
    agent = AnswerAgent(BrokenGenerator(substitute_title="Master of Puppets"))
    answer = agent.answer("q", chunks(), [record()], PREFS,
                          catalog_titles=["Plastic Love", "Master of Puppets"])

    assert answer.fell_back
    assert agent.stats["withheld"] == 1
    assert agent.stats["substituted_titles"] == 1
    assert "Master of Puppets" not in answer.text


def test_a_failing_generator_falls_back_rather_than_printing_an_error():
    """DocuBot returns the error string as the answer; here we degrade instead."""
    agent = AnswerAgent(RaisingGenerator())
    answer = agent.answer("q", chunks(), [record()], PREFS, catalog_titles=["Plastic Love"])

    assert answer.fell_back
    assert answer.backend == "template"
    assert "API error" not in answer.text
    assert "Traceback" not in answer.text


def test_an_explicit_model_refusal_is_passed_through():
    agent = AnswerAgent(FakeGenerator([REFUSAL]))
    answer = agent.answer("q", chunks(), [record()], PREFS, catalog_titles=["Plastic Love"])

    assert answer.text == REFUSAL
    assert not answer.fell_back


def test_no_records_produces_a_refusal():
    agent = AnswerAgent(FakeGenerator(["unused"]))
    answer = agent.answer("q", chunks(), [], PREFS)

    assert answer.text == REFUSAL


def test_the_template_backend_emits_citations():
    """
    So the offline path runs through the grounding check for real rather than
    bypassing it — the guardrail is exercised by every keyless run.
    """
    agent = AnswerAgent(None)
    answer = agent.answer("q", chunks(), [record()], PREFS)

    assert answer.backend == "template"
    assert "[song:42]" in answer.text


def test_grounding_rate_keeps_the_explanation_agent_contract():
    agent = AnswerAgent(FakeGenerator(["It is city pop [song:42]."]))
    agent.answer("q", chunks(), [record()], PREFS, catalog_titles=["Plastic Love"])

    assert agent.grounding_rate() == 1.0
