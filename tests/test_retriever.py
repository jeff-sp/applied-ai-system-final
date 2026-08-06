"""Tests for the Retriever: catalog loading, ranking, and confidence scoring."""

import pytest

from src.guardrails import CatalogError, ProfileError
from src.models import BAND_HIGH, BAND_LOW, MAX_SCORE
from src.recommender import load_songs
from src.retriever import Retriever

CATALOG_HEADER = (
    "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
)


def make_songs():
    return [
        {
            "id": 1, "title": "Test Pop Track", "artist": "Test Artist",
            "genre": "pop", "mood": "happy", "energy": 0.8, "tempo_bpm": 120,
            "valence": 0.9, "danceability": 0.8, "acousticness": 0.2,
        },
        {
            "id": 2, "title": "Chill Lofi Loop", "artist": "Test Artist",
            "genre": "lofi", "mood": "chill", "energy": 0.4, "tempo_bpm": 80,
            "valence": 0.6, "danceability": 0.5, "acousticness": 0.9,
        },
    ]


def pop_prefs():
    return {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}


# --- catalog loading --------------------------------------------------------

def test_load_songs_skips_malformed_rows_but_keeps_the_run_alive(tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        CATALOG_HEADER
        + "1,Good Track,Artist,pop,happy,0.8,120,0.9,0.8,0.2\n"
        + "2,Bad Energy,Artist,pop,happy,7.5,120,0.9,0.8,0.2\n"   # out of range
        + "3,Bad Number,Artist,pop,happy,loud,120,0.9,0.8,0.2\n"  # unparseable
        + "4,Also Good,Artist,lofi,chill,0.4,80,0.6,0.5,0.9\n",
        encoding="utf-8",
    )

    songs = load_songs(str(csv_path))

    assert [s["title"] for s in songs] == ["Good Track", "Also Good"]


def test_load_songs_raises_on_missing_file():
    with pytest.raises(CatalogError, match="not found"):
        load_songs("data/does_not_exist.csv")


def test_load_songs_raises_on_missing_columns(tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text("id,title,artist\n1,Track,Artist\n", encoding="utf-8")

    with pytest.raises(CatalogError, match="missing required column"):
        load_songs(str(csv_path))


def test_load_songs_raises_when_every_row_is_bad(tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        CATALOG_HEADER + "1,Broken,Artist,pop,happy,9.9,120,0.9,0.8,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="no valid songs"):
        load_songs(str(csv_path))


# --- retrieval --------------------------------------------------------------

def test_retrieve_returns_k_records_ranked_by_score():
    results = Retriever(make_songs()).retrieve(pop_prefs(), k=2)

    assert len(results) == 2
    assert results[0].song["genre"] == "pop"
    assert results[0].score >= results[1].score


def test_retrieve_respects_k_and_never_exceeds_the_catalog():
    retriever = Retriever(make_songs())
    assert len(retriever.retrieve(pop_prefs(), k=1)) == 1
    assert len(retriever.retrieve(pop_prefs(), k=50)) == 2


def test_retrieve_rejects_an_invalid_profile():
    with pytest.raises(ProfileError):
        Retriever(make_songs()).retrieve({"favorite_genre": "pop"}, k=2)


def test_retriever_rejects_an_empty_catalog():
    with pytest.raises(ValueError, match="non-empty"):
        Retriever([])


def test_every_record_carries_its_own_evidence():
    # The point of the RAG shape: the explanation layer downstream never has
    # to go back to the catalog, because the record already holds the evidence.
    record = Retriever(make_songs()).retrieve(pop_prefs(), k=1)[0]

    assert record.reasons
    assert len(record.breakdown.signals) == 3
    assert {s.name for s in record.breakdown.signals} == {"genre", "mood", "energy"}


# --- confidence scoring -----------------------------------------------------

def test_perfect_match_is_maximum_confidence():
    record = Retriever(make_songs()).retrieve(pop_prefs(), k=1)[0]

    assert record.score == MAX_SCORE
    assert record.confidence == 1.0
    assert record.band == BAND_HIGH


def test_energy_only_match_lands_in_the_low_band():
    # A listener whose genre and mood match nothing: the song still scores on
    # energy alone, and confidence must expose that as a weak recommendation.
    prefs = {"favorite_genre": "metal", "favorite_mood": "angry", "target_energy": 0.8}
    record = Retriever(make_songs()).retrieve(prefs, k=1)[0]

    assert record.band == BAND_LOW
    assert record.confidence < 0.45


def test_confidence_stays_within_bounds_for_every_song_and_profile():
    retriever = Retriever(make_songs())
    for prefs in (pop_prefs(), {"favorite_genre": "metal", "favorite_mood": "angry", "target_energy": 0.0}):
        for record in retriever.retrieve(prefs, k=5):
            assert 0.0 <= record.confidence <= 1.0


def test_confidence_ranks_a_full_match_above_a_partial_one():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.6}
    results = Retriever(make_songs()).retrieve(prefs, k=2)

    assert results[0].confidence > results[1].confidence


def test_average_confidence_summarizes_a_result_set():
    retriever = Retriever(make_songs())
    results = retriever.retrieve(pop_prefs(), k=2)

    expected = round(sum(r.confidence for r in results) / 2, 3)
    assert retriever.average_confidence(results) == expected
    assert retriever.average_confidence([]) == 0.0


def test_retriever_tracks_low_confidence_results():
    retriever = Retriever(make_songs())
    retriever.retrieve({"favorite_genre": "metal", "favorite_mood": "angry", "target_energy": 0.8}, k=2)

    assert retriever.stats["queries"] == 1
    assert retriever.stats["low_confidence"] >= 1


def test_real_catalog_loads_and_serves_a_profile():
    retriever = Retriever.from_csv("data/songs.csv")
    results = retriever.retrieve({"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3}, k=3)

    assert len(results) == 3
    assert results[0].song["genre"] == "lofi"
    assert results[0].band == BAND_HIGH


# --- the semantic path ------------------------------------------------------
#
# SemanticRetriever composes the Retriever above rather than replacing it, so
# everything already asserted in this file still holds. What follows pins the
# two-stage hybrid: embeddings shortlist, the weighted scorer reranks.

import pytest

from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import HashingEmbedder
from src.main import PROFILES
from src.models import (
    MAX_SCORE,
    RetrievedSong,
    ScoreBreakdown,
    SignalMatch,
    compute_confidence,
)
from src.retriever import (
    RERANK_CANDIDATES,
    SemanticRetriever,
    blended_rank,
    extract_prefs,
)
from src.vector_store import VectorStore


@pytest.fixture(scope="module")
def semantic():
    """One store for the module: building it 277 times would be wasteful."""
    chunks = build_corpus()
    embedder = HashingEmbedder()
    store = VectorStore.build(chunks, embedder, corpus_fingerprint(chunks))
    return SemanticRetriever(Retriever.from_csv("data/songs.csv"), store, embedder)


# --- extract_prefs ----------------------------------------------------------

def test_extract_prefs_reads_genre_and_mood_from_plain_language(semantic):
    prefs = extract_prefs("energetic jazz with real swing", semantic.songs)

    assert prefs["favorite_genre"] == "jazz"
    assert prefs["favorite_mood"] == "energetic"


def test_extract_prefs_prefers_the_longest_genre_match(semantic):
    """'indie pop' and 'pop' are both catalog genres; shortest-match would lose."""
    assert extract_prefs("some indie pop please", semantic.songs)["favorite_genre"] == "indie pop"


def test_extract_prefs_maps_everyday_synonyms(semantic):
    assert extract_prefs("something sad", semantic.songs)["favorite_mood"] == "melancholy"
    assert extract_prefs("music for the gym", semantic.songs)["favorite_mood"] == "energetic"


def test_extract_prefs_honours_negation(semantic):
    """
    Regression: "nothing too energetic" matched the mood `energetic`, so the
    lofi demo profile recommended exactly what the listener asked to avoid.
    """
    prefs = extract_prefs("mellow lofi, nothing too energetic", semantic.songs)

    assert prefs["favorite_mood"] != "energetic"
    assert prefs["target_energy"] <= 0.4


def test_extract_prefs_reads_energy_hints(semantic):
    assert extract_prefs("music for a hard workout", semantic.songs)["target_energy"] >= 0.8
    assert extract_prefs("something to study to", semantic.songs)["target_energy"] <= 0.4


def test_extract_prefs_falls_back_to_the_top_retrieved_song(semantic):
    """A miss degrades ranking slightly rather than breaking the request."""
    fallback = {"genre": "reggae", "mood": "relaxed", "energy": 0.44}
    prefs = extract_prefs("something nice please", semantic.songs, fallback_song=fallback)

    assert prefs["favorite_genre"] == "reggae"
    assert prefs["favorite_mood"] == "relaxed"


def test_extract_prefs_rejects_an_empty_query(semantic):
    from src.guardrails import QueryError

    with pytest.raises(QueryError):
        extract_prefs("   ", semantic.songs)


@pytest.mark.parametrize("name", list(PROFILES))
def test_each_demo_query_recovers_its_own_profile(name, semantic):
    """
    Pins the two representations together. PROFILES carries both a natural
    language `query` and a structured `prefs`; if someone edits one without the
    other, this fails rather than silently drifting.
    """
    profile = PROFILES[name]
    prefs = extract_prefs(profile["query"], semantic.songs)

    assert prefs["favorite_genre"] == profile["prefs"]["favorite_genre"]
    assert prefs["favorite_mood"] == profile["prefs"]["favorite_mood"]


# --- retrieval and reranking ------------------------------------------------

def test_semantic_retrieve_returns_songs_prose_and_the_inferred_profile(semantic):
    records, prose, prefs = semantic.retrieve("confident funk with a groove", k=3)

    assert len(records) == 3
    assert prose
    assert prefs["favorite_genre"] == "funk"


def test_reranking_surfaces_the_requested_genre(semantic):
    """The hybrid's whole justification: the scorer fixes retrieval's ordering."""
    records, _, _ = semantic.retrieve("angry metal, as heavy as you have", k=3)

    assert all(r.song["genre"] == "metal" for r in records)


def test_results_carry_their_retrieval_provenance(semantic):
    records, _, _ = semantic.retrieve("nostalgic city pop for a late night drive", k=3)

    for record in records:
        assert record.similarity is not None
        assert record.chunk_id == f"song:{record.song['id']}"


def test_the_shortlist_contains_the_target_genre_for_every_demo_query(semantic):
    """recall@k is the metric the two-stage design actually depends on."""
    for name, profile in PROFILES.items():
        records, _, _ = semantic.retrieve(profile["query"], k=RERANK_CANDIDATES)
        genres = {r.song["genre"] for r in records}
        assert profile["prefs"]["favorite_genre"] in genres, name


def test_score_still_means_the_weighted_score(semantic):
    """The blend is a private sort key; it is never written back onto a record."""
    records, _, _ = semantic.retrieve("confident funk with a groove", k=5)

    for record in records:
        assert 0.0 <= record.score <= MAX_SCORE


def test_blended_rank_weights_similarity_and_score():
    breakdown = ScoreBreakdown(score=MAX_SCORE, signals=[], reasons=[])
    perfect = RetrievedSong(song={}, score=MAX_SCORE, reasons=[], breakdown=breakdown,
                            confidence=1.0, band=BAND_HIGH)

    assert blended_rank(1.0, perfect) == pytest.approx(1.0)
    assert blended_rank(0.0, perfect) < blended_rank(1.0, perfect)


def test_semantic_retriever_keeps_the_retriever_counters(semantic):
    before = semantic.stats["queries"]
    semantic.retrieve("something upbeat and happy", k=2)

    assert semantic.stats["queries"] == before + 1
    assert semantic.stats["chunks_retrieved"] > 0
    assert semantic.average_similarity() != 0.0


def test_context_k_controls_how_much_prose_is_retrieved(semantic):
    _, prose, _ = semantic.retrieve("something for studying", k=2, context_k=2)
    assert len(prose) == 2


# --- confidence -------------------------------------------------------------

def test_similarity_none_reproduces_the_original_confidence():
    """The 63 pre-RAG tests depend on this being byte-identical."""
    breakdown = ScoreBreakdown(score=3.0, signals=[
        SignalMatch("genre", "genre", True, 1.5, 1.5, "lofi", "lofi"),
        SignalMatch("mood", "mood", False, 0.0, 0.5, "chill", "happy"),
    ], reasons=[])

    assert compute_confidence(breakdown) == compute_confidence(breakdown, MAX_SCORE, None)


def test_the_offline_embedder_does_not_move_confidence(semantic):
    """
    An uncalibrated lexical fallback orders results but must not tell the user
    how sure the system is, so offline confidence matches classic mode exactly.
    """
    records, _, prefs = semantic.retrieve("mellow lofi beats to study to", k=3)
    classic = semantic.retriever.retrieve(prefs, k=3)

    by_id = {r.song["id"]: r.confidence for r in classic}
    for record in records:
        if record.song["id"] in by_id:
            assert record.confidence == pytest.approx(by_id[record.song["id"]])
