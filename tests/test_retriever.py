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
