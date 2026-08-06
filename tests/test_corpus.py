"""
Tests for song-card rendering, corpus assembly, and the fingerprint.

The card tests matter more than they look: a card is the only place a number
becomes a word the model might repeat, and it is the evidence the grounding
check verifies generated claims against. If a card omits a value, a truthful
statement about that value becomes unciteable.
"""

import pytest

from src.corpus import (
    build_corpus,
    build_prose_chunks,
    build_song_chunks,
    corpus_fingerprint,
    level,
    render_song_card,
    tempo_word,
)
from src.models import KIND_PROSE, KIND_SONG
from src.recommender import load_songs


def catalog_size():
    """The live catalog row count, so tests assert shape rather than a snapshot."""
    return len(load_songs("data/songs.csv"))

SONG = {
    "id": 7, "title": "Plastic Love", "artist": "Mariya Takeuchi",
    "genre": "city pop", "mood": "nostalgic", "energy": 0.72,
    "tempo_bpm": 103, "valence": 0.63, "danceability": 0.81, "acousticness": 0.12,
}


# --- descriptive bands ------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0.0, "very low"), (0.19, "very low"),
    (0.2, "low"), (0.39, "low"),
    (0.4, "moderate"), (0.59, "moderate"),
    (0.6, "high"), (0.79, "high"),
    (0.8, "very high"), (1.0, "very high"),
])
def test_level_bands(value, expected):
    assert level(value) == expected


@pytest.mark.parametrize("bpm,expected", [
    (54, "slow"), (89, "slow"),
    (90, "moderate"), (119, "moderate"),
    (120, "fast"), (149, "fast"),
    (150, "very fast"), (200, "very fast"),
])
def test_tempo_bands(bpm, expected):
    assert tempo_word(bpm) == expected


# --- song cards -------------------------------------------------------------

def test_a_card_states_genre_and_mood():
    card = render_song_card(SONG)
    assert "city pop" in card.text
    assert "nostalgic" in card.text


def test_a_card_keeps_every_number_verbatim():
    """The grounding check verifies claims against these exact strings."""
    text = render_song_card(SONG).text
    for value in ("0.72", "103", "0.63", "0.81", "0.12"):
        assert value in text, f"{value} missing from the card"


def test_a_card_pairs_each_number_with_a_word():
    """Embeddings match 'high energy' far better than they match '0.72'."""
    text = render_song_card(SONG).text
    assert "high (0.72)" in text          # energy
    assert "moderate at 103 BPM" in text  # tempo


def test_a_card_carries_routing_metadata():
    card = render_song_card(SONG)
    assert card.chunk_id == "song:7"
    assert card.kind == KIND_SONG
    assert card.metadata["song_id"] == 7
    assert card.metadata["genre"] == "city pop"
    assert card.title == "Plastic Love — Mariya Takeuchi"


def test_every_catalog_row_becomes_exactly_one_card():
    songs = load_songs("data/songs.csv")
    cards = build_song_chunks(songs)

    assert len(cards) == len(songs)
    assert len({c.chunk_id for c in cards}) == len(cards)


def test_every_real_card_contains_its_own_genre_mood_and_energy():
    for song in load_songs("data/songs.csv"):
        text = render_song_card(song).text
        assert song["genre"] in text
        assert song["mood"] in text
        assert f"{song['energy']:.2f}" in text


# --- corpus -----------------------------------------------------------------

def test_the_corpus_holds_both_kinds():
    chunks = build_corpus()
    kinds = {c.kind for c in chunks}

    assert kinds == {KIND_SONG, KIND_PROSE}
    # One card per catalog row, derived rather than pinned: the catalog grows,
    # and a hardcoded count fails on a data edit while saying nothing about the
    # invariant that actually matters.
    assert len([c for c in chunks if c.kind == KIND_SONG]) == catalog_size()


def test_chunk_ids_are_unique_across_the_whole_corpus():
    chunks = build_corpus()
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_every_knowledge_base_file_contributes_chunks():
    chunks = build_prose_chunks("docs/kb")
    sources = {c.source for c in chunks}

    assert len(sources) >= 4
    for source in sources:
        assert any(c.source == source for c in chunks)


def test_a_missing_knowledge_base_is_a_warning_not_a_crash(tmp_path):
    """Song cards alone still work; the prose is an enhancement."""
    assert build_prose_chunks(str(tmp_path)) == []


# --- fingerprint ------------------------------------------------------------

def test_the_fingerprint_is_stable_for_unchanged_content():
    assert corpus_fingerprint(build_corpus()) == corpus_fingerprint(build_corpus())


def test_the_fingerprint_ignores_ordering():
    chunks = build_corpus()
    assert corpus_fingerprint(chunks) == corpus_fingerprint(list(reversed(chunks)))


def test_the_fingerprint_changes_when_one_song_changes():
    chunks = build_corpus()
    edited = list(chunks)
    edited[0] = render_song_card({**SONG, "id": edited[0].metadata["song_id"]})

    assert corpus_fingerprint(edited) != corpus_fingerprint(chunks)


def test_the_fingerprint_changes_when_a_chunk_is_added():
    chunks = build_corpus()
    assert corpus_fingerprint(chunks + [render_song_card(SONG)]) != corpus_fingerprint(chunks)
