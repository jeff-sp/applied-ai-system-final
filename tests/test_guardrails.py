"""Tests for the guardrail layer: row validation, profile validation, grounding."""

import pytest

from src.guardrails import (
    ProfileError,
    RowError,
    is_claim_grounded,
    validate_song_row,
    validate_user_prefs,
)


def make_row(**overrides):
    row = {
        "id": "1",
        "title": "Test Pop Track",
        "artist": "Test Artist",
        "genre": "Pop",
        "mood": "Happy",
        "energy": "0.8",
        "tempo_bpm": "120",
        "valence": "0.9",
        "danceability": "0.8",
        "acousticness": "0.2",
    }
    row.update(overrides)
    return row


# --- validate_song_row ------------------------------------------------------

def test_valid_row_is_typed_and_normalized():
    song = validate_song_row(make_row(), line_no=2)

    assert song["id"] == 1 and isinstance(song["id"], int)
    assert song["energy"] == 0.8 and isinstance(song["energy"], float)
    assert song["tempo_bpm"] == 120 and isinstance(song["tempo_bpm"], int)
    # case is normalized so "Pop" from a hand-edited CSV still matches "pop"
    assert song["genre"] == "pop"
    assert song["mood"] == "happy"


def test_row_missing_a_value_is_rejected():
    with pytest.raises(RowError, match="missing value"):
        validate_song_row(make_row(genre=""), line_no=7)


def test_row_with_unparseable_number_is_rejected():
    with pytest.raises(RowError, match="line 4"):
        validate_song_row(make_row(energy="loud"), line_no=4)


@pytest.mark.parametrize("bad_energy", ["1.4", "-0.2"])
def test_row_with_out_of_range_energy_is_rejected(bad_energy):
    # Out-of-range values would make the 1 - abs(distance) closeness score
    # go negative, so they are rejected rather than silently mis-scored.
    with pytest.raises(RowError, match="outside the 0.0-1.0 range"):
        validate_song_row(make_row(energy=bad_energy), line_no=5)


def test_row_with_implausible_tempo_is_rejected():
    with pytest.raises(RowError, match="implausible"):
        validate_song_row(make_row(tempo_bpm="9000"), line_no=6)


# --- validate_user_prefs ----------------------------------------------------

def test_profile_missing_key_is_fatal():
    with pytest.raises(ProfileError, match="target_energy"):
        validate_user_prefs({"favorite_genre": "pop", "favorite_mood": "happy"})


def test_profile_with_non_numeric_energy_is_fatal():
    with pytest.raises(ProfileError, match="numeric"):
        validate_user_prefs(
            {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": "high"}
        )


def test_profile_with_blank_genre_is_fatal():
    with pytest.raises(ProfileError, match="favorite_genre"):
        validate_user_prefs(
            {"favorite_genre": "  ", "favorite_mood": "happy", "target_energy": 0.5}
        )


def test_out_of_range_energy_is_clamped_not_fatal():
    validated = validate_user_prefs(
        {"favorite_genre": "Pop", "favorite_mood": "Happy", "target_energy": 2.5}
    )
    assert validated["target_energy"] == 1.0
    assert validated["favorite_genre"] == "pop"


def test_validation_does_not_mutate_the_callers_profile():
    original = {"favorite_genre": "Pop", "favorite_mood": "Happy", "target_energy": 2.5}
    validate_user_prefs(original)
    assert original == {"favorite_genre": "Pop", "favorite_mood": "Happy", "target_energy": 2.5}


# --- is_claim_grounded ------------------------------------------------------

def test_claim_matching_the_record_is_grounded():
    song = {"genre": "lofi", "energy": 0.42}
    assert is_claim_grounded("genre", "lofi", song)
    assert is_claim_grounded("energy", 0.42, song)


def test_claim_contradicting_the_record_is_not_grounded():
    song = {"genre": "lofi", "energy": 0.42}
    assert not is_claim_grounded("genre", "metal", song)
    assert not is_claim_grounded("energy", 0.91, song)


def test_claim_about_a_missing_field_is_not_grounded():
    assert not is_claim_grounded("tempo_bpm", 120, {"genre": "lofi"})
