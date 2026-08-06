"""Tests for cosine similarity, top-k ordering, and index persistence."""

import json
import math

import pytest

from src.models import KIND_PROSE, KIND_SONG, Chunk
from src.vector_store import (
    DimensionMismatchError,
    IndexUnavailableError,
    StoredVector,
    VectorStore,
    cosine_similarity,
)


def chunk(chunk_id, kind=KIND_SONG, text="text", title="Title"):
    return Chunk(chunk_id=chunk_id, kind=kind, title=title, text=text,
                 source="s", metadata={"song_id": 1})


def store_of(*pairs):
    entries = [
        StoredVector(chunk=c, vector=v, norm=math.sqrt(sum(x * x for x in v)))
        for c, v in pairs
    ]
    return VectorStore(entries, "test-embedder", len(pairs[0][1]), "fingerprint")


# --- cosine -----------------------------------------------------------------

def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_change_similarity():
    """Cosine is computed explicitly, so an unnormalized backend still works."""
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_hand_computed_value():
    # cos between (1,0) and (1,1) is 1/sqrt(2)
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))


def test_unequal_lengths_raise_rather_than_truncate():
    """
    Zipping a 768-dim query against 2048-dim stored vectors produced numbers in
    a believable range from two unrelated vector spaces. Nothing looked wrong.
    """
    with pytest.raises(DimensionMismatchError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0, 0.0])


# --- search -----------------------------------------------------------------

def test_results_are_ordered_by_descending_similarity():
    store = store_of(
        (chunk("far"), [0.0, 1.0]),
        (chunk("near"), [1.0, 0.0]),
        (chunk("mid"), [1.0, 1.0]),
    )
    ids = [r.chunk.chunk_id for r in store.search([1.0, 0.0], k=3)]

    assert ids == ["near", "mid", "far"]
    similarities = [r.similarity for r in store.search([1.0, 0.0], k=3)]
    assert similarities == sorted(similarities, reverse=True)


def test_ties_break_on_chunk_id_for_reproducibility():
    store = store_of((chunk("b"), [1.0, 0.0]), (chunk("a"), [1.0, 0.0]))
    assert [r.chunk.chunk_id for r in store.search([1.0, 0.0], k=2)] == ["a", "b"]


def test_k_larger_than_the_store_returns_everything():
    store = store_of((chunk("a"), [1.0, 0.0]), (chunk("b"), [0.0, 1.0]))
    assert len(store.search([1.0, 0.0], k=99)) == 2


def test_k_of_zero_returns_nothing():
    store = store_of((chunk("a"), [1.0, 0.0]))
    assert store.search([1.0, 0.0], k=0) == []


def test_negative_k_is_rejected():
    store = store_of((chunk("a"), [1.0, 0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        store.search([1.0, 0.0], k=-1)


def test_search_can_filter_by_kind():
    store = store_of(
        (chunk("s1", KIND_SONG), [1.0, 0.0]),
        (chunk("p1", KIND_PROSE), [1.0, 0.0]),
    )
    assert [r.chunk.chunk_id for r in store.search([1.0, 0.0], k=5, kind=KIND_PROSE)] == ["p1"]


def test_stratified_search_returns_both_kinds_separately():
    """A single top-k would let 203 song cards crowd the prose out entirely."""
    store = store_of(
        (chunk("s1", KIND_SONG), [1.0, 0.0]),
        (chunk("s2", KIND_SONG), [0.9, 0.1]),
        (chunk("p1", KIND_PROSE), [0.5, 0.5]),
    )
    songs, prose = store.search_stratified([1.0, 0.0], k_songs=2, k_prose=1)

    assert [r.chunk.chunk_id for r in songs] == ["s1", "s2"]
    assert [r.chunk.chunk_id for r in prose] == ["p1"]


def test_counts_reports_each_kind():
    store = store_of(
        (chunk("s1", KIND_SONG), [1.0]),
        (chunk("p1", KIND_PROSE), [1.0]),
        (chunk("p2", KIND_PROSE), [1.0]),
    )
    assert store.counts() == {KIND_SONG: 1, KIND_PROSE: 2}


# --- persistence ------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    original = store_of(
        (chunk("song:1", KIND_SONG, text="a lofi song"), [0.123456, 0.5]),
        (chunk("genres:lofi:0", KIND_PROSE, text="lofi is warm"), [0.25, 0.75]),
    )
    path = str(tmp_path / "index.jsonl")
    original.save(path)
    loaded = VectorStore.load(path)

    assert len(loaded) == 2
    assert loaded.embedder_name == original.embedder_name
    assert loaded.dimensions == original.dimensions
    assert loaded.fingerprint == original.fingerprint

    for before, after in zip(original.entries, loaded.entries):
        assert after.chunk.chunk_id == before.chunk.chunk_id
        assert after.chunk.kind == before.chunk.kind
        assert after.chunk.text == before.chunk.text
        assert after.chunk.metadata == before.chunk.metadata
        assert after.vector == pytest.approx(before.vector, abs=1e-6)


def test_load_reports_a_missing_index_actionably(tmp_path):
    with pytest.raises(IndexUnavailableError, match="src.ingest"):
        VectorStore.load(str(tmp_path / "absent.jsonl"))


def test_load_rejects_a_file_without_a_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"chunk_id": "a"}) + "\n", encoding="utf-8")

    with pytest.raises(IndexUnavailableError, match="header"):
        VectorStore.load(str(path))


def test_load_rejects_a_malformed_record(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "embedder": "e", "dimensions": 2, "fingerprint": "f"})
        + "\n" + "{not json}\n",
        encoding="utf-8",
    )
    with pytest.raises(IndexUnavailableError, match="malformed"):
        VectorStore.load(str(path))


def test_load_rejects_a_header_with_no_vectors(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "embedder": "e", "dimensions": 2, "fingerprint": "f"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(IndexUnavailableError, match="no vectors"):
        VectorStore.load(str(path))


# --- freshness --------------------------------------------------------------

def test_a_matching_fingerprint_is_fresh():
    assert store_of((chunk("a"), [1.0])).check_fresh("fingerprint") is True


def test_a_different_fingerprint_is_stale():
    assert store_of((chunk("a"), [1.0])).check_fresh("something-else") is False


def test_an_index_with_no_fingerprint_is_never_considered_fresh():
    store = store_of((chunk("a"), [1.0]))
    store.fingerprint = ""
    assert store.check_fresh("") is False


# --- freshness against the embedder ------------------------------------------
#
# A fingerprint says which corpus the vectors describe, never which vector space
# they live in. These cover the second half.

class _StubEmbedder:
    """Just the two attributes mismatch_reason reads."""

    def __init__(self, name, dimensions):
        self.name = name
        self.dimensions = dimensions


def test_the_same_embedder_is_fresh():
    store = store_of((chunk("a"), [1.0]))
    assert store.check_fresh("fingerprint", _StubEmbedder("test-embedder", 1)) is True


def test_a_different_embedder_is_not_fresh():
    """The shipped bug: matching corpus, incompatible vectors."""
    store = store_of((chunk("a"), [1.0]))
    embedder = _StubEmbedder("gemini-embedding-2@768", 768)

    assert store.check_fresh("fingerprint", embedder) is False
    assert "test-embedder" in store.mismatch_reason("fingerprint", embedder)


def test_the_same_embedder_name_at_a_different_width_is_not_fresh():
    store = store_of((chunk("a"), [1.0]))
    embedder = _StubEmbedder("test-embedder", 768)

    assert store.check_fresh("fingerprint", embedder) is False
    assert "768" in store.mismatch_reason("fingerprint", embedder)


def test_a_stale_corpus_is_reported_before_the_embedder():
    """Rebuilding is the fix for both, and the corpus is the root cause."""
    store = store_of((chunk("a"), [1.0]))
    reason = store.mismatch_reason("other", _StubEmbedder("other-embedder", 99))

    assert "different corpus" in reason
