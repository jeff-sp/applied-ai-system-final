"""
Tests for the build-time ingest CLI.

`--check` is the cache-invalidation contract: it is what a pre-commit hook or
CI job runs to catch an index that no longer matches the data it was built
from. `--dry-run` must never call the API, since its whole purpose is making
chunk tuning free.
"""

import json

import pytest

from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import HashingEmbedder
from src.ingest import main as ingest_main
from src.models import KIND_SONG
from src.vector_store import VectorStore


def corpus_size():
    """Chunks in the live corpus — derived, so a catalog edit is not a failure."""
    return len(build_corpus())


def song_card_count():
    return len([c for c in build_corpus() if c.kind == KIND_SONG])


@pytest.fixture
def offline_index(tmp_path):
    """
    A real index built with the offline embedder.

    Built at the embedder's default dimensionality on purpose: --check now
    verifies the embedder that produced the index, so a fixture at a reduced
    dimension would be reported unusable — correctly, since no real run could
    query it.
    """
    path = str(tmp_path / "index.jsonl")
    chunks = build_corpus()
    store = VectorStore.build(chunks, HashingEmbedder(), corpus_fingerprint(chunks))
    store.save(path)
    return path


def test_dry_run_reports_without_writing(tmp_path, capsys):
    out_path = str(tmp_path / "should-not-exist.jsonl")
    code = ingest_main(["--dry-run", "--out", out_path])
    out = capsys.readouterr().out

    assert code == 0
    assert f"song cards   : {song_card_count()}" in out
    assert "prose chunks :" in out
    assert "fingerprint" in out
    assert not (tmp_path / "should-not-exist.jsonl").exists()


def test_dry_run_makes_no_embedder_call(tmp_path, monkeypatch):
    """The point of --dry-run is tuning chunk sizes for free."""
    called = []
    monkeypatch.setattr(
        "src.ingest.make_embedder",
        lambda pref: called.append(pref) or pytest.fail("embedder built during --dry-run"),
    )
    assert ingest_main(["--dry-run", "--out", str(tmp_path / "x.jsonl")]) == 0
    assert called == []


def test_check_passes_on_a_matching_index(offline_index, capsys):
    code = ingest_main(["--check", "--out", offline_index])

    assert code == 0
    assert "OK:" in capsys.readouterr().out


def test_check_fails_when_the_corpus_changed(offline_index, tmp_path, capsys):
    """Append a song to the catalog and the committed index is stale."""
    catalog = tmp_path / "songs.csv"
    original = open("data/songs.csv", encoding="utf-8").read()
    catalog.write_text(
        original + "\n999,New Song,New Artist,pop,happy,0.5,120,0.5,0.5,0.5\n",
        encoding="utf-8",
    )

    code = ingest_main(["--check", "--out", offline_index, "--catalog", str(catalog)])
    err = capsys.readouterr().err

    assert code == 1
    assert "STALE" in err
    assert "src.ingest" in err


def test_check_fails_when_the_index_embedder_differs(tmp_path, capsys):
    """
    The failure that shipped: an index built offline, queried by Gemini.

    The corpus is untouched, so a fingerprint comparison alone calls this fresh.
    """
    path = str(tmp_path / "index.jsonl")
    chunks = build_corpus()
    VectorStore.build(chunks, HashingEmbedder(dimensions=64),
                      corpus_fingerprint(chunks)).save(path)

    code = ingest_main(["--check", "--out", path])
    err = capsys.readouterr().err

    assert code == 1
    assert "STALE" in err
    assert "offline-hashing@64" in err


def test_check_fails_when_the_index_is_missing(tmp_path, capsys):
    code = ingest_main(["--check", "--out", str(tmp_path / "absent.jsonl")])

    assert code == 1
    assert "STALE" in capsys.readouterr().err


def test_offline_ingest_writes_a_loadable_index(tmp_path):
    out_path = str(tmp_path / "offline.jsonl")
    code = ingest_main(["--embedder", "offline", "--out", out_path])

    assert code == 0
    store = VectorStore.load(out_path)
    assert len(store) == corpus_size()
    assert store.embedder_name.startswith("offline")

    # And it is immediately fresh against the corpus it was built from.
    assert store.check_fresh(corpus_fingerprint(build_corpus()))


def test_the_written_header_records_provenance(tmp_path):
    out_path = str(tmp_path / "offline.jsonl")
    ingest_main(["--embedder", "offline", "--out", out_path])

    with open(out_path, encoding="utf-8") as handle:
        header = json.loads(handle.readline())

    assert header["kind"] == "header"
    assert header["chunk_count"] == corpus_size()
    assert header["fingerprint"]
    assert header["dimensions"] > 0


def test_a_missing_catalog_fails_cleanly(tmp_path, capsys):
    code = ingest_main(["--dry-run", "--catalog", str(tmp_path / "nope.csv")])
    err = capsys.readouterr().err

    assert code == 1
    assert "Error:" in err
    assert "Traceback" not in err
