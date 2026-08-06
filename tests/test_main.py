"""End-to-end tests for the CLI runner: the whole pipeline in one call."""

import re

import pytest

from src.corpus import build_corpus, corpus_fingerprint
from src.embeddings import HashingEmbedder
from src.guardrails import validate_user_prefs
from src.main import PROFILES, main
from src.retriever import DEFAULT_CATALOG, Retriever
from src.vector_store import VectorStore


def test_cli_runs_a_profile_end_to_end(capsys):
    # Pinned to classic mode: this test predates RAG and asserts the pre-RAG
    # output, which --mode classic must keep producing unchanged.
    exit_code = main(["--mode", "classic", "--profile", "lofi", "-k", "3"])
    out = capsys.readouterr().out

    assert exit_code == 0
    # Don't pin the catalog size - it grows. Just assert the load line printed.
    assert re.search(r"Loaded \d+ songs from data/songs\.csv", out)
    assert "CHILL LOFI" in out
    # the recommendation text is generated from retrieved evidence, not raw dumps
    assert "Strong match" in out
    assert "your favorite genre" in out


def test_cli_prints_a_reliability_report(capsys):
    main(["--mode", "classic", "--profile", "pop", "-k", "3"])
    out = capsys.readouterr().out

    assert "RUN REPORT" in out
    assert "average confidence" in out
    assert "grounding rate        : 1.00" in out
    assert "explanations withheld : 0" in out


# --- rag mode ---------------------------------------------------------------
#
# rag is the default mode. These run with no API key, so they exercise the
# offline embedder and the template generator - the same code path a grader
# gets on a fresh clone before setting GEMINI_API_KEY.

def test_rag_mode_is_the_default(capsys):
    assert main(["--profile", "lofi", "-k", "3"]) == 0
    out = capsys.readouterr().out

    assert "mode                  : rag" in out
    # The query, and what the system understood it to mean, are both shown.
    assert 'query: "mellow lofi beats to study to' in out
    assert "understood as: genre=lofi" in out


def test_rag_report_has_retrieval_and_grounding_counters(capsys):
    main(["--profile", "funk", "-k", "3"])
    out = capsys.readouterr().out

    assert "chunks retrieved      :" in out
    assert "avg chunk similarity  :" in out
    assert "fabricated citations  : 0" in out
    assert "substituted songs     : 0" in out
    assert "answers withheld      : 0" in out


def test_rag_mode_reranks_to_the_requested_genre(capsys):
    """The whole point of the hybrid: retrieval shortlists, the scorer reranks."""
    assert main(["--profile", "metal", "-k", "3"]) == 0
    out = capsys.readouterr().out

    section = out.split("ANGRY METAL")[1].split("RUN REPORT")[0]
    assert "understood as: genre=metal" in section
    assert "mood=angry" in section


def test_show_chunks_prints_retrieved_context(capsys):
    main(["--profile", "lofi", "-k", "2", "--show-chunks"])
    out = capsys.readouterr().out

    assert "retrieved context:" in out
    # Prose chunk ids are the citation handles the grounding check verifies.
    assert "[genres:" in out or "[listening_contexts:" in out or "[moods:" in out


def test_cli_runs_an_ad_hoc_query(capsys):
    exit_code = main(["--query", "something nostalgic for a late night drive", "-k", "3"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "YOUR QUERY" in out
    assert "understood as: genre=" in out


def test_cli_rejects_an_empty_query(capsys):
    exit_code = main(["--query", "   ", "-k", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0  # one bad query is skipped, the run still completes
    assert "Skipping" in captured.err
    assert "Traceback" not in captured.err


def test_generator_off_still_retrieves(capsys):
    assert main(["--profile", "jazz", "-k", "2", "--generator", "off"]) == 0
    out = capsys.readouterr().out

    assert "generator             : off" in out
    assert "chunks retrieved      :" in out


# --- index compatibility ----------------------------------------------------


def test_an_index_from_another_embedder_is_refused_not_used(tmp_path, capsys):
    """
    Regression: a committed index built by a different embedder was accepted on
    a fingerprint match alone, and its vectors were compared against this run's
    query vectors as if they shared a space. The run has to notice and rebuild.
    """
    index = tmp_path / "index.jsonl"
    chunks = build_corpus()
    VectorStore.build(chunks, HashingEmbedder(dimensions=64),
                      corpus_fingerprint(chunks)).save(str(index))

    exit_code = main(["--profile", "city pop", "-k", "3", "--index", str(index)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Warning" in captured.err
    assert "offline-hashing@64" in captured.err
    # Rebuilt in memory with this run's embedder rather than using the index.
    assert "built in memory" in captured.out
    assert "Plastic Love" in captured.out


# --- the demo profiles ------------------------------------------------------
#
# These are driven off PROFILES rather than a hardcoded list. Profiles get
# added over time, and a test that names three of them keeps passing while the
# rest go unexercised.

def test_cli_runs_all_profiles(capsys):
    assert main(["-k", "2"]) == 0
    out = capsys.readouterr().out

    for profile in PROFILES.values():
        assert profile["label"].upper() in out


@pytest.mark.parametrize("name", list(PROFILES))
def test_each_profile_runs_on_its_own(name, capsys):
    # Also pins profile keys that contain a space (e.g. "city pop") as usable
    # --profile values.
    exit_code = main(["--profile", name, "-k", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert PROFILES[name]["label"].upper() in captured.out
    assert "Skipping profile" not in captured.err


def test_run_report_counts_every_profile(capsys):
    # main() catches a ProfileError per profile and continues, so a broken
    # profile drops out of the run without changing the exit code. The report
    # is where that shows up.
    main(["-k", "2"])
    out = capsys.readouterr().out

    assert f"queries served        : {len(PROFILES)}" in out
    assert f"recommendations made  : {len(PROFILES) * 2}" in out


def test_every_demo_profile_is_valid_and_covered():
    """A typo in a profile must fail here, not silently rank on energy alone."""
    retriever = Retriever.from_csv(DEFAULT_CATALOG)

    for name, profile in PROFILES.items():
        prefs = validate_user_prefs(dict(profile["prefs"]))
        assert retriever.genre_index[prefs["favorite_genre"]] > 0, \
            f"{name}: genre {prefs['favorite_genre']!r} is absent from the catalog"
        assert retriever.mood_index[prefs["favorite_mood"]] > 0, \
            f"{name}: mood {prefs['favorite_mood']!r} is absent from the catalog"


# --- failure modes ----------------------------------------------------------

def test_cli_fails_cleanly_on_a_missing_catalog(capsys):
    exit_code = main(["--catalog", "data/nope.csv"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_a_non_positive_k(capsys):
    assert main(["-k", "0"]) == 1
    assert "positive integer" in capsys.readouterr().err
