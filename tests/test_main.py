"""End-to-end tests for the CLI runner: the whole pipeline in one call."""

import re

import pytest

from src.guardrails import validate_user_prefs
from src.main import PROFILES, main
from src.retriever import DEFAULT_CATALOG, Retriever


def test_cli_runs_a_profile_end_to_end(capsys):
    exit_code = main(["--profile", "lofi", "-k", "3"])
    out = capsys.readouterr().out

    assert exit_code == 0
    # Don't pin the catalog size - it grows. Just assert the load line printed.
    assert re.search(r"Loaded \d+ songs from data/songs\.csv", out)
    assert "CHILL LOFI" in out
    # the recommendation text is generated from retrieved evidence, not raw dumps
    assert "Strong match" in out
    assert "your favorite genre" in out


def test_cli_prints_a_reliability_report(capsys):
    main(["--profile", "pop", "-k", "3"])
    out = capsys.readouterr().out

    assert "RUN REPORT" in out
    assert "average confidence" in out
    assert "grounding rate        : 1.00" in out
    assert "explanations withheld : 0" in out


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
