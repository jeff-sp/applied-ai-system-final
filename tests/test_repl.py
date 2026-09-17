"""
End-to-end tests for the interactive session runner.

Driven the same way tests/test_main.py drives the CLI: call main() in-process
with a scripted input source and assert on captured output. conftest.py blocks
sockets and deletes the API key, so every test here runs the offline path.
"""

import io
import re

import pytest

from src.repl import main
from src.vector_store import VectorStore

OFFLINE = ["--embedder", "offline", "--generator", "template", "-k", "3"]
QUERY = "mellow lofi beats to study to, nothing too energetic"


def run(script, extra=None):
    """Runs a session over a scripted stdin. `script` is a list of typed lines."""
    argv = OFFLINE + (extra or [])
    return main(argv, stdin=io.StringIO("".join(f"{line}\n" for line in script)))


def test_a_question_then_quit_answers_and_exits_cleanly(capsys):
    exit_code = run([QUERY, "quit"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert re.search(r"Loaded \d+ songs from data/songs\.csv", out)
    assert "understood as: genre=lofi" in out
    assert "ANSWER (template)" in out
    assert "Session over" in out


def test_several_questions_are_answered_in_one_session(capsys):
    exit_code = run([QUERY, "loud intense rock for a hard workout", "quit"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "genre=lofi" in out
    assert "genre=rock" in out


def test_the_stack_is_built_once_no_matter_how_many_questions(monkeypatch):
    """
    The whole point of a session: pay for the vector store once.

    Guards the regression that would make this feature pointless while still
    passing every other test - rebuilding the store per turn is invisible
    except in the time it takes.
    """
    builds = []
    real_build = VectorStore.build

    def counting_build(chunks, embedder, fingerprint):
        builds.append(1)
        return real_build(chunks, embedder, fingerprint)

    monkeypatch.setattr(VectorStore, "build", staticmethod(counting_build))

    exit_code = run([QUERY, "loud intense rock for a hard workout", "angry metal", "quit"])

    assert exit_code == 0
    assert len(builds) == 1


def test_why_shows_the_evidence_behind_a_pick(capsys):
    exit_code = run([QUERY, "why 1", "quit"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "WHY #1" in out
    # per-signal points out of max, which the normal output never shows
    assert re.search(r"genre\s+1\.50 / 1\.50", out)
    assert re.search(r"score\s+\d\.\d\d / 4\.00\s+\d of 3 signals matched", out)
    assert "confidence :" in out


def test_why_reports_that_offline_similarity_did_not_move_confidence(capsys):
    """The hashing embedder sets contributes_confidence = False. Say so."""
    run([QUERY, "why 1", "quit"])
    out = capsys.readouterr().out

    assert "recorded, NOT folded into confidence" in out


def test_why_names_the_chunk_the_song_was_retrieved_from(capsys):
    run([QUERY, "why 1", "quit"])
    out = capsys.readouterr().out

    assert re.search(r"retrieved  : \[song:\d+\]", out)


def test_why_out_of_range_is_refused_and_the_session_survives(capsys):
    exit_code = run([QUERY, "why 9", "loud intense rock for a hard workout", "quit"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "There is no #9" in captured.err
    assert "genre=rock" in captured.out          # the session kept going
    assert "Traceback" not in captured.err


def test_why_before_any_question_is_refused(capsys):
    exit_code = run(["why 1", "quit"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ask for some music first" in captured.err
    assert "WHY #1" not in captured.out


def test_why_without_a_number_explains_the_usage(capsys):
    exit_code = run([QUERY, "why later", "quit"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "takes a rank number" in captured.err


def test_a_blank_line_just_reprompts(capsys):
    exit_code = run(["", "   ", "quit"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "YOUR QUERY" not in captured.out      # nothing was treated as a question
    # Not asserting stderr is empty: configure_logging's process-global _configured
    # flag means an earlier test's log level can still be in force here. What
    # matters is that a blank line produced no complaint of its own.
    assert "Skipped:" not in captured.err
    assert "Traceback" not in captured.err


def test_an_absurdly_long_question_is_refused_not_answered(capsys):
    exit_code = run(["x" * 600, "quit"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Skipped:" in captured.err
    assert "limit is 500" in captured.err
    assert "Traceback" not in captured.err


def test_a_bad_question_does_not_end_the_session(capsys):
    exit_code = run(["x" * 600, QUERY, "quit"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "understood as: genre=lofi" in out


def test_end_of_input_ends_the_session(capsys):
    """No quit command - stdin simply runs out, which is what Ctrl-D does."""
    exit_code = run([QUERY])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Session over" in out


def test_help_lists_the_commands(capsys):
    run(["help", "quit"])
    out = capsys.readouterr().out

    assert out.count("why <n>") >= 2             # the banner, then help again


@pytest.mark.parametrize("word", ["quit", "exit", "QUIT"])
def test_every_way_of_leaving_works(word, capsys):
    assert run([word]) == 0
    assert "Session over" in capsys.readouterr().out


def test_a_nonpositive_k_is_rejected_before_any_work(capsys):
    exit_code = main(["-k", "0"], stdin=io.StringIO(""))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "-k must be a positive integer" in captured.err
    assert "Loaded" not in captured.out          # refused before loading the catalog


def test_a_missing_catalog_fails_with_one_line_not_a_traceback(capsys):
    exit_code = main(
        OFFLINE + ["--catalog", "data/nope.csv"], stdin=io.StringIO("")
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_show_chunks_reveals_the_retrieved_context(capsys):
    run([QUERY, "quit"], extra=["--show-chunks"])
    out = capsys.readouterr().out

    assert "retrieved context:" in out
