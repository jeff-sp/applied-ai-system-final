"""
Makes the repo root importable so `from src.recommender import ...` works
whether tests are run as `pytest` or `python3 -m pytest`.

Also enforces that the suite is hermetic. Since this project gained a network
dependency, "the tests pass" only means something if the tests could not have
reached the API. The two autouse fixtures below make that structural rather
than aspirational: a passing run is now proof that nothing was sent anywhere
and nothing was billed.
"""

import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """
    Fails any test that opens a socket.

    Without this, a regression that made the offline path call Gemini would
    still pass on a developer machine with a key set, and only fail in CI - or
    worse, quietly cost money on every test run.
    """
    def blocked(*args, **kwargs):
        raise RuntimeError(
            "network access is not allowed in tests - the offline embedder and "
            "template generator exist so the suite never calls the API"
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """
    Removes GEMINI_API_KEY for the duration of every test.

    A developer with a key in their environment must get exactly the same
    results as CI, which has none. Tests that need a Gemini backend construct
    it explicitly with a dummy key and a stub client.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
