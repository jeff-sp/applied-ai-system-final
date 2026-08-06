"""
Tests for the embedding backends.

The centrepiece is the pair of tests around the google-genai batching footgun:
passing a list of strings to `embed_content` returns ONE aggregated vector for
the whole list rather than one per item. That failure is silent — every chunk
gets the same vector and retrieval degrades to noise with no exception — so it
is pinned from two directions: the argument type we send, and the length of
what we get back.
"""

import pytest

from src.embeddings import (
    Embedder,
    EmbeddingError,
    EmbeddingUnavailable,
    GeminiEmbedder,
    HashingEmbedder,
    RateLimited,
    RateLimiter,
    is_rate_limit,
    l2_normalize,
    make_embedder,
    retry_delay_seconds,
)
from src.models import KIND_PROSE, Chunk

CHUNKS = [
    Chunk(chunk_id="a", kind=KIND_PROSE, title="Lofi", text="mellow beats for studying", source="s"),
    Chunk(chunk_id="b", kind=KIND_PROSE, title="Metal", text="loud aggressive distorted guitars", source="s"),
    Chunk(chunk_id="c", kind=KIND_PROSE, title="Funk", text="groove bass swagger rhythm", source="s"),
]


# --- the offline embedder ---------------------------------------------------

def test_one_vector_per_chunk():
    vectors = HashingEmbedder().embed_documents(CHUNKS)

    assert len(vectors) == len(CHUNKS)
    assert len({tuple(v) for v in vectors}) == len(CHUNKS), "chunks share a vector"


def test_vectors_have_the_declared_dimensionality():
    embedder = HashingEmbedder(dimensions=128)
    vectors = embedder.embed_documents(CHUNKS)

    assert embedder.dimensions == 128
    assert all(len(v) == 128 for v in vectors)


def test_embedding_is_deterministic():
    a = HashingEmbedder().embed_documents(CHUNKS)
    b = HashingEmbedder().embed_documents(CHUNKS)

    assert a == b


def test_vectors_are_normalized():
    for vector in HashingEmbedder().embed_documents(CHUNKS):
        assert sum(x * x for x in vector) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalize_leaves_a_zero_vector_alone():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_offline_embedder_skips_the_gemini_prefixes():
    """
    Measured regression: feeding a bag-of-words model the query prefix made
    'task', 'search' and 'result' the highest-weighted terms in every query, so
    "jazz music" retrieved the rock song "Search and Destroy".
    """
    embedder = HashingEmbedder()

    assert embedder.query_text("jazz music") == "jazz music"
    assert "task:" not in embedder.doc_text(CHUNKS[0])


def test_offline_embedder_does_not_move_confidence():
    """A lexical fallback orders results; it is not calibrated to score them."""
    assert HashingEmbedder().contributes_confidence is False


def test_unseen_terms_carry_no_weight():
    """A word no document contains cannot discriminate between documents."""
    embedder = HashingEmbedder()
    embedder.embed_documents(CHUNKS)

    only_unknown = embedder.embed_query("zzzznonexistentword")
    assert all(x == 0.0 for x in only_unknown)


def test_offline_retrieval_actually_discriminates():
    """The fake must retrieve, not merely return deterministic noise."""
    from src.vector_store import VectorStore

    embedder = HashingEmbedder()
    store = VectorStore.build(CHUNKS, embedder, "fp")
    top = store.search(embedder.embed_query("loud distorted guitars"), k=1)

    assert top[0].chunk.chunk_id == "b"


# --- the Gemini embedder: the types.Content footgun -------------------------

class _RecordingClient:
    """Stands in for genai.Client, recording exactly what it was handed."""

    def __init__(self, aggregate=False):
        self.aggregate = aggregate
        self.calls = []
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append(contents)
        count = 1 if self.aggregate else len(contents)
        return type("R", (), {"embeddings": [
            type("E", (), {"values": [0.1, 0.2, 0.3]})() for _ in range(count)
        ]})()


def _embedder_with(client, limiter=None, sleep=None):
    """
    Builds a GeminiEmbedder without touching the network or the SDK.

    Pacing is off by default (limit 0) so the existing tests measure the call
    shape rather than the rate limiter; the pacing tests pass their own.
    """
    from google.genai import types

    embedder = GeminiEmbedder.__new__(GeminiEmbedder)
    embedder._types = types
    embedder.client = client
    embedder.model = "gemini-embedding-2"
    embedder.dimensions = 3
    embedder.name = "gemini-embedding-2@3"
    embedder.similarity_floor = 0.25
    embedder.similarity_ceiling = 0.85
    embedder.contributes_confidence = True
    embedder._limiter = limiter if limiter is not None else RateLimiter(0)
    embedder._sleep = sleep if sleep is not None else (lambda seconds: None)
    return embedder


class _FakeClock:
    """
    A clock that only moves when something sleeps.

    Lets the pacing tests assert real waits — three minutes of them — without
    the suite taking three minutes.
    """

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def test_each_item_is_wrapped_in_its_own_content_object():
    """
    The footgun, pinned by argument type.

    A plain list of strings is the shape that silently aggregates; we must send
    one types.Content per item.
    """
    from google.genai import types

    client = _RecordingClient()
    vectors = _embedder_with(client).embed_documents(CHUNKS)

    assert len(vectors) == len(CHUNKS)
    sent = client.calls[0]
    assert isinstance(sent, list)
    assert len(sent) == len(CHUNKS)
    assert all(isinstance(item, types.Content) for item in sent), \
        "passing List[str] makes Gemini return one aggregated embedding"


def test_an_aggregating_response_raises_rather_than_corrupting_the_index():
    """The footgun, pinned by response length — caught in production, not just here."""
    embedder = _embedder_with(_RecordingClient(aggregate=True))

    with pytest.raises(EmbeddingError, match="types.Content"):
        embedder.embed_documents(CHUNKS)


def test_gemini_applies_the_document_and_query_prefixes():
    embedder = _embedder_with(_RecordingClient())

    assert embedder.doc_text(CHUNKS[0]).startswith("title: Lofi | text:")
    assert embedder.query_text("jazz") == "task: search result | query: jazz"


def test_query_embedding_expects_exactly_one_vector():
    embedder = _embedder_with(_RecordingClient())
    embedder.client.aggregate = False

    vector = embedder.embed_query("something upbeat")
    assert vector == [0.1, 0.2, 0.3]


def test_sdk_errors_are_wrapped():
    class Boom:
        models = None

        def embed_content(self, **kwargs):
            raise RuntimeError("transport exploded")

    boom = Boom()
    boom.models = boom

    with pytest.raises(EmbeddingError, match="RuntimeError"):
        _embedder_with(boom).embed_documents(CHUNKS)


# --- rate limiting ----------------------------------------------------------
#
# The free tier charges one unit per embedded *item*, so a 277-chunk ingest is
# 277 units against a 100/minute budget. These pin the pacing that makes that a
# three-minute job instead of a 429 with 96 chunks paid for and discarded.

# The real 429 body, trimmed to what the parser reads.
QUOTA_ERROR = (
    "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
    "'retryDelay': '37s'}]}}"
)


def test_a_quota_error_is_recognised():
    assert is_rate_limit(RuntimeError(QUOTA_ERROR)) is True


def test_an_ordinary_error_is_not_mistaken_for_a_quota_error():
    assert is_rate_limit(RuntimeError("transport exploded")) is False


def test_the_servers_own_retry_delay_wins():
    """37s from the body beats guessing, plus a second of slack."""
    assert retry_delay_seconds(RuntimeError(QUOTA_ERROR), attempt=0) == 38.0


def test_backoff_is_used_when_the_body_carries_no_delay():
    plain = RuntimeError("429 RESOURCE_EXHAUSTED")
    delays = [retry_delay_seconds(plain, attempt=i) for i in range(4)]

    assert delays == sorted(delays), "backoff must not shrink"
    assert delays[0] == 2.0


def test_the_limiter_lets_work_under_the_budget_straight_through():
    clock = _FakeClock()
    limiter = RateLimiter(100, clock=clock, sleep=clock.sleep)

    assert limiter.reserve(32) == 0.0
    assert limiter.reserve(32) == 0.0
    assert limiter.reserve(32) == 0.0  # 96 of 100
    assert clock.slept == []


def test_the_limiter_waits_for_the_window_at_the_budget():
    """The exact shape of the failure: the 4th batch of 32 crosses 100."""
    clock = _FakeClock()
    limiter = RateLimiter(100, window=60.0, clock=clock, sleep=clock.sleep)

    for _ in range(3):
        limiter.reserve(32)
    waited = limiter.reserve(32)

    assert waited == 60.0, "must wait out the window rather than take a 429"
    assert clock.now == 60.0


def test_a_full_ingest_is_paced_rather_than_rejected():
    """277 items at 100/minute: finishes, and takes minutes not seconds."""
    clock = _FakeClock()
    limiter = RateLimiter(100, window=60.0, clock=clock, sleep=clock.sleep)

    for _ in range(0, 277, 32):
        limiter.reserve(32)

    assert clock.now >= 120.0, "277 items cannot lawfully clear in under 2 windows"


def test_a_batch_larger_than_the_whole_budget_does_not_hang():
    clock = _FakeClock()
    limiter = RateLimiter(10, clock=clock, sleep=clock.sleep)

    assert limiter.reserve(32) == 0.0  # let it through; the API will judge it


def test_a_rate_limited_call_is_retried_and_succeeds():
    class _FlakyClient:
        def __init__(self):
            self.attempts = 0
            self.models = self

        def embed_content(self, model, contents, config):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError(QUOTA_ERROR)
            return type("R", (), {"embeddings": [
                type("E", (), {"values": [0.1, 0.2, 0.3]})() for _ in contents
            ]})()

    clock = _FakeClock()
    client = _FlakyClient()
    vectors = _embedder_with(client, sleep=clock.sleep).embed_documents(CHUNKS)

    assert len(vectors) == len(CHUNKS)
    assert client.attempts == 2
    assert clock.slept == [38.0], "waited the delay the server asked for"


def test_persistent_rate_limiting_fails_with_an_actionable_message():
    class _AlwaysLimited:
        def __init__(self):
            self.attempts = 0
            self.models = self

        def embed_content(self, model, contents, config):
            self.attempts += 1
            raise RuntimeError(QUOTA_ERROR)

    client = _AlwaysLimited()
    embedder = _embedder_with(client, sleep=lambda s: None)

    with pytest.raises(RateLimited, match="per minute"):
        embedder.embed_documents(CHUNKS)

    assert client.attempts > 1, "gave up without retrying"


def test_a_non_quota_error_is_not_retried():
    """Retrying a malformed request just spends quota to fail again."""
    class _Broken:
        def __init__(self):
            self.attempts = 0
            self.models = self

        def embed_content(self, model, contents, config):
            self.attempts += 1
            raise RuntimeError("transport exploded")

    client = _Broken()
    with pytest.raises(EmbeddingError, match="RuntimeError"):
        _embedder_with(client).embed_documents(CHUNKS)

    assert client.attempts == 1


# --- backend selection ------------------------------------------------------

def test_missing_key_raises_unavailable():
    with pytest.raises(EmbeddingUnavailable, match="GEMINI_API_KEY"):
        GeminiEmbedder(key="")


def test_auto_falls_back_to_offline_without_a_key():
    """conftest strips GEMINI_API_KEY, so 'auto' must degrade rather than fail."""
    embedder = make_embedder("auto")

    assert isinstance(embedder, HashingEmbedder)
    assert isinstance(embedder, Embedder)


def test_explicit_offline_is_honoured():
    assert isinstance(make_embedder("offline"), HashingEmbedder)


def test_explicit_gemini_without_a_key_raises():
    with pytest.raises(EmbeddingUnavailable):
        make_embedder("gemini")
