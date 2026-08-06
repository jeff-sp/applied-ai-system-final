"""
Turning text into vectors — the "embed" stage of the RAG pipeline.

Two implementations behind one interface:

- `GeminiEmbedder` calls the Gemini embeddings API. Real semantic retrieval.
- `HashingEmbedder` computes deterministic vectors locally with no network and
  no dependencies. It is not a stub that returns constants — it genuinely
  retrieves, using signed feature hashing over word unigrams and bigrams — so
  the offline path exercises the same code as the online one.

The offline embedder is why this project still has a hermetic test suite and a
demo that runs without an API key.
"""

import abc
import math
import re
import time
from hashlib import blake2b
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.config import (
    DOC_PREFIX,
    EMBED_BATCH_SIZE,
    EMBED_DIMENSIONS,
    EMBED_MAX_RETRIES,
    EMBED_REQUESTS_PER_MINUTE,
    GEMINI_EMBED_MODEL,
    QUERY_PREFIX,
    RATE_LIMIT_WINDOW_SECONDS,
    api_key,
)
from src.logging_setup import get_logger
from src.models import Chunk

logger = get_logger("embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class EmbeddingError(Exception):
    """An embedding call failed or returned something unusable."""


class EmbeddingUnavailable(Exception):
    """The embedding backend cannot be constructed (usually a missing API key)."""


class RateLimited(EmbeddingError):
    """The API refused a call because the quota window is full."""


# Gemini reports how long to wait inside the 429 body. Preferring its number to
# our own backoff is the difference between waiting 37 seconds and waiting 64.
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def is_rate_limit(exc: BaseException) -> bool:
    """True when an SDK exception is a quota rejection rather than a real error."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    """
    How long to wait before retrying a rate-limited call.

    Uses the server's own retryDelay when it offers one, plus a second of slack
    so a clock skew of a few hundred ms does not walk straight into a second
    429. Falls back to exponential backoff when the body has no delay.
    """
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0
    return min(60.0, 2.0 * (2 ** attempt))


class RateLimiter:
    """
    Keeps a rolling window under a fixed item budget.

    Deliberately counts *items*, not calls: the free tier charges one unit per
    embedded input, so a 32-item batch spends 32 units of a 100/minute budget.
    Counting calls would report 9 requests for a 277-chunk ingest and sail
    straight into the limit — which is exactly what happened.

    `clock` and `sleep` are injectable so the tests can prove the pacing maths
    without spending three real minutes to do it.
    """

    def __init__(self, limit: int, window: float = RATE_LIMIT_WINDOW_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._spend: List[Tuple[float, int]] = []

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def _spent_in_window(self, now: float) -> int:
        cutoff = now - self.window
        self._spend = [(at, n) for at, n in self._spend if at > cutoff]
        return sum(n for _, n in self._spend)

    def reserve(self, count: int) -> float:
        """
        Blocks until `count` items fit in the window, then records them.

        Returns the seconds spent waiting, so the caller can tell the user why
        an ingest appears to be doing nothing.
        """
        if not self.enabled:
            return 0.0

        waited = 0.0
        while True:
            now = self._clock()
            spent = self._spent_in_window(now)

            # `not self._spend` lets a batch larger than the whole budget
            # through rather than blocking forever. It will 429 and be retried;
            # spinning here would just hang with no explanation.
            if spent + count <= self.limit or not self._spend:
                self._spend.append((now, count))
                return waited

            oldest = self._spend[0][0]
            pause = max(0.0, oldest + self.window - now)
            # A pause of zero means the window rolled between the check and
            # here; loop round and re-measure rather than announcing a wait
            # that is not happening.
            if pause > 0.5:
                logger.info("Rate limit: %d/%d used this window, waiting %.0fs "
                            "for room to embed %d more",
                            spent, self.limit, pause, count)
            self._sleep(pause)
            waited += pause

    def clear(self) -> None:
        """Forgets recorded spend — used after a 429 wait, when the window rolled."""
        self._spend = []


def l2_normalize(vector: List[float]) -> List[float]:
    """Scales a vector to unit length. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


class Embedder(abc.ABC):
    """
    Interface every embedding backend implements.

    `similarity_floor` and `similarity_ceiling` describe the range of cosine
    values this backend actually produces on real data. They are tuned policy,
    not math: cosine similarities do not span [0, 1] in practice, so
    models.rescale_similarity needs to know where a given backend's useful range
    begins and ends before folding similarity into a confidence score.
    """

    name: str = "embedder"
    dimensions: int = 0
    similarity_floor: float = 0.0
    similarity_ceiling: float = 1.0

    # Whether this backend's similarity scores are trustworthy enough to move
    # the user-facing confidence number. A real semantic model earns that; a
    # lexical fallback does not (see HashingEmbedder).
    contributes_confidence: bool = True

    @abc.abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embeds already-prepared texts, one vector per input, in order."""

    def doc_text(self, chunk: Chunk) -> str:
        """The exact string embedded for a stored chunk."""
        return DOC_PREFIX.format(title=chunk.title, content=chunk.text)

    def query_text(self, text: str) -> str:
        """The exact string embedded for a user query."""
        return QUERY_PREFIX.format(text=text)

    def embed_documents(self, chunks: Sequence[Chunk]) -> List[List[float]]:
        """Embeds chunks for storage, applying the document prefix."""
        texts = [self.doc_text(c) for c in chunks]
        vectors = self.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise EmbeddingError(
                f"{self.name}: expected {len(chunks)} vectors, got {len(vectors)}"
            )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embeds a user query, applying the query prefix."""
        vectors = self.embed_texts([self.query_text(text)])
        if len(vectors) != 1:
            raise EmbeddingError(
                f"{self.name}: expected exactly 1 query vector, got {len(vectors)}"
            )
        return vectors[0]


class GeminiEmbedder(Embedder):
    """
    Gemini embeddings via the google-genai SDK.

    gemini-embedding-2 has no `task_type` parameter (gemini-embedding-001 did).
    Asymmetric retrieval is expressed through the DOC_PREFIX / QUERY_PREFIX
    templates in config.py instead.
    """

    def __init__(self, key: Optional[str] = None, model: str = GEMINI_EMBED_MODEL,
                 dimensions: int = EMBED_DIMENSIONS,
                 limiter: Optional[RateLimiter] = None,
                 sleep: Callable[[float], None] = time.sleep):
        key = (key or api_key()).strip()
        if not key:
            raise EmbeddingUnavailable(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable Gemini embeddings."
            )

        # Imported lazily so the offline path never needs google-genai installed.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise EmbeddingUnavailable(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model
        self.dimensions = dimensions
        self.name = f"{model}@{dimensions}"
        self._limiter = limiter if limiter is not None else RateLimiter(
            EMBED_REQUESTS_PER_MINUTE)
        self._sleep = sleep

        # Real Gemini cosines on this corpus cluster in roughly this band.
        self.similarity_floor = 0.25
        self.similarity_ceiling = 0.85

    def _call(self, contents) -> object:
        """
        One embed_content call, paced ahead of time and retried on 429.

        Pacing and retrying are both here rather than one or the other because
        they cover different failures: the limiter keeps a long ingest from
        walking into the wall, the retry recovers when something outside this
        process (another run, a shared project) has already spent the budget.
        """
        types = self._types
        last: Optional[Exception] = None

        for attempt in range(EMBED_MAX_RETRIES + 1):
            self._limiter.reserve(len(contents))
            try:
                return self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions),
                )
            except Exception as exc:  # SDK raises many transport error types
                if not is_rate_limit(exc):
                    raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc

                last = exc
                if attempt == EMBED_MAX_RETRIES:
                    break

                delay = retry_delay_seconds(exc, attempt)
                logger.warning("Rate limited on attempt %d/%d; retrying in %.0fs",
                               attempt + 1, EMBED_MAX_RETRIES, delay)
                self._sleep(delay)
                # The window has rolled by the time the wait is over, so the
                # spend recorded before it no longer applies.
                self._limiter.clear()

        raise RateLimited(
            f"still rate limited after {EMBED_MAX_RETRIES} retries. The free tier "
            f"allows {EMBED_REQUESTS_PER_MINUTE} embedded items per minute; a "
            f"smaller corpus or a billed project would clear this. Last error: {last}"
        )

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        types = self._types

        # CRITICAL: passing a plain list of strings to `contents` returns ONE
        # aggregated embedding for the whole list, not one per item. Every chunk
        # would then share a vector and retrieval would silently degrade to
        # noise - no exception, no warning. Each item must be its own Content.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]

        result = self._call(contents)
        vectors = [list(e.values) for e in (result.embeddings or [])]

        # The footgun, caught in production and not just in tests.
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} vectors, got {len(vectors)} — each item "
                "must be wrapped in types.Content, or Gemini aggregates the batch "
                "into a single embedding"
            )
        return vectors

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[start:start + EMBED_BATCH_SIZE])
            logger.debug("Embedding batch %d-%d", start, start + len(batch))
            vectors.extend(self._embed_batch(batch))
            # At INFO so a paced run shows progress. An ingest that sits silent
            # for three minutes reads as a hang, and the natural response to a
            # hang is ctrl-C — which throws away everything already paid for.
            if len(texts) > EMBED_BATCH_SIZE:
                logger.info("Embedded %d/%d chunk(s)", len(vectors), len(texts))
        return vectors


class HashingEmbedder(Embedder):
    """
    Deterministic offline embedder: IDF-weighted signed feature hashing.

    Each unigram and bigram is hashed to a dimension and a sign, accumulated
    with a sublinear term-frequency weight, then L2-normalized. This is a real
    (if crude) lexical vector space, so tests and keyless demos exercise the
    same retrieval code path the Gemini backend does.

    Two details are load-bearing, both learned by watching it fail:

    - **IDF is not optional here.** All 203 song cards share the same template
      ("A {genre} song with a {mood} mood. Its energy is..."), so without
      document-frequency weighting the boilerplate dominates every vector and
      all songs look alike — retrieval degrades to noise. `fit()` computes
      document frequencies over the corpus; features absent from the corpus
      score zero, since a term nothing contains carries no retrieval signal.
    - **No prefixes.** DOC_PREFIX and QUERY_PREFIX are instruction text for
      gemini-embedding-2, which understands them semantically. A bag-of-words
      model does not: it just sees the literal tokens. Measured, feeding this
      embedder the query prefix made "task", "search" and "result" the three
      highest-weighted terms in every query — outweighing the actual subject —
      so "jazz music" retrieved the rock song "Search and Destroy". The prefix
      is a Gemini technique, not a universal one, so this backend skips it.
    """

    def __init__(self, dimensions: int = 2048):
        # 2048 measured best on the cost/quality curve for this corpus: top-1
        # genre+mood recall of 19/32 at ~31ms per query, against 14/32 at 256
        # dims and 21/32 at 4096 for twice the latency.
        self.dimensions = dimensions
        self.name = f"offline-hashing@{dimensions}"
        # Measured, not guessed: over the seven demo queries by top-20 song
        # hits, offline cosines run min -0.01 / median 0.06 / p90 0.10 / max
        # 0.16 — and the spread between queries is wide enough that no single
        # rescaling band fits them all.
        self.similarity_floor = 0.0
        self.similarity_ceiling = 0.12

        # So this backend orders results but does not move confidence. Lexical
        # overlap is a good enough signal to build a shortlist from and too
        # noisy to tell a user how sure the system is. Keeping it out means the
        # offline path reports exactly the confidence the weighted scorer
        # computes — identical to classic mode — instead of a number depressed
        # by an artefact of the fallback.
        self.contributes_confidence = False
        self._idf: Dict[str, float] = {}
        self._fitted = False

    def doc_text(self, chunk: Chunk) -> str:
        """Title plus body, without the Gemini instruction prefix."""
        return f"{chunk.title}. {chunk.text}"

    def query_text(self, text: str) -> str:
        """The query as written — see the class docstring on prefixes."""
        return text

    @staticmethod
    def _features(text: str) -> List[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        """
        Learns inverse document frequencies from the corpus.

        Called once at store-build time. Queries embedded afterwards reuse the
        same weights, which is what lets a rare word like "lofi" outweigh the
        template boilerplate every card shares.
        """
        document_count = len(texts)
        frequencies: Dict[str, int] = {}
        for text in texts:
            for feature in set(self._features(text)):
                frequencies[feature] = frequencies.get(feature, 0) + 1

        self._idf = {
            feature: math.log((1.0 + document_count) / (1.0 + count)) + 1.0
            for feature, count in frequencies.items()
        }
        self._fitted = True
        logger.debug("Fitted offline embedder on %d document(s), %d feature(s)",
                     document_count, len(self._idf))
        return self

    def _weight(self, feature: str) -> float:
        if not self._fitted:
            return 1.0
        # Unseen features score zero: a term no document contains cannot
        # discriminate between documents, and the query prefix tokens would
        # otherwise receive maximum weight for being unique.
        return self._idf.get(feature, 0.0)

    def _vector(self, text: str) -> List[float]:
        counts: Dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1

        vector = [0.0] * self.dimensions
        for feature, count in counts.items():
            weight = self._weight(feature)
            if weight == 0.0:
                continue
            # Sublinear term frequency: a word repeated ten times is more
            # important than one used once, but not ten times more important.
            magnitude = (1.0 + math.log(count)) * weight

            digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if (value >> 32) & 1 else -1.0
            vector[index] += sign * magnitude

        return l2_normalize(vector)

    def embed_documents(self, chunks: Sequence[Chunk]) -> List[List[float]]:
        """Fits on the corpus before embedding it, then defers to the base."""
        if not self._fitted:
            self.fit([self.doc_text(c) for c in chunks])
        return super().embed_documents(chunks)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]


def make_embedder(preference: str = "auto") -> Embedder:
    """
    Resolves an embedder from a CLI preference.

    'auto' uses Gemini when a key is present and the SDK imports, and falls back
    to the offline embedder otherwise — with a log line naming the reason, so a
    silent downgrade is never mistaken for a working Gemini run.
    """
    if preference == "offline":
        return HashingEmbedder()

    if preference == "gemini":
        return GeminiEmbedder()

    try:
        return GeminiEmbedder()
    except EmbeddingUnavailable as exc:
        logger.info("Falling back to the offline embedder: %s", exc)
        return HashingEmbedder()
