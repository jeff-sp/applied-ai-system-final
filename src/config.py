"""
Every tunable in the retrieval-augmented pipeline, in one place.

This module imports no project code (same rule as models.py), so it can never
take part in an import cycle and can be read on its own to understand how the
system is configured.

It is also where .env is loaded, precisely because everything imports it. When
only main.py called load_dotenv(), `python3 -m src.ingest` ran without the key,
fell back to the offline embedder, and wrote offline vectors to the Gemini index
path — which main.py then queried with Gemini vectors. Loading here means every
entry point resolves the key the same way.

The values here are *policy*, not math. Where a number was chosen by judgement
rather than derived, the comment says so and says why — the same honesty the
README applies to the scoring weights and confidence bands.
"""

import os

# Optional: loads GEMINI_API_KEY from .env when python-dotenv is installed.
# Guarded so the stdlib-only offline path never hard-depends on it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - exercised only without the dependency
    pass

# --- Paths -----------------------------------------------------------------

DEFAULT_CATALOG = "data/songs.csv"
DEFAULT_KB_DIR = "docs/kb"
DEFAULT_INDEX_PATH = "data/index/gemini_index.jsonl"
OFFLINE_INDEX_PATH = "data/index/offline_index.jsonl"

# --- Gemini models ---------------------------------------------------------
#
# Both are single constants on purpose: model ids move, and when they do this
# is the only place to edit. `-latest` aliases track upstream automatically,
# which is convenient but means recorded outputs are dated (see model_card.md).

GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_GENERATION_MODEL = "gemini-flash-lite-latest"

# 768 is the smallest of Gemini's three recommended dimensionalities. At 275
# chunks the retrieval quality difference against 1536/3072 is not measurable
# here, and 768 keeps the committed index near 2MB instead of 8MB.
EMBED_DIMENSIONS = 768

# Chunks per embed_content call during ingest. A latency knob only — measured
# against the free tier, quota is charged per *item*, not per HTTP call, so
# batching buys round trips and nothing else. Do not raise this expecting to fit
# more work under the rate limit.
EMBED_BATCH_SIZE = 32

# --- Rate limits -----------------------------------------------------------
#
# The free tier allows 100 embed_content items per minute per model
# (quotaId EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier).
# A 277-chunk corpus is therefore a ~3 minute job that *must* be paced: without
# this, ingest sprinted through 96 items in two seconds, took a 429, and threw
# away everything it had already paid for.
#
# Set to 0 to disable pacing (appropriate on a billed project with a higher cap).
EMBED_REQUESTS_PER_MINUTE = 100
RATE_LIMIT_WINDOW_SECONDS = 60.0

# How many times a rate-limited batch is retried before giving up. Gemini
# reports its own retryDelay, which is honoured in preference to backoff.
EMBED_MAX_RETRIES = 5

# --- Retrieval prefixes ----------------------------------------------------
#
# gemini-embedding-2 dropped the `task_type` parameter that gemini-embedding-001
# had. Asymmetric retrieval (queries and documents embedded differently) is now
# expressed as a prompt prefix instead. Getting these wrong does not raise - it
# silently degrades ranking - so PREFIX_SCHEME_VERSION is folded into the corpus
# fingerprint to force a re-ingest whenever they change.

DOC_PREFIX = "title: {title} | text: {content}"
QUERY_PREFIX = "task: search result | query: {text}"
PREFIX_SCHEME_VERSION = 1

# --- Chunking --------------------------------------------------------------
#
# 900 chars is roughly 225 tokens: far below the 8,192-token per-item input cap,
# and small enough that injecting three prose chunks costs only ~700 tokens of
# prompt. 150 chars of overlap is about one sentence - enough to carry a thought
# across a seam without duplicating a whole paragraph.

CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_CHARS = 200
CHUNK_HARD_MAX_CHARS = 1400

# --- Ranking ---------------------------------------------------------------
#
# Embeddings retrieve a shortlist; the weighted scorer in recommender.py
# reranks it. RERANK_CANDIDATES is how wide that shortlist is: large enough
# that the scorer has real choices, small enough that a genre miss in
# retrieval cannot be rescued by brute force.

RERANK_CANDIDATES = 20

# Weight on semantic similarity in the blended sort key. 0.6 leans on the
# embedding (it is what understood the natural-language request) while leaving
# the hand-tuned scorer enough authority to honour a numeric energy target.
BLEND_ALPHA = 0.6

# How many prose chunks are injected into the generation prompt.
DEFAULT_CONTEXT_K = 3

# Share of the confidence score that comes from retrieval similarity once a
# semantic path is in play. The remaining 0.7 is the existing score/coverage
# blend, so confidence stays mostly a statement about the scorer.
CONFIDENCE_SIMILARITY_WEIGHT = 0.3

# --- Grounding -------------------------------------------------------------

# Below this share of cited sentences, a generated answer is withheld entirely.
# 0.6 tolerates a connective sentence or two while catching an answer that
# narrates freely and sprinkles one decorative citation.
MIN_CITED_RATIO = 0.6

# Longest query we will embed. Guards against a pasted document arriving as a
# "query" and blowing past the model's input limit.
MAX_QUERY_CHARS = 500


def api_key() -> str:
    """Returns GEMINI_API_KEY from the environment, or an empty string."""
    return os.getenv("GEMINI_API_KEY", "").strip()
