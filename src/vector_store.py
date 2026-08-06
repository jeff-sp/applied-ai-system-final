"""
The vector store — the "store" and "top-k nearest" stages of the pipeline.

Deliberately a flat list scanned in pure Python rather than a vector database.
At 277 chunks by 768 dimensions a query is about 210,000 multiply-adds, which
measures at roughly 20-30ms — invisible next to a ~300ms embedding round trip.
Adding numpy or FAISS would buy nothing measurable here and would cost the
property that makes this repo useful to read: the retrieval math is right there
in the source.

Persistence is JSONL: one header record, then one record per chunk. The header
carries the corpus fingerprint, which is how a stale index gets caught.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.embeddings import Embedder
from src.logging_setup import get_logger
from src.models import KIND_PROSE, KIND_SONG, Chunk, RetrievedChunk

logger = get_logger("vector_store")


class IndexUnavailableError(Exception):
    """The index file is missing, malformed, or unreadable."""


class IndexStaleError(Exception):
    """The index no longer matches the corpus it was built from."""


class DimensionMismatchError(ValueError):
    """Vectors of different lengths were compared — different embedders."""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two vectors.

    Computed explicitly rather than assuming unit vectors. Gemini normalizes its
    output and HashingEmbedder normalizes too, so a plain dot product would be
    equivalent today — but a future embedder that does not normalize would then
    silently produce garbage rankings instead of correct ones.

    Unequal lengths raise rather than zip-truncate. Truncation is the failure
    that hid a Gemini query vector being scored against an offline-built index:
    every number stayed in a plausible range, so nothing looked wrong except the
    recommendations.
    """
    if len(a) != len(b):
        raise DimensionMismatchError(
            f"cannot compare a {len(a)}-dim vector with a {len(b)}-dim one; "
            "the query and the index came from different embedders"
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass
class StoredVector:
    """One chunk and its embedding, with the norm precomputed at load time."""
    chunk: Chunk
    vector: List[float]
    norm: float


class VectorStore:
    """Holds embedded chunks and answers nearest-neighbour queries over them."""

    def __init__(self, entries: List[StoredVector], embedder_name: str,
                 dimensions: int, fingerprint: str):
        self.entries = entries
        self.embedder_name = embedder_name
        self.dimensions = dimensions
        self.fingerprint = fingerprint

    def __len__(self) -> int:
        return len(self.entries)

    def counts(self) -> Dict[str, int]:
        """How many chunks of each kind the store holds."""
        counts: Dict[str, int] = {KIND_SONG: 0, KIND_PROSE: 0}
        for entry in self.entries:
            counts[entry.chunk.kind] = counts.get(entry.chunk.kind, 0) + 1
        return counts

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: Embedder, fingerprint: str) -> "VectorStore":
        """Embeds every chunk and returns a ready-to-query store."""
        logger.info("Embedding %d chunk(s) with %s", len(chunks), embedder.name)
        vectors = embedder.embed_documents(chunks)
        entries = [
            StoredVector(chunk=chunk, vector=vector,
                         norm=math.sqrt(sum(x * x for x in vector)))
            for chunk, vector in zip(chunks, vectors)
        ]
        return cls(entries, embedder.name, embedder.dimensions, fingerprint)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Writes the store as JSONL. Only src/ingest.py should call this."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as handle:
            header = {
                "kind": "header",
                "embedder": self.embedder_name,
                "dimensions": self.dimensions,
                "fingerprint": self.fingerprint,
                "chunk_count": len(self.entries),
            }
            handle.write(json.dumps(header) + "\n")

            for entry in self.entries:
                record = {
                    "chunk_id": entry.chunk.chunk_id,
                    "kind": entry.chunk.kind,
                    "title": entry.chunk.title,
                    "text": entry.chunk.text,
                    "source": entry.chunk.source,
                    "metadata": entry.chunk.metadata,
                    # 6dp keeps the file near 2MB with no measurable ranking
                    # difference against full float64 precision.
                    "vector": [round(x, 6) for x in entry.vector],
                }
                handle.write(json.dumps(record) + "\n")

        logger.info("Wrote %s (%d vectors, %d dims)", path, len(self.entries), self.dimensions)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Reads a JSONL index written by `save`."""
        if not os.path.exists(path):
            raise IndexUnavailableError(
                f"No index at {path}. Build one with: python3 -m src.ingest"
            )

        entries: List[StoredVector] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                header_line = handle.readline()
                if not header_line.strip():
                    raise IndexUnavailableError(f"{path} is empty")
                header = json.loads(header_line)
                if header.get("kind") != "header":
                    raise IndexUnavailableError(f"{path} does not start with a header record")

                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    vector = [float(x) for x in record["vector"]]
                    chunk = Chunk(
                        chunk_id=record["chunk_id"],
                        kind=record["kind"],
                        title=record["title"],
                        text=record["text"],
                        source=record["source"],
                        metadata=record.get("metadata", {}),
                    )
                    entries.append(StoredVector(
                        chunk=chunk, vector=vector,
                        norm=math.sqrt(sum(x * x for x in vector)),
                    ))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IndexUnavailableError(f"{path} is malformed: {exc}") from exc

        if not entries:
            raise IndexUnavailableError(f"{path} contains a header but no vectors")

        return cls(
            entries=entries,
            embedder_name=header.get("embedder", "unknown"),
            dimensions=int(header.get("dimensions", len(entries[0].vector))),
            fingerprint=header.get("fingerprint", ""),
        )

    def check_fresh(self, fingerprint: str,
                    embedder: Optional[Embedder] = None) -> bool:
        """True when the index matches this corpus and this embedder."""
        return self.mismatch_reason(fingerprint, embedder) is None

    def mismatch_reason(self, fingerprint: str,
                        embedder: Optional[Embedder] = None) -> Optional[str]:
        """
        Why this index cannot be used, or None when it can be.

        The corpus fingerprint alone is not enough. An index carries vectors from
        one specific embedder, and a fingerprint match says nothing about which
        one — so an offline-built index passed as "fresh" to a Gemini run and the
        two vector spaces were compared as if they were the same space.
        """
        if not self.fingerprint:
            return "the index carries no corpus fingerprint"
        if self.fingerprint != fingerprint:
            return (
                f"it was built from a different corpus "
                f"(index {self.fingerprint[:16]}..., corpus {fingerprint[:16]}...)"
            )
        if embedder is None:
            return None
        if self.embedder_name != embedder.name:
            return (
                f"it was built with {self.embedder_name}, "
                f"but this run embeds queries with {embedder.name}"
            )
        if self.dimensions != embedder.dimensions:
            return (
                f"it stores {self.dimensions}-dim vectors, "
                f"but {embedder.name} produces {embedder.dimensions}-dim ones"
            )
        return None

    # -- search --------------------------------------------------------------

    def search(self, query_vector: Sequence[float], k: int = 5,
               kind: Optional[str] = None) -> List[RetrievedChunk]:
        """
        Returns the k most similar chunks, highest similarity first.

        Ties break on chunk_id so ordering is reproducible across runs — without
        that, two equally-scoring chunks could swap places between runs and make
        recorded demo output non-reproducible for reasons unrelated to the model.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k == 0:
            return []

        scored: List[Tuple[float, str, RetrievedChunk]] = []
        for entry in self.entries:
            if kind is not None and entry.chunk.kind != kind:
                continue
            similarity = cosine_similarity(query_vector, entry.vector)
            scored.append((
                similarity,
                entry.chunk.chunk_id,
                RetrievedChunk(chunk=entry.chunk, similarity=similarity),
            ))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:k]]

    def search_stratified(self, query_vector: Sequence[float], k_songs: int = 8,
                          k_prose: int = 3) -> Tuple[List[RetrievedChunk], List[RetrievedChunk]]:
        """
        Retrieves songs and prose separately.

        A single top-k over the combined store would let 203 song cards crowd
        out the background prose entirely (or, on a conceptual query, the
        reverse). Retrieving each kind to its own depth guarantees the prompt
        gets both the candidates and the context that explains them.
        """
        return (
            self.search(query_vector, k=k_songs, kind=KIND_SONG),
            self.search(query_vector, k=k_prose, kind=KIND_PROSE),
        )
