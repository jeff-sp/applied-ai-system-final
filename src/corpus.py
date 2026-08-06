"""
Building the retrievable corpus: song cards plus the prose knowledge base.

Two very different sources become one uniform list of Chunks:

- `data/songs.csv` -> one "song card" per row. A card is a couple of sentences
  of natural language rendered from the structured fields, because embeddings
  match "high energy" far better than they match "0.82". The raw numbers are
  kept verbatim alongside the words, since guardrails.check_generated_answer
  verifies generated claims against exactly those strings.
- `docs/kb/*.md` -> overlapping windows via chunking.chunk_text.

Also home to `corpus_fingerprint`, which is how the system notices that the
committed index no longer matches the data it was built from.
"""

import glob
import hashlib
import os
from typing import Any, Dict, List

from src.chunking import chunk_text
from src.config import (
    DEFAULT_CATALOG,
    DEFAULT_KB_DIR,
    PREFIX_SCHEME_VERSION,
)
from src.logging_setup import get_logger
from src.models import KIND_SONG, Chunk
from src.recommender import load_songs

logger = get_logger("corpus")

# Descriptive bands for the 0.0-1.0 audio features. These are the ONLY place a
# number becomes a word the model might repeat back, so they are tested.
_LEVELS = (
    (0.20, "very low"),
    (0.40, "low"),
    (0.60, "moderate"),
    (0.80, "high"),
    (1.01, "very high"),
)

# Tempo bands chosen to agree with the prose in docs/kb/genres.md, which
# describes 100-130 BPM as the moderate, mid-tempo range.
_TEMPO_BANDS = (
    (90, "slow"),
    (120, "moderate"),
    (150, "fast"),
    (10_000, "very fast"),
)


def level(value: float) -> str:
    """Maps a 0.0-1.0 feature onto a descriptive band ('high', 'very low')."""
    for threshold, word in _LEVELS:
        if value < threshold:
            return word
    return _LEVELS[-1][1]


def tempo_word(bpm: float) -> str:
    """Maps a tempo onto a descriptive band ('slow', 'fast')."""
    for threshold, word in _TEMPO_BANDS:
        if bpm < threshold:
            return word
    return _TEMPO_BANDS[-1][1]


def render_song_card(song: Dict[str, Any]) -> Chunk:
    """
    Renders one catalog row as a retrievable natural-language card.

    Never split: a card is roughly 50 tokens, far below any chunking threshold.
    Both the descriptive word and the raw number appear for every feature — the
    word is what the embedder matches on, the number is what the grounding check
    verifies against.
    """
    text = (
        f"A {song['genre']} song with a {song['mood']} mood. "
        f"Its energy is {level(song['energy'])} ({song['energy']:.2f}) and the tempo is "
        f"{tempo_word(song['tempo_bpm'])} at {song['tempo_bpm']} BPM. "
        f"Positivity is {level(song['valence'])} (valence {song['valence']:.2f}), "
        f"danceability is {level(song['danceability'])} ({song['danceability']:.2f}), "
        f"and acousticness is {level(song['acousticness'])} ({song['acousticness']:.2f})."
    )
    return Chunk(
        chunk_id=f"song:{song['id']}",
        kind=KIND_SONG,
        title=f"{song['title']} — {song['artist']}",
        text=text,
        source=DEFAULT_CATALOG,
        metadata={
            "song_id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
        },
    )


def build_song_chunks(songs: List[Dict[str, Any]]) -> List[Chunk]:
    """Renders every catalog row as a song card."""
    return [render_song_card(song) for song in songs]


def build_prose_chunks(kb_dir: str = DEFAULT_KB_DIR) -> List[Chunk]:
    """
    Chunks every Markdown file in the knowledge base directory.

    A missing or empty directory is a warning rather than an error: the system
    still works with song cards alone, it just loses the background prose that
    bridges everyday phrasing to catalog labels.
    """
    chunks: List[Chunk] = []
    paths = sorted(glob.glob(os.path.join(kb_dir, "*.md")))

    if not paths:
        logger.warning(
            "No knowledge base files found in %s - retrieval will use song cards only",
            kb_dir,
        )
        return chunks

    for path in paths:
        slug = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        file_chunks = chunk_text(text, source=path, doc_slug=slug)
        logger.debug("Chunked %s into %d chunk(s)", path, len(file_chunks))
        chunks.extend(file_chunks)

    return chunks


def build_corpus(
    catalog_path: str = DEFAULT_CATALOG,
    kb_dir: str = DEFAULT_KB_DIR,
) -> List[Chunk]:
    """
    Builds the full retrievable corpus: song cards followed by prose chunks.

    Reuses `load_songs`, so every catalog guardrail (missing columns, malformed
    rows, out-of-range values) applies here unchanged.
    """
    songs = load_songs(catalog_path)
    song_chunks = build_song_chunks(songs)
    prose_chunks = build_prose_chunks(kb_dir)

    logger.info(
        "Corpus: %d song card(s) + %d prose chunk(s) = %d total",
        len(song_chunks), len(prose_chunks), len(song_chunks) + len(prose_chunks),
    )
    return song_chunks + prose_chunks


def corpus_fingerprint(chunks: List[Chunk]) -> str:
    """
    A stable hash over the corpus, used to detect a stale index.

    Covers chunk ids and text, plus PREFIX_SCHEME_VERSION — changing the
    document or query prefix does not alter any chunk's text, but it does change
    what gets embedded, and that must invalidate the index too. Sorted so
    ordering changes alone do not register as drift.
    """
    digest = hashlib.sha256()
    digest.update(f"prefix-scheme:{PREFIX_SCHEME_VERSION}\n".encode("utf-8"))
    for chunk_id, text in sorted((c.chunk_id, c.text) for c in chunks):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
