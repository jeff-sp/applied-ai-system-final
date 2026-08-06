"""
Splitting prose into overlapping, retrievable windows.

Pure text in, Chunks out — no file I/O, no network, no embedding. That makes
the boundary rules directly testable, which matters because chunking is the
one stage whose bugs are invisible at runtime: a bad split does not raise, it
just quietly retrieves the wrong passage.

The rule, in priority order:

1. Split on Markdown `##` headings. The heading becomes the chunk title (which
   feeds DOC_PREFIX) and a slug in the chunk id.
2. Inside a section, pack whole paragraphs up to CHUNK_TARGET_CHARS.
3. A paragraph too big to fit alone is split on sentence boundaries.
4. A sentence too big to fit alone is hard-split on whitespace. This never
   fires on hand-written prose; it exists so the function cannot emit a
   runaway chunk or loop.
5. Every chunk after the first carries the tail of its predecessor, snapped to
   a word boundary — but overlap NEVER crosses a `##` boundary. Carrying metal
   prose into a folk chunk would poison retrieval for both.
6. A final chunk shorter than CHUNK_MIN_CHARS is merged back into its
   predecessor rather than emitted, so no orphan fragment sits in the store
   scoring noise-high against short queries.

Invariant: no emitted chunk exceeds CHUNK_HARD_MAX_CHARS. Because overlap is
prepended *after* packing, packing runs against a ceiling reduced by the
overlap budget rather than against the hard max directly.
"""

import re
from typing import List, Optional

from src.config import (
    CHUNK_HARD_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
)
from src.models import KIND_PROSE, Chunk

# A Markdown `## Heading` line. `#` (title) and `###` (sub-detail) are left
# inside the section body on purpose: only `##` is a retrieval boundary.
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# Sentence boundary: punctuation followed by whitespace. Deliberately simple —
# it only has to fire on paragraphs already too long to keep whole.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def slugify(text: str) -> str:
    """Turns a heading into a chunk-id fragment: 'City Pop' -> 'city-pop'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def _split_sections(text: str) -> List[tuple]:
    """
    Splits a document into (heading, body) pairs on `##` lines.

    Any text before the first heading is returned under an empty heading, so a
    document with no `##` at all still produces exactly one section.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((match.group(1), body))
    return sections


def _hard_split(text: str, limit: int) -> List[str]:
    """Last-resort whitespace split for a single oversized sentence."""
    words = text.split()
    pieces: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_paragraph(paragraph: str, target: int, hard_max: int) -> List[str]:
    """Breaks an oversized paragraph into sentence-aligned pieces."""
    pieces: List[str] = []
    current = ""
    for sentence in SENTENCE_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > hard_max:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, hard_max))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack(body: str, target: int, hard_max: int) -> List[str]:
    """Greedily packs whole paragraphs into target-sized bodies."""
    bodies: List[str] = []
    current = ""

    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue

        if len(paragraph) > target:
            # Too big to pack; flush what we have and split it on sentences.
            if current:
                bodies.append(current)
                current = ""
            bodies.extend(_split_paragraph(paragraph, target, hard_max))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > target:
            bodies.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        bodies.append(current)
    return bodies


def _tail(text: str, overlap: int) -> str:
    """
    Returns the last `overlap` chars of text, snapped forward to a word start.

    Snapping forward (rather than back) guarantees the overlap never begins
    mid-word, which would hand the embedder a nonsense token.
    """
    if overlap <= 0 or len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_text(
    text: str,
    *,
    source: str,
    doc_slug: str,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
    hard_max_chars: int = CHUNK_HARD_MAX_CHARS,
    doc_title: Optional[str] = None,
) -> List[Chunk]:
    """
    Splits one document into overlapping Chunks.

    `doc_slug` namespaces the chunk ids: a genres.md section titled "City Pop"
    yields ids like `genres:city-pop:0`, `genres:city-pop:1`. Those ids are what
    the model is asked to cite, so they must be stable and human-readable.

    Deterministic: the same text always produces the same chunks, which is what
    lets the corpus fingerprint detect real drift rather than churn.
    """
    chunks: List[Chunk] = []

    # Overlap is prepended after packing, so pack against a ceiling reduced by
    # the overlap budget. That keeps the simple, testable invariant that no
    # emitted chunk ever exceeds hard_max_chars.
    pack_ceiling = max(1, hard_max_chars - overlap_chars)

    for heading, body in _split_sections(text):
        title = heading or doc_title or doc_slug
        section_slug = slugify(heading) if heading else "intro"
        bodies = _pack(body, min(target_chars, pack_ceiling), pack_ceiling)

        # Tail merge: fold a runt final piece back into its predecessor rather
        # than emitting a fragment that will score noise-high on short queries.
        if len(bodies) > 1 and len(bodies[-1]) < min_chars:
            bodies[-2] = f"{bodies[-2]}\n\n{bodies[-1]}"
            bodies.pop()

        for i, piece in enumerate(bodies):
            # Overlap is scoped to this section, so it can never leak across a
            # `##` boundary into unrelated subject matter.
            if i > 0:
                piece = f"{_tail(bodies[i - 1], overlap_chars)} {piece}".strip()

            chunks.append(Chunk(
                chunk_id=f"{doc_slug}:{section_slug}:{i}",
                kind=KIND_PROSE,
                title=title,
                text=piece,
                source=source,
                metadata={"section": heading, "position": i},
            ))

    return chunks
