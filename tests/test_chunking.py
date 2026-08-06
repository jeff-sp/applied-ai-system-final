"""
Tests for the chunker.

Chunking is the stage whose bugs are invisible at runtime: a bad split does not
raise, it just retrieves the wrong passage. So the boundary rules are pinned
here rather than trusted.
"""

import glob

import pytest

from src.chunking import chunk_text, slugify
from src.config import CHUNK_HARD_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS
from src.models import KIND_PROSE

PARAGRAPH = (
    "Ska is the fast bright ancestor of reggae with the same offbeat guitar chop "
    "played at double the speed and horn sections carrying the melody. "
)


def make_doc(sections):
    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)


def test_slugify_makes_readable_chunk_ids():
    assert slugify("City Pop") == "city-pop"
    assert slugify("R&B and Soul") == "r-b-and-soul"
    assert slugify("!!!") == "section"


def test_sections_become_separate_chunks():
    doc = make_doc([("Ska", PARAGRAPH * 3), ("Funk", PARAGRAPH * 3)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="genres")

    slugs = {c.chunk_id.split(":")[1] for c in chunks}
    assert slugs == {"ska", "funk"}
    assert all(c.kind == KIND_PROSE for c in chunks)
    assert all(c.source == "kb.md" for c in chunks)


def test_heading_becomes_the_chunk_title():
    doc = make_doc([("City Pop", PARAGRAPH * 4)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="genres")

    assert all(c.title == "City Pop" for c in chunks)
    assert chunks[0].chunk_id == "genres:city-pop:0"


def test_consecutive_chunks_overlap():
    doc = make_doc([("Ska", PARAGRAPH * 12)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="genres")

    assert len(chunks) > 1, "test needs a section long enough to split"
    for previous, current in zip(chunks, chunks[1:]):
        # The overlap is a suffix of the previous chunk's own text, so some
        # opening words of `current` must appear in `previous`.
        opening = " ".join(current.text.split()[:4])
        assert opening in previous.text


def test_overlap_is_word_aligned():
    """A mid-word overlap would hand the embedder a nonsense token."""
    doc = make_doc([("Ska", PARAGRAPH * 12)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="genres")

    for chunk in chunks[1:]:
        first_word = chunk.text.split()[0]
        assert first_word.isalnum() or first_word.rstrip(".,").isalnum()


def test_overlap_never_crosses_a_section_boundary():
    """Carrying metal prose into a folk chunk would poison retrieval for both."""
    doc = make_doc([
        ("Metal", "MetalWordUnique " * 120),
        ("Folk", "FolkWordUnique " * 120),
    ])
    chunks = chunk_text(doc, source="kb.md", doc_slug="genres")

    for chunk in chunks:
        section = chunk.chunk_id.split(":")[1]
        if section == "folk":
            assert "MetalWordUnique" not in chunk.text
        if section == "metal":
            assert "FolkWordUnique" not in chunk.text


def test_no_chunk_exceeds_the_hard_maximum():
    """Overlap is prepended after packing, so the ceiling must account for it."""
    doc = make_doc([("Big", "word " * 2000), ("Normal", PARAGRAPH * 6)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="t")

    assert chunks
    assert all(len(c.text) <= CHUNK_HARD_MAX_CHARS for c in chunks)


def test_an_oversized_paragraph_falls_back_to_sentence_splitting():
    giant = PARAGRAPH * 20  # one paragraph, many sentences, no blank lines
    chunks = chunk_text(make_doc([("Ska", giant)]), source="kb.md", doc_slug="t")

    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_HARD_MAX_CHARS for c in chunks)


def test_a_sentence_with_no_punctuation_is_hard_split():
    runaway = "word " * 3000  # no sentence boundaries at all
    chunks = chunk_text(make_doc([("Big", runaway)]), source="kb.md", doc_slug="t")

    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_HARD_MAX_CHARS for c in chunks)


def test_a_short_tail_is_merged_not_orphaned():
    """An orphan fragment scores noise-high against short queries."""
    doc = make_doc([("Ska", PARAGRAPH * 9 + "\n\nShort tail.")])
    chunks = chunk_text(doc, source="kb.md", doc_slug="t")

    assert chunks[-1].text.endswith("Short tail.")
    assert len(chunks[-1].text) >= CHUNK_MIN_CHARS


def test_a_document_with_no_headings_still_chunks():
    chunks = chunk_text(PARAGRAPH * 4, source="kb.md", doc_slug="notes")

    assert chunks
    assert chunks[0].chunk_id.startswith("notes:intro:")


def test_chunking_is_deterministic():
    """The corpus fingerprint depends on this: churn would look like drift."""
    doc = make_doc([("Ska", PARAGRAPH * 10), ("Funk", PARAGRAPH * 10)])
    first = chunk_text(doc, source="kb.md", doc_slug="t")
    second = chunk_text(doc, source="kb.md", doc_slug="t")

    assert [(c.chunk_id, c.text) for c in first] == [(c.chunk_id, c.text) for c in second]


def test_every_chunk_is_non_empty():
    doc = make_doc([("A", PARAGRAPH), ("B", ""), ("C", PARAGRAPH * 5)])
    chunks = chunk_text(doc, source="kb.md", doc_slug="t")

    assert all(c.text.strip() for c in chunks)


@pytest.mark.parametrize("path", sorted(glob.glob("docs/kb/*.md")))
def test_the_real_knowledge_base_chunks_cleanly(path):
    """The shipped prose must survive the shipped chunker."""
    with open(path, encoding="utf-8") as handle:
        chunks = chunk_text(handle.read(), source=path, doc_slug="kb")

    assert chunks, f"{path} produced no chunks"
    assert all(len(c.text) <= CHUNK_HARD_MAX_CHARS for c in chunks)
    assert all(c.text.strip() for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks), "duplicate chunk ids"
