import numpy as np
import pytest

from modules.overview import build_overview, page_label, select_representative
from modules.pdf_reader import (
    find_page_references,
    pages_to_text,
    split_pages,
)


# ---------- helpers ----------

def make_pages(count, words_per_page=300):
    """A fake book: page N contains its own marker word 'marker<N>'."""
    pages = []
    for n in range(1, count + 1):
        body = " ".join(f"w{n}_{i}" for i in range(words_per_page))
        pages.append((n, f"marker{n} {body}"))
    return pages


# ---------- split_pages ----------

def test_split_pages_empty():
    assert split_pages([]) == ([], [])
    assert split_pages([(1, "   "), (2, "")]) == ([], [])


def test_split_pages_single_page():
    chunks, pages = split_pages([(7, "hello world")])
    assert chunks == ["hello world"]
    assert pages == [(7, 7)]


def test_split_pages_keeps_real_page_numbers_when_pages_are_skipped():
    chunks, pages = split_pages([(1, "first page"), (4, "fourth page")])
    assert pages[0][0] == 1
    assert pages[-1][1] == 4


def test_split_pages_maps_chunks_to_the_right_pages():
    book = make_pages(60)
    chunks, chunk_pages = split_pages(book)

    assert len(chunks) == len(chunk_pages) > 60

    for chunk, (first, last) in zip(chunks, chunk_pages):
        assert first <= last
        # The first marker word in a chunk that starts on its page's
        # beginning identifies the page exactly.
        if chunk.startswith("marker"):
            assert chunk.split()[0] == f"marker{first}"
        # Every word w<N>_<i> in a chunk must belong to a page in range.
        for word in chunk.split():
            if word.startswith("w"):
                page = int(word[1:].split("_")[0])
                assert first <= page <= last


def test_split_pages_covers_the_whole_book():
    book = make_pages(40)
    chunks, chunk_pages = split_pages(book)
    assert chunk_pages[0][0] == 1
    assert chunk_pages[-1][1] == 40
    joined = " ".join(chunks)
    for n in (1, 20, 40):
        assert f"marker{n}" in joined


def test_pages_to_text_has_page_markers():
    text = pages_to_text([(1, "a"), (3, "b")])
    assert text == "[Page 1]\na\n\n[Page 3]\nb"


# ---------- find_page_references ----------

@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is on page 450?", [450]),
        ("explain page 12 please", [12]),
        ("pg. 7 ka summary", [7]),
        ("p 33 mein kya hai", [33]),
        ("page no. 45 par kya likha hai", [45]),
        ("पेज 88 समझाओ", [88]),
        ("450 page par kya hai", [450]),
        ("summarise pages 10-12", [10, 11, 12]),
        ("pages 10 to 11", [10, 11]),
        ("pages 10-20", [10, 11, 12]),  # capped at 3 pages
        ("page 12 of 700", [12]),
        ("what is photosynthesis", []),
        ("step 3 of the method", []),
        ("I read 700 words", []),
        ("page 9999", []),  # beyond the document
        ("page 0", []),
        ("", []),
    ],
)
def test_find_page_references(question, expected):
    assert find_page_references(question, 700) == expected


def test_find_page_references_without_page_count():
    assert find_page_references("page 5", 0) == []


# ---------- overview ----------

def fake_embeddings(count, dim=16, seed=1):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype("float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_page_label():
    assert page_label((5, 5)) == "[Page 5]"
    assert page_label((5, 7)) == "[Pages 5-7]"


def test_select_representative_spreads_over_whole_document():
    emb = fake_embeddings(1000)
    ids = select_representative(emb, 20)
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 20
    # One pick from each twentieth of the document.
    for k, chunk_id in enumerate(ids):
        assert k * 50 <= chunk_id < (k + 1) * 50


def test_select_representative_edge_cases():
    emb = fake_embeddings(5)
    assert select_representative(emb, 0) == []
    assert select_representative(emb, 10) == [0, 1, 2, 3, 4]
    assert select_representative(np.zeros((0, 4)), 3) == []


def test_select_representative_prefers_the_typical_chunk():
    emb = np.array(
        [[1, 0], [0.99, 0.14], [0.95, 0.31], [0, 1], [0.1, 0.99], [0.2, 0.98]],
        dtype="float32",
    )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ids = select_representative(emb, 2)
    assert ids[0] in (0, 1, 2) and ids[1] in (3, 4, 5)


def test_overview_returns_small_documents_unchanged():
    text = "short document"
    out = build_overview(text, ["short document"], [(1, 1)], fake_embeddings(1), 1000)
    assert out == text


def test_overview_covers_a_700_page_book_within_budget():
    book = make_pages(700)
    text = pages_to_text(book)
    chunks, chunk_pages = split_pages(book)
    emb = fake_embeddings(len(chunks))

    budget = 18000
    out = build_overview(text, chunks, chunk_pages, emb, budget)

    assert len(out) <= budget

    # The old "start / middle / end" approach saw ~3 slices. The overview
    # must reach pages from every part of the book.
    pages_seen = set()
    for line in out.splitlines():
        if line.startswith("[Page"):
            label = line.strip("[]").replace("Pages ", "").replace("Page ", "")
            for part in label.split("-"):
                pages_seen.add(int(part))
    assert min(pages_seen) <= 40
    assert max(pages_seen) >= 660
    for lo in range(0, 700, 175):  # each quarter of the book is represented
        assert any(lo < p <= lo + 175 for p in pages_seen)
    assert len(pages_seen) >= 10


def test_bigger_budget_means_more_coverage():
    book = make_pages(700)
    text = pages_to_text(book)
    chunks, chunk_pages = split_pages(book)
    emb = fake_embeddings(len(chunks))

    small = build_overview(text, chunks, chunk_pages, emb, 18000)
    large = build_overview(text, chunks, chunk_pages, emb, 60000)

    assert len(large) <= 60000
    assert large.count("[Page") > small.count("[Page")
