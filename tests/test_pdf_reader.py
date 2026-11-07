import random

from modules.pdf_reader import split_text


def _words(n, seed=0):
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "photosynthesis", "mitochondria", "x"]
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_empty_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\t ") == []


def test_short_text_is_a_single_chunk():
    assert split_text("hello   world\nagain") == ["hello world again"]


def test_chunks_respect_size_limit():
    text = _words(2000)
    chunks = split_text(text, chunk_size=900, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_words_are_never_cut_in_half():
    text = _words(2000)
    vocab = set(text.split())
    for chunk in split_text(text):
        assert set(chunk.split()) <= vocab


def test_every_word_is_covered_in_order():
    text = _words(1500, seed=3)
    chunks = split_text(text, chunk_size=300, overlap=60)
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])
    # Consecutive chunks overlap, so no text is lost between them.
    for a, b in zip(chunks, chunks[1:]):
        assert text.index(b) <= text.index(a) + len(a)


def test_terminates_on_text_without_spaces():
    chunks = split_text("a" * 5000, chunk_size=900, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_overlap_larger_than_chunk_still_terminates():
    chunks = split_text(_words(500), chunk_size=100, overlap=500)
    assert chunks
