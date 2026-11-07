"""
Whole-document coverage for summary, topics, quiz, flashcards and plans.

A language model can only read a small slice of a big PDF per request. Taking
"the start, the middle and the end" (the old behaviour) ignores almost all of
a 700-page book. Instead, the book is split into equal parts and the most
representative passage of each part is chosen, so every region of the
document contributes. The prompt stays the same size, so it stays fast.

Only numpy is needed here, so this module is easy to test.
"""

import numpy as np


def page_label(pages):
    """(12, 12) -> "[Page 12]"; (12, 14) -> "[Pages 12-14]"."""
    first, last = pages
    if first == last:
        return f"[Page {first}]"
    return f"[Pages {first}-{last}]"


def select_representative(embeddings, n):
    """
    Pick `n` chunk ids spread evenly over the whole document.

    The chunks (in document order) are cut into `n` equal consecutive
    segments. From each segment the chunk closest to the segment's average
    embedding is chosen, i.e. its most typical passage. Returns ids in
    ascending (document) order.
    """
    vectors = np.asarray(embeddings, dtype="float32")
    total = len(vectors)

    if n <= 0 or total == 0:
        return []

    if n >= total:
        return list(range(total))

    bounds = np.linspace(0, total, n + 1).astype(int)
    chosen = []

    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue
        segment = vectors[start:end]
        centre = segment.mean(axis=0)
        chosen.append(int(start + (segment @ centre).argmax()))

    return chosen


def build_overview(text, chunks, chunk_pages, embeddings, max_chars):
    """
    Return study material of at most ~`max_chars` characters that covers
    the entire document.

    * A document that already fits is returned unchanged.
    * Otherwise representative chunks from every part of the document are
      returned in reading order, each labelled with its page number(s).
    """
    if len(text) <= max_chars:
        return text

    if not chunks:
        return text[:max_chars]

    average = sum(len(c) for c in chunks) / len(chunks)
    n = max(1, int(max_chars // (average + 24) * 0.95))

    ids = select_representative(embeddings, n)
    parts = [f"{page_label(chunk_pages[i])}\n{chunks[i]}" for i in ids]

    # Stay inside the budget even if some chunks are longer than average:
    # drop the longest passages first so coverage stays spread out.
    while len(parts) > 1 and sum(len(p) for p in parts) + 2 * (len(parts) - 1) > max_chars:
        parts.pop(max(range(len(parts)), key=lambda k: len(parts[k])))

    return "\n\n".join(parts)[:max_chars]
