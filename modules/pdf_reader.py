import re
from bisect import bisect_right


def extract_pdf_pages(uploaded_file):
    """
    Return ([(page_number, text), ...], page_count) from a Streamlit
    UploadedFile. Pages without selectable text are skipped, but page
    numbers always match the real PDF page numbers.
    """
    # Imported here so the text helpers below can be used (and tested)
    # without PyMuPDF installed.
    import fitz

    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append((i + 1, text))
    return pages, len(doc)


def pages_to_text(pages):
    """Join (page_number, text) pairs into one string with [Page N] markers."""
    return "\n\n".join(f"[Page {n}]\n{t}" for n, t in pages)


def extract_pdf(uploaded_file):
    """Return (full_text, page_count) from a Streamlit UploadedFile."""
    pages, count = extract_pdf_pages(uploaded_file)
    return pages_to_text(pages), count


def _chunk_spans(text, chunk_size=900, overlap=150):
    """
    Return (start, end) character spans that cut `text` into overlapping
    chunks on word boundaries. `text` should already have its whitespace
    normalised to single spaces.
    """
    n = len(text)
    spans = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # Pull the end back to the last space in the second half of the chunk.
        if end < n:
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut

        spans.append((start, end))

        if end >= n:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end

        # Move the next chunk's start forward to a word boundary.
        if text[next_start - 1] != " ":
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1

        start = next_start

    return spans


def split_text(text, chunk_size=900, overlap=150):
    """
    Character-based chunking with overlap for a simple RAG pipeline.

    Chunks end and start on word boundaries where possible, so words are
    not cut in half, which gives the embedding model cleaner input.
    """
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    for start, end in _chunk_spans(text, chunk_size, overlap):
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_pages(pages, chunk_size=900, overlap=150):
    """
    Chunk a whole document while remembering where each chunk came from.

    `pages` is a list of (page_number, text). Returns (chunks, chunk_pages)
    where chunk_pages[i] is (first_page, last_page) for chunks[i]. Chunks
    can run across a page break, so the text flows like the real document.
    """
    pieces = []
    starts = []
    page_numbers = []
    position = 0

    for number, text in pages:
        normalized = " ".join((text or "").split())
        if not normalized:
            continue
        if pieces:
            position += 1  # the joining space
        starts.append(position)
        page_numbers.append(number)
        pieces.append(normalized)
        position += len(normalized)

    if not pieces:
        return [], []

    full = " ".join(pieces)

    def page_at(offset):
        return page_numbers[max(0, bisect_right(starts, offset) - 1)]

    chunks = []
    chunk_pages = []

    for start, end in _chunk_spans(full, chunk_size, overlap):
        raw = full[start:end]
        chunk = raw.strip()
        if not chunk:
            continue
        first = start + (len(raw) - len(raw.lstrip()))
        last = max(first, end - 1)
        chunks.append(chunk)
        chunk_pages.append((page_at(first), page_at(last)))

    return chunks, chunk_pages


_NUM = r"(\d{1,5})"
_RANGE_JOIN = r"(?:-|–|—|to|se|से)"
_PAGE_WORD = r"(?:pages?|pg|p|पेज|पृष्ठ)"

# "page 450", "pg. 12", "p 7", "pages 10-12", "पेज 45", "page no. 5"
_AFTER_WORD = re.compile(
    r"(?<![A-Za-z])" + _PAGE_WORD + r"\.?\s*(?:no\.?|number|नंबर)?\s*[:#]?\s*"
    + _NUM + r"(?:\s*" + _RANGE_JOIN + r"\s*" + _NUM + r")?",
    re.IGNORECASE,
)

# "450 page", "450 number page"
_BEFORE_WORD = re.compile(
    r"(?<![\d-])" + _NUM + r"\s*(?:number|no\.?|नंबर)?\s*"
    + r"(?:pages?|पेज|पृष्ठ)(?![A-Za-z])",
    re.IGNORECASE,
)


def find_page_references(question, max_page, limit=3):
    """
    Return the page numbers a question explicitly asks about, for example
    "what is on page 450?" or "summarise pages 10-12". Only pages that exist
    (1..max_page) are returned, at most `limit` of them, in ascending order.
    """
    if not question or not max_page:
        return []

    found = []

    for match in _AFTER_WORD.finditer(question):
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        if last < first:
            first, last = last, first
        if last - first >= limit:
            last = first + limit - 1
        found.extend(range(first, last + 1))

    for match in _BEFORE_WORD.finditer(question):
        found.append(int(match.group(1)))

    pages = sorted({p for p in found if 1 <= p <= max_page})
    return pages[:limit]
