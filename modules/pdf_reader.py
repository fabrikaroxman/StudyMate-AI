def extract_pdf(uploaded_file):
    """Return (full_text, page_count) from a Streamlit UploadedFile."""
    # Imported here so the text-splitting helpers below can be used
    # (and tested) without PyMuPDF installed.
    import fitz

    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(parts), len(doc)


def split_text(text, chunk_size=900, overlap=150):
    """
    Character-based chunking with overlap for a simple RAG pipeline.

    Chunks end and start on word boundaries where possible, so words are
    not cut in half, which gives the embedding model cleaner input.
    """
    text = " ".join(text.split())
    if not text:
        return []

    n = len(text)
    chunks = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # Pull the end back to the last space in the second half of the chunk.
        if end < n:
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

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

    return chunks
