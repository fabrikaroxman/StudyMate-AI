\
import fitz


def extract_pdf(uploaded_file):
    """Return (full_text, page_count) from a Streamlit UploadedFile."""
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(parts), len(doc)


def split_text(text, chunk_size=900, overlap=150):
    """Character-based chunking with overlap for a simple RAG pipeline."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
