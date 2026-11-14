from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """Split extracted text into overlapping chunks while keeping source metadata."""
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document.get("page"),
                })

            if end == len(text):
                break
            start += step

    return chunks
