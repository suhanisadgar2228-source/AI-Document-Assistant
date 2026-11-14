
from io import BytesIO
from pathlib import Path
import re
import unicodedata

from docx import Document
from pypdf import PdfReader


def clean_text(text):
    """Clean extracted text while preserving meaningful content."""

    if not text:
        return ""

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove replacement characters and null bytes
    text = text.replace("\ufffd", "")
    text = text.replace("\x00", "")

    # Remove invisible zero-width characters
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unwanted control characters but preserve newlines and tabs
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_docx_text(file_bytes):
    """Extract text from DOCX paragraphs and tables."""

    document = Document(BytesIO(file_bytes))
    parts = []

    # Extract paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text.strip())

    # Extract table content
    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def load_document(uploaded_file):
    """Extract cleaned text and source metadata from PDF, DOCX, or TXT."""

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    sections = []

    if extension == ".pdf":

        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):

            text = clean_text(page.extract_text() or "")

            if text:
                sections.append({
                    "text": text,
                    "source": filename,
                    "page": page_number,
                })

    elif extension == ".docx":

        text = extract_docx_text(file_bytes)
        text = clean_text(text)

        if text:
            sections.append({
                "text": text,
                "source": filename,
                "page": None,
            })

    elif extension == ".txt":

        text = file_bytes.decode("utf-8-sig", errors="replace")
        text = clean_text(text)

        if text:
            sections.append({
                "text": text,
                "source": filename,
                "page": None,
            })

    else:
        raise ValueError(
            "Unsupported file type. Upload PDF, DOCX, or TXT."
        )

    if not sections:
        raise ValueError(
            f"No readable text found in {filename}. "
            "If it is a scanned PDF, OCR support is required."
        )

    return sections
