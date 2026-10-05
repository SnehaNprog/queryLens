from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


def pdfloader(path):
    """Load a PDF as one Document per page, keeping the page number for citations."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such PDF: '{path}'")

    reader = PdfReader(path)
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": number, "type": "pdf"},
                )
            )
    return docs
