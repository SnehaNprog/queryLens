from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# HTML tags that hold menus, ads and code, not readable content
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]


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


def webloader(url):
    """Load a webpage as a single Document with its URL and title as metadata."""
    # 1. FETCH: download the HTML.
    #    timeout: give up after 15s instead of hanging forever.
    #    headers: some sites reject requests that don't look like a browser.
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; QueryLens/1.0)"},
    )
    #    raise_for_status: turns 404/500 responses into an exception
    #    instead of silently parsing an error page as content.
    response.raise_for_status()

    # 2. PARSE: turn the HTML string into a searchable tree.
    soup = BeautifulSoup(response.text, "html.parser")

    # 3. CLEAN: delete noisy tags from the tree.
    #    find_all(tag) returns every matching tag; decompose() removes each one.
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 4. EXTRACT: get the remaining text, one line per HTML element.
    raw_text = soup.get_text(separator="\n")

    # Strip every line and drop the empty ones.
    clean_text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())

    # 5. METADATA: soup.title can be missing, so fall back to the URL.
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    return [
        Document(
            page_content=clean_text,
            metadata={"source": url, "title": title, "type": "web"},
        )
    ]
