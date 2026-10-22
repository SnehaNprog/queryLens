import pytest

from Loaders import pdfloader, webloader


class FakeResponse:
    text = "<html><title>Hi</title><body><script>bad()</script><p>Hello</p></body></html>"

    def raise_for_status(self):
        pass


def test_pdfloader_returns_one_document_per_page():
    docs = pdfloader("data/sample.pdf")
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1


def test_pdfloader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pdfloader("data/nope.pdf")


def test_pdfloader_metadata():
    docs = pdfloader("data/sample.pdf")
    assert docs[0].metadata["source"] == "sample.pdf"
    assert docs[0].metadata["type"] == "pdf"


def test_webloader_removes_noise(monkeypatch):
    monkeypatch.setattr(
        "Loaders.requests.get",
        lambda *a, **k: FakeResponse()
    )

    docs = webloader("http://x.test")

    assert "Hello" in docs[0].page_content
    assert "bad()" not in docs[0].page_content
    assert docs[0].metadata["title"] == "Hi"



def test_pdfloader_accepts_uploaded_file_object():
    with open("data/sample.pdf", "rb") as upload:
        docs = pdfloader(upload, source_name="up.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "up.pdf"
