from Loaders import Document


def chunk_documents(docs, chunk_size=500, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for doc in docs:
        text = doc.page_content
        start = 0

        while start < len(text):
            chunk_text = text[start : start + chunk_size]

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "chunk": len(chunks)}
                )
            )

            if start+chunk_size >= len(text):
                break

            start += chunk_size - overlap

    return chunks