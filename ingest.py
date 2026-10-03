import pymupdf

PDF_PATH = "data/sample.pdf"
OUTPUT_PATH = "output.txt"

with pymupdf.open(PDF_PATH) as doc, open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for page in doc:
        out.write(page.get_text())
        out.write("\f")

print(f"Extracted text from {PDF_PATH} to {OUTPUT_PATH}")
