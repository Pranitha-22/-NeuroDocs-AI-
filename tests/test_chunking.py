from app.ingestion.pdf_loader import extract_text_from_pdf
from app.rag.chunking import chunk_text

pdf_path = "data/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")

print(chunks[0])