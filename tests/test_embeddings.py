from app.ingestion.pdf_loader import extract_text_from_pdf
from app.rag.chunking import chunk_text
from app.rag.embeddings import create_embeddings

pdf_path = "data/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

print(f"Total embeddings: {len(embeddings)}")

print(embeddings[0][:10])