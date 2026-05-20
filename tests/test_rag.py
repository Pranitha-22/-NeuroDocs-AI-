from app.ingestion.pdf_loader import extract_text_from_pdf
from app.rag.chunking import chunk_text
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import create_faiss_index
from app.rag.qa_pipeline import generate_rag_response

pdf_path = "data/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

index = create_faiss_index(embeddings)

query = input("Ask a question: ")

response = generate_rag_response(
    query,
    chunks,
    index
)

print(response)