from app.ingestion.pdf_loader import extract_text_from_pdf
from app.rag.chunking import chunk_text
from app.rag.embeddings import create_embeddings, model
from app.rag.vector_store import create_faiss_index, search_similar_chunks

pdf_path = "data/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

index = create_faiss_index(embeddings)

query = "What is artificial intelligence?"

query_embedding = model.encode([query])[0]

results = search_similar_chunks(query_embedding, index)

print("Top matching chunks:\n")

for idx in results:
    print(chunks[idx])
    print("\n-----------------\n")