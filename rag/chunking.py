from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 300


def _splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def chunk_text(text):
    chunks = _splitter().split_text(text or "")
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]


def chunk_pages(pages):
    page_chunks = []
    splitter = _splitter()

    for page in pages:
        page_number = int(page.get("page", 0))
        text = page.get("text", "")
        chunks = splitter.split_text(text)

        for chunk_index, chunk in enumerate(chunks, start=1):
            cleaned = chunk.strip()
            if len(cleaned) <= 100:
                continue
            page_chunks.append({
                "id": f"p{page_number}-c{chunk_index}",
                "page": page_number,
                "chunk_index": chunk_index,
                "text": cleaned,
            })

    return page_chunks


def chunk_texts(page_chunks):
    return [item["text"] if isinstance(item, dict) else str(item) for item in page_chunks]
