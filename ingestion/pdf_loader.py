import fitz


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    try:
        for page in doc:
            text += page.get_text()
    finally:
        doc.close()

    return text


def extract_pages_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    try:
        for page_index, page in enumerate(doc, start=1):
            pages.append({
                "page": page_index,
                "text": page.get_text(),
            })
    finally:
        doc.close()

    return pages
