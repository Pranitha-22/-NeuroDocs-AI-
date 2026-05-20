from app.ingestion.ocr_loader import extract_text_from_scanned_pdf

pdf_path = "data/sample.pdf"

text = extract_text_from_scanned_pdf(pdf_path)

print(text[:3000])