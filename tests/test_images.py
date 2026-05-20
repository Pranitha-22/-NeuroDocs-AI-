from app.ingestion.image_extraction import extract_images_from_pdf

pdf_path = "data/sample.pdf"

output_folder = "data/extracted_images"

count = extract_images_from_pdf(pdf_path, output_folder)

print(f"Extracted {count} images")