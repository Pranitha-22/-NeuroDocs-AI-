import fitz
import os

def extract_images_from_pdf(pdf_path, output_folder):

    doc = fitz.open(pdf_path)

    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    for page_index in range(len(doc)):

        page = doc.load_page(page_index)

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):

            xref = img[0]

            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            image_filename = f"page_{page_index+1}_img_{img_index+1}.{image_ext}"

            image_path = os.path.join(output_folder, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_count += 1

    return image_count