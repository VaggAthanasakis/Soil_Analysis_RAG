from pdf2image import convert_from_path
import pytesseract

pdf_file = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/ΕΔΑΦΟΣ ΤΟΠΟΘ 1 20221103 114361 (1).pdf"
output_file = "extracted_text.txt"

pages = convert_from_path(pdf_file, dpi=300)  # Adjust DPI as needed

with open(output_file, "w", encoding="utf-8") as out:
    for i, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image,lang='eng+ell')
        out.write(f"--- Page {i+1} ---\n{text}\n\n")