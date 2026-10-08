def _check_ocr() -> bool:
    try:
        import pytesseract
        import fitz            # PyMuPDF
        from PIL import Image
        pytesseract.get_tesseract_version()  # raises if the Tesseract binary isn't found
        return True
    except Exception:
        return False

OCR_AVAILABLE = _check_ocr()

def ocr_pdf(path: str) -> str:

    if not OCR_AVAILABLE:
        return ""
    try:
        import pytesseract
        import fitz
        from PIL import Image

        parts = []
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            parts.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""