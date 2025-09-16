from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

app = FastAPI(title="GScan OCR Prático")

# Ajuste o caminho do Tesseract se necessário
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_pdf_digital(pdf_path):
    """Extrai texto de PDFs digitais (não escaneados)"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def ocr_image_without_pre_processing(img: Image.Image):
    """OCR de imagem com Tesseract"""
    gray = img.convert("L")
    text = pytesseract.image_to_string(gray, lang='por')
    return text.strip()

def ocr_image(img: Image.Image):
    """OCR de imagem com pré-processamento para Tesseract"""
    # converter PIL -> numpy
    cv_img = np.array(img)

    # garantir que é BGR
    if len(cv_img.shape) == 2:
        gray = cv_img
    else:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # remover ruído
    gray = cv2.medianBlur(gray, 3)

    # binarização adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # config mais estável para documentos
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(thresh, lang="por", config=config)
    return text.strip()

def ocr_pdf_scanned(pdf_path, dpi=200):
    """OCR de PDFs escaneados ou imagens em PDF"""
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"C:\poppler\Library\bin")
    texts = []
    for page in pages:
        texts.append(ocr_image(page))
    return "\n".join(texts)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    ext = os.path.splitext(file.filename)[1].lower()

    try:
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(temp_file)
            text = ocr_image(img)
        elif ext == ".pdf":
            # Tenta extrair PDF digital primeiro
            text = extract_text_pdf_digital(temp_file)
            if not text.strip():
                # PDF escaneado → OCR
                text = ocr_pdf_scanned(temp_file, dpi=200)
        else:
            os.remove(temp_file)
            return {"error": "Formato não suportado. Use PDF ou imagem."}
    except Exception as e:
        os.remove(temp_file)
        return {"error": f"Ocorreu um erro no OCR: {str(e)}"}

    os.remove(temp_file)

    return {"documento": file.filename, "texto": text}
