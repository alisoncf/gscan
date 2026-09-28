from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

from comum import POPPLER_PATH, TESSERACT_CMD, salvar_upload

router = APIRouter()

# Caminho do Tesseract: variável de ambiente TESSERACT_CMD (veja comum.py)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

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
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    texts = []
    for page in pages:
        texts.append(ocr_image(page))
    return "\n".join(texts)

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    temp_file = await salvar_upload(file)
    try:
        if ext == ".pdf":
            # Tenta extrair PDF digital primeiro
            text = extract_text_pdf_digital(temp_file)
            if not text.strip():
                # PDF escaneado → OCR
                text = ocr_pdf_scanned(temp_file, dpi=200)
        else:
            with Image.open(temp_file) as img:
                text = ocr_image(img)
    except Exception as e:
        return {"error": f"Ocorreu um erro no OCR: {str(e)}"}
    finally:
        os.remove(temp_file)

    return {"documento": file.filename, "texto": text}

# App próprio, para rodar só este endpoint: uvicorn app:app --port 8000
app = FastAPI(title="GScan OCR Prático")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
