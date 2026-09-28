from fastapi import APIRouter, FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
from pdf2image import convert_from_path

from comum import POPPLER_PATH, caminho_temporario, paddle_predict, salvar_upload

router = APIRouter()

def preprocess_image(img: Image.Image, max_size=(1200, 1200)):
    img.thumbnail(max_size)
    return img.convert("L")  # grayscale

def ocr_image(img: Image.Image):
    temp_path = caminho_temporario(".png")
    img.save(temp_path)
    try:
        result = paddle_predict(temp_path)
    finally:
        os.remove(temp_path)
    # Junta todo o texto em linhas
    lines = []
    for page in result:
        lines.extend(page["rec_texts"])
    return lines

def ocr_pdf(pdf_path, dpi=200):
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    all_lines = []
    for page in pages:
        pre = preprocess_image(page)
        all_lines.extend(ocr_image(pre))
    return all_lines

def extract_fields(lines, fields):
    """Procura os campos no texto e devolve valor após ':' ou próximo"""
    data = {}
    for field in fields:
        found = False
        for line in lines:
            if field.lower() in line.lower():
                # tenta pegar valor após ':'
                if ':' in line:
                    _, valor = line.split(':', 1)
                    data[field] = valor.strip()
                else:
                    # se não tiver ':', pega texto inteiro da linha
                    data[field] = line.strip()
                found = True
                break
        if not found:
            data[field] = None
    return data

@router.post("/extract_fields")
async def extract_fields_endpoint(
    file: UploadFile = File(...),
    fields: str = Form(...)
):
    # Recebe lista de campos como string separada por vírgula
    fields_list = [f.strip() for f in fields.split(',')]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    temp_file = await salvar_upload(file)
    try:
        if ext == ".pdf":
            lines = ocr_pdf(temp_file)
        else:
            with Image.open(temp_file) as img:
                lines = ocr_image(preprocess_image(img))
    finally:
        os.remove(temp_file)

    data = extract_fields(lines, fields_list)

    return {"documento": file.filename, "extraido": data}

# App próprio, para rodar só este endpoint: uvicorn appfield:app --port 8001
app = FastAPI(title="GScan Field Extraction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
