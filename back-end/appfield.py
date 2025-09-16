from fastapi import FastAPI, File, UploadFile, Form
from paddleocr import PaddleOCR
from PIL import Image
import os
from pdf2image import convert_from_path

app = FastAPI(title="GScan Field Extraction")

ocr = PaddleOCR(use_angle_cls=True, lang="pt")

def preprocess_image(img: Image.Image, max_size=(1200, 1200)):
    img.thumbnail(max_size)
    return img.convert("L")  # grayscale

def ocr_image(img: Image.Image):
    temp_path = "temp.png"
    img.save(temp_path)
    result = ocr.predict(temp_path)
    os.remove(temp_path)
    # Junta todo o texto em linhas
    lines = []
    for page in result:
        for line in page:
            lines.append(line[1][0])
    return lines

def ocr_pdf(pdf_path, dpi=200):
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"C:\poppler\Library\bin")
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

@app.post("/extract_fields")
async def extract_fields_endpoint(
    file: UploadFile = File(...),
    fields: str = Form(...)
):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    # Recebe lista de campos como string separada por vírgula
    fields_list = [f.strip() for f in fields.split(',')]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(temp_file)
        lines = ocr_image(preprocess_image(img))
    elif ext == ".pdf":
        lines = ocr_pdf(temp_file)
    else:
        os.remove(temp_file)
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    os.remove(temp_file)

    data = extract_fields(lines, fields_list)

    return {"documento": file.filename, "extraido": data}
