from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import os
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="GScan", description="API OCR otimizada para PDF e imagens")

ocr = PaddleOCR(use_angle_cls=True, lang="pt")

# Configurações de redimensionamento
MAX_WIDTH = 1200
MAX_HEIGHT = 1200

def preprocess_image(img: Image.Image):
    """Redimensiona e converte para grayscale"""
    img.thumbnail((MAX_WIDTH, MAX_HEIGHT))
    gray = img.convert("L")  # grayscale
    return gray

def ocr_image(img: Image.Image):
    """Executa OCR em imagem PIL"""
    temp_path = "temp_page.png"
    img.save(temp_path)
    result = ocr.predict(temp_path)
    os.remove(temp_path)
    text = " ".join([line[1][0] for page in result for line in page])
    return text

def ocr_pdf(pdf_path, dpi=200):
    """Processa PDF multipágina em paralelo"""
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, poppler_path=r"C:\poppler\Library\bin")
    texts = []

    def process_page(page):
        preprocessed = preprocess_image(page)
        return ocr_image(preprocessed)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_page, pages)
        texts.extend(results)

    return "\n".join(texts)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # Salva temporariamente
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(temp_file)
        text = ocr_image(preprocess_image(img))
    elif ext == ".pdf":
        text = ocr_pdf(temp_file, dpi=200)
    else:
        os.remove(temp_file)
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    os.remove(temp_file)

    # Parser simples chave:valor
    data = {}
    for line in text.split("\n"):
        if ":" in line:
            chave, valor = line.split(":", 1)
            data[chave.strip().lower()] = valor.strip()
        elif line.strip():  # ignora linhas vazias
            data[f"campo_{len(data)+1}"] = line.strip()

    return {"documento": file.filename, "extraido": data}
