from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import os
import re
import difflib
import unicodedata
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import uuid

app = FastAPI(title="GScan", description="API OCR otimizada para PDF e imagens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    """Executa OCR em imagem PIL e devolve [(texto, [x1, y1, x2, y2]), ...]"""
    temp_path = f"temp_page_{uuid.uuid4().hex}.png"
    img.save(temp_path)
    result = ocr.predict(temp_path)
    os.remove(temp_path)
    items = []
    for page in result:
        for text, box in zip(page["rec_texts"], page["rec_boxes"]):
            items.append((text.strip(), [float(v) for v in box]))
    return items

def ocr_pdf(pdf_path, dpi=200):
    """Processa PDF multipágina em paralelo; devolve uma lista de itens por página"""
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, poppler_path=r"C:\poppler\Library\bin")

    def process_page(page):
        preprocessed = preprocess_image(page)
        return ocr_image(preprocessed)

    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(process_page, pages))

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # Salva temporariamente
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(temp_file)
        pages = [ocr_image(preprocess_image(img))]
    elif ext == ".pdf":
        pages = ocr_pdf(temp_file, dpi=200)
    else:
        os.remove(temp_file)
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    os.remove(temp_file)

    return {"documento": file.filename, "extraido": parse_key_values(pages)}

# ':' que separa chave e valor; ignora ':' entre dígitos (ex.: horário 14:30)
SEPARADOR = re.compile(r"(?<!\d):|:(?!\d)")

# Rótulos comuns em documentos sem ':' (ex.: RG, CNH). Sem acento, em maiúsculas.
ROTULOS = [
    "NOME", "NOME SOCIAL", "CPF", "RG", "REGISTRO GERAL", "DOC IDENTIDADE",
    "ORGAO EMISSOR", "UF", "DATA DE EXPEDICAO", "DATA NASCIMENTO",
    "DATA DE NASCIMENTO", "FILIACAO", "NATURALIDADE", "NACIONALIDADE",
    "REGISTRO CIVIL", "DOC ORIGEM", "CTPS", "SERIE", "NIS PIS PASEP",
    "CERT MILITAR", "CNH", "CNS", "TITULO DE ELEITOR", "ZONA", "SECAO",
    "TIPO SANGUINEO", "FATOR RH", "TIPO FATOR RH", "VALIDADE", "OBSERVACAO",
    "SEXO", "ENDERECO", "BAIRRO", "MUNICIPIO", "CEP", "TELEFONE", "EMAIL",
    "MATRICULA", "CURSO",
]
_ROTULOS_COMPACTOS = {r.replace(" ", ""): r for r in ROTULOS}
# rótulos curtos (RG, UF, CPF...) só valem com acerto exato, para não pegar lixo do OCR
_ROTULOS_APROXIMAVEIS = [c for c in _ROTULOS_COMPACTOS if len(c) >= 4]

def normalizar(texto):
    """Maiúsculas, sem acento e sem pontuação"""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t).upper()
    return " ".join(t.split())

def identificar_rotulo(texto):
    """Devolve o rótulo conhecido que o texto representa (tolera erros do OCR) ou None"""
    compacto = normalizar(texto).replace(" ", "")
    if compacto in _ROTULOS_COMPACTOS:
        return _ROTULOS_COMPACTOS[compacto]
    if len(compacto) < 4:
        return None
    match = difflib.get_close_matches(compacto, _ROTULOS_APROXIMAVEIS, n=1, cutoff=0.8)
    return _ROTULOS_COMPACTOS[match[0]] if match else None

def distancia_valor(rotulo, candidato):
    """Distância do rótulo até uma caixa logo abaixo ou logo à direita dele; None se não for vizinha"""
    rx1, ry1, rx2, ry2 = rotulo
    cx1, cy1, cx2, cy2 = candidato
    altura = max(ry2 - ry1, 1)
    folga = altura * 0.5
    # abaixo: começa depois do rótulo e se sobrepõe a ele na horizontal
    if cy1 >= ry2 - folga and min(rx2, cx2) > max(rx1, cx1):
        d = cy1 - ry2
        return max(d, 0) if d <= altura * 3 else None
    # à direita: na mesma altura do rótulo
    if ry1 <= (cy1 + cy2) / 2 <= ry2 and cx1 >= rx2 - folga:
        d = cx1 - rx2
        return max(d, 0) if d <= altura * 8 else None
    return None

def parse_key_values(pages):
    """Monta o dicionário chave:valor a partir das caixas do OCR de cada página"""
    data = {}

    def add(chave, valor):
        # chave repetida ganha sufixo (_2, _3...) em vez de sobrescrever
        final = chave
        n = 2
        while final in data:
            final = f"{chave}_{n}"
            n += 1
        data[final] = valor

    for items in pages:
        # descarta caixas sem letra nem número (ex.: "*…", "-")
        items = [(t, b) for t, b in items if re.search(r"\w", t)]
        diretos = {}  # índice -> (chave, valor) de linhas "chave: valor"
        rotulos = {}  # índice -> chave de rótulos que esperam valor em outra caixa
        for i, (texto, _) in enumerate(items):
            m = SEPARADOR.search(texto)
            chave = texto[:m.start()].strip().lower() if m else ""
            if chave:
                valor = texto[m.end():].strip()
                if valor:
                    diretos[i] = (chave, valor)
                else:
                    rotulos[i] = chave
                continue
            rotulo = identificar_rotulo(texto)
            if rotulo:
                rotulos[i] = rotulo.lower().replace(" ", "_")

        # liga cada rótulo à caixa livre mais próxima, começando pelos pares mais próximos
        pares = []
        for i in rotulos:
            for j, (_, box) in enumerate(items):
                if j in rotulos or j in diretos:
                    continue
                d = distancia_valor(items[i][1], box)
                if d is not None:
                    pares.append((d, i, j))
        valor_de = {}
        usados = set()
        for _, i, j in sorted(pares):
            if i not in valor_de and j not in usados:
                valor_de[i] = j
                usados.add(j)

        for i, (texto, _) in enumerate(items):
            if i in diretos:
                add(*diretos[i])
            elif i in rotulos:
                add(rotulos[i], items[valor_de[i]][0] if i in valor_de else "")
            elif i not in usados:
                add(f"campo_{len(data)+1}", texto)
    return data
