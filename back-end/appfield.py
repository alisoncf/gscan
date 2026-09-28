from fastapi import APIRouter, FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import difflib

from appall import (FORMATOS_ACEITOS, ROTULOS, ler_documento, normalizar,
                    parse_key_values, separar_tabelas, vocabulario)

router = APIRouter()

def compactar(texto):
    """'Data de Nascimento' / 'data_de_nascimento_2' -> 'DATADENASCIMENTO'"""
    return normalizar(re.sub(r"_\d+$", "", texto)).replace(" ", "")

def extract_fields(data, fields):
    """Procura cada campo pedido entre as chaves extraídas; None se não achar"""
    # campo_N são textos soltos, sem rótulo
    chaves = [c for c in data if not re.fullmatch(r"campo_\d+", c)]
    compactas = [compactar(c) for c in chaves]
    resultado = {}
    for field in fields:
        alvo = compactar(field)
        # primeira ocorrência no documento; se não houver igual, a mais parecida (erros de OCR)
        if alvo in compactas:
            resultado[field] = data[chaves[compactas.index(alvo)]]
            continue
        match = difflib.get_close_matches(alvo, compactas, n=1, cutoff=0.85)
        resultado[field] = data[chaves[compactas.index(match[0])]] if match else None
    return resultado

@router.post("/extract_fields")
async def extract_fields_endpoint(
    file: UploadFile = File(...),
    fields: str = Form(...)
):
    # Recebe lista de campos como string separada por vírgula
    fields_list = [f.strip() for f in fields.split(',') if f.strip()]

    if os.path.splitext(file.filename)[1].lower() not in FORMATOS_ACEITOS:
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    pages = await ler_documento(file)
    _, restantes = separar_tabelas(pages)
    # os campos pedidos viram rótulos conhecidos: são achados mesmo sem ':',
    # com o valor ao lado ou embaixo
    vocab = vocabulario(ROTULOS + [normalizar(f) for f in fields_list])
    data = parse_key_values(restantes, vocab)

    return {"documento": file.filename, "extraido": extract_fields(data, fields_list)}

# App próprio, para rodar só este endpoint: uvicorn appfield:app --port 8001
app = FastAPI(title="GScan Field Extraction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
