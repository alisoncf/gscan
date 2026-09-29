"""Servidor único com todos os endpoints do GScan: uvicorn main:app --port 8000"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

import app as transcribe
import appall
import appfield
import appsplit

app = FastAPI(title="GScan", description="OCR de PDF e imagens: transcrição, campos, chave:valor, tabelas e split de PDF")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(transcribe.router, tags=["Transcrição (Tesseract)"])
app.include_router(appfield.router, tags=["Campos específicos (PaddleOCR)"])
app.include_router(appall.router, tags=["Chave:valor e tabelas (PaddleOCR)"])
app.include_router(appsplit.router, tags=["Split de PDF"])

PAINEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "painel.html")


# Painel servido pela própria API: mesma origem, sem depender de CORS nem da
# permissão de "rede local" que o navegador pede para páginas abertas como arquivo
@app.get("/painel", include_in_schema=False)
def painel():
    return FileResponse(PAINEL)


@app.get("/", include_in_schema=False)
def raiz():
    return RedirectResponse("/painel")
