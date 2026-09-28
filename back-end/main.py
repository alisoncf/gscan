"""Servidor único com todos os endpoints do GScan: uvicorn main:app --port 8000"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
