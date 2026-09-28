"""Recursos compartilhados entre os endpoints: caminhos externos, upload temporário e PaddleOCR"""
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import UploadFile

# Caminhos de programas externos. No Windows o padrão é o local de instalação usual;
# em Linux/Docker ficam vazios e os programas são encontrados pelo PATH.
_WINDOWS = os.name == "nt"
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler\Library\bin" if _WINDOWS else "") or None
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe" if _WINDOWS else "") or None


async def salvar_upload(file: UploadFile):
    """Salva o upload na pasta temporária do sistema e devolve o caminho (apague com os.remove)"""
    ext = os.path.splitext(file.filename)[1].lower()
    caminho = os.path.join(tempfile.gettempdir(), f"gscan_{uuid.uuid4().hex}{ext}")
    with open(caminho, "wb") as f:
        f.write(await file.read())
    return caminho


def caminho_temporario(ext):
    """Caminho único na pasta temporária do sistema (ex.: para imagens de página)"""
    return os.path.join(tempfile.gettempdir(), f"gscan_{uuid.uuid4().hex}{ext}")


# PaddleOCR é carregado uma vez só, na primeira vez que for usado, e compartilhado
# entre os endpoints. Ele precisa ser criado e usado sempre na mesma thread: chamadas
# simultâneas derrubam o processo, e chamadas de outra thread falham com
# "RuntimeError: std::exception". Por isso todo o uso passa por uma thread dedicada.
_ocr = None
_ocr_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddleocr")


def _predict(caminho_imagem):
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="pt")
    return _ocr.predict(caminho_imagem)


def paddle_predict(caminho_imagem):
    return _ocr_thread.submit(_predict, caminho_imagem).result()
