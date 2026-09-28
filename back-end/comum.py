"""Recursos compartilhados entre os endpoints: caminhos externos, upload temporário e PaddleOCR"""
import os
import tempfile
import threading
import uuid

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
# entre os endpoints. Ele não aceita chamadas simultâneas: duas threads no predict()
# derrubam o processo, por isso a trava.
_ocr = None
_ocr_lock = threading.Lock()


def paddle_predict(caminho_imagem):
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(use_angle_cls=True, lang="pt")
        return _ocr.predict(caminho_imagem)
