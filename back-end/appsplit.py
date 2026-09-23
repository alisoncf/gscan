from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import io
import zipfile
import fitz  # PyMuPDF

app = FastAPI(title="GScan - Split PDF")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def parse_intervalo_paginas(paginas: str, total_paginas: int):
    """Converte '1,3,5-7' (1-based) em índices de página 0-based, validados contra o total"""
    if not paginas or not paginas.strip():
        return list(range(total_paginas))

    indices = set()
    for parte in paginas.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            inicio_str, fim_str = parte.split("-", 1)
            inicio, fim = int(inicio_str), int(fim_str)
        else:
            inicio = fim = int(parte)

        for pagina in range(inicio, fim + 1):
            if 1 <= pagina <= total_paginas:
                indices.add(pagina - 1)

    return sorted(indices)

@app.post("/split")
async def split_pdf(file: UploadFile = File(...), paginas: str = Form(default="")):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    ext = os.path.splitext(file.filename)[1].lower()

    try:
        if ext != ".pdf":
            os.remove(temp_file)
            return {"error": "Formato não suportado. Use PDF."}

        documento = fitz.open(temp_file)
        total_paginas = documento.page_count
        indices = parse_intervalo_paginas(paginas, total_paginas)

        if not indices:
            documento.close()
            os.remove(temp_file)
            return {"error": "Nenhuma página válida foi selecionada."}

        nome_base = os.path.splitext(file.filename)[0]

        buffer_zip = io.BytesIO()
        with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for indice in indices:
                pagina_doc = fitz.open()
                pagina_doc.insert_pdf(documento, from_page=indice, to_page=indice)

                buffer_pagina = io.BytesIO()
                pagina_doc.save(buffer_pagina)
                pagina_doc.close()

                numero_pagina = indice + 1
                nome_arquivo = f"{nome_base}_pagina_{numero_pagina}.pdf"
                zf.writestr(nome_arquivo, buffer_pagina.getvalue())

        documento.close()
    except Exception as e:
        os.remove(temp_file)
        return {"error": f"Ocorreu um erro ao dividir o PDF: {str(e)}"}

    os.remove(temp_file)
    buffer_zip.seek(0)

    return StreamingResponse(
        buffer_zip,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{nome_base}_paginas.zip"'},
    )
