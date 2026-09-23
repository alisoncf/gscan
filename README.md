# gscan
Generic Scan

O GScan é um conjunto de APIs leves (FastAPI) que recebem documentos digitalizados (PDF, JPG, PNG) e retornam texto ou pares chave:valor em JSON, usando OCR (Tesseract ou PaddleOCR).

O back-end é composto por **quatro aplicações independentes**, cada uma em seu próprio arquivo, com um único endpoint. Elas não rodam juntas na mesma porta — suba cada uma separadamente (ou em portas diferentes) conforme a necessidade.

| Arquivo | Endpoint | Motor | Função |
|---|---|---|---|
| [app.py](back-end/app.py) | `POST /transcribe` | Tesseract | Transcreve o texto completo de PDF (digital ou escaneado) ou imagem |
| [appfield.py](back-end/appfield.py) | `POST /extract_fields` | PaddleOCR | Extrai valores de campos específicos informados na requisição |
| [appall.py](back-end/appall.py) | `POST /extract` | PaddleOCR (paralelo) | OCR completo + parser automático de linhas `chave: valor` |
| [appsplit.py](back-end/appsplit.py) | `POST /split` | PyMuPDF | Divide um PDF em páginas individuais (todas ou um subconjunto), devolvidas como .zip |

## Requisitos

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) instalado (usado por `app.py`). Caminho configurado em [app.py:13](back-end/app.py#L13) — ajuste se instalado em outro local. É preciso ter o pacote de idioma **por** (`tesseract --list-langs` deve listar `por`).
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows) instalado (usado por `pdf2image` em todos os apps). Caminho configurado em cada arquivo, por exemplo [app.py:59](back-end/app.py#L59) — ajuste se instalado em outro local.

## Instalação

```bash
cd back-end
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Como rodar

Cada app sobe em sua própria porta:

```bash
uvicorn app:app --port 8000        # /transcribe
uvicorn appfield:app --port 8001   # /extract_fields
uvicorn appall:app --port 8002     # /extract
uvicorn appsplit:app --port 8003   # /split
```

Cada um expõe documentação interativa (Swagger) em `http://127.0.0.1:<porta>/docs`.

## Painel de testes

Com os quatro servidores no ar, abra [back-end/painel.html](back-end/painel.html) direto no navegador (duplo clique no arquivo) para testar qualquer endpoint sem precisar do front-end nem de curl — escolha o endpoint, selecione o arquivo, preencha os campos opcionais e envie. É só um HTML estático com JavaScript puro, sem servidor próprio; as APIs precisam ter CORS habilitado (já vem configurado em todos os apps) para o navegador aceitar as chamadas.

## Endpoints

### `POST /transcribe` — app.py

Recebe um PDF ou imagem e devolve o texto completo. Para PDF, tenta extrair texto digital primeiro; se o PDF for escaneado (sem texto embutido), faz OCR das páginas automaticamente.

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF, JPG ou PNG

**Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@documento.pdf"
```

**Resposta:**
```json
{
  "documento": "documento.pdf",
  "texto": "Nome: Alison Filgueiras\nCPF: 000.000.000-00"
}
```

### `POST /extract_fields` — appfield.py

Recebe um PDF ou imagem e uma lista de campos desejados; procura cada campo no texto reconhecido e devolve o valor encontrado (texto após `:` na mesma linha, quando existir).

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF, JPG ou PNG
- `fields`: nomes dos campos separados por vírgula (ex.: `"Nome,CPF"`)

**Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8001/extract_fields" \
  -F "file=@documento.png" \
  -F "fields=Nome,CPF"
```

**Resposta:**
```json
{
  "documento": "documento.png",
  "extraido": {
    "Nome": "Alison Filgueiras",
    "CPF": null
  }
}
```
Campos não encontrados vêm como `null`.

### `POST /extract` — appall.py

Recebe um PDF ou imagem, faz OCR (páginas de PDF são processadas em paralelo) e tenta estruturar automaticamente qualquer linha no formato `chave: valor` encontrada no texto — sem precisar informar os campos antecipadamente.

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF, JPG ou PNG

**Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8002/extract" \
  -F "file=@documento.pdf"
```

**Resposta:**
```json
{
  "documento": "documento.pdf",
  "extraido": {
    "nome": "Alison Filgueiras",
    "campo_2": "Endereço não estruturado, por exemplo"
  }
}
```

> **Limitação conhecida:** como o OCR de cada página junta as linhas reconhecidas em uma única string (sem preservar quebras de linha), documentos com múltiplos campos na mesma página podem ter seus valores concatenados incorretamente pelo parser `chave: valor`. Funciona melhor com poucos campos por página ou documentos já bem espaçados.

### `POST /split` — appsplit.py

Recebe um PDF e devolve um `.zip` com uma página por arquivo PDF. Não depende de OCR nem de Poppler — usa só o PyMuPDF.

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF
- `paginas` (opcional): páginas a extrair, 1-based, ex.: `"1,3,5-7"`. Vazio (padrão) extrai todas as páginas.

**Exemplo — páginas específicas:**
```bash
curl -X POST "http://127.0.0.1:8003/split" \
  -F "file=@documento.pdf" \
  -F "paginas=1,3" \
  -o paginas.zip
```

**Exemplo — PDF inteiro:**
```bash
curl -X POST "http://127.0.0.1:8003/split" \
  -F "file=@documento.pdf" \
  -o paginas.zip
```

**Resposta:** arquivo `.zip` (`application/zip`) contendo `documento_pagina_1.pdf`, `documento_pagina_3.pdf`, etc.

## Erros

Todos os endpoints retornam `{"error": "..."}` (HTTP 200) quando o formato do arquivo não é suportado (`app.py`, `appfield.py` e `appall.py` aceitam `.pdf`, `.jpg`, `.jpeg`, `.png`; `appsplit.py` aceita apenas `.pdf`).
