# gscan
Generic Scan

O GScan é um conjunto de APIs leves (FastAPI) que recebem documentos digitalizados (PDF, JPG, PNG) e retornam texto ou pares chave:valor em JSON, usando OCR (Tesseract ou PaddleOCR).

O back-end é um servidor único ([main.py](back-end/main.py)) com quatro endpoints. Cada endpoint fica no seu próprio arquivo; o que é compartilhado (PaddleOCR, caminhos externos, arquivos temporários) fica em [comum.py](back-end/comum.py).

| Arquivo | Endpoint | Motor | Função |
|---|---|---|---|
| [app.py](back-end/app.py) | `POST /transcribe` | Tesseract | Transcreve o texto completo de PDF (digital ou escaneado) ou imagem |
| [appfield.py](back-end/appfield.py) | `POST /extract_fields` | PaddleOCR | Extrai valores de campos específicos informados na requisição |
| [appall.py](back-end/appall.py) | `POST /extract` | PaddleOCR | OCR completo + pares chave:valor e tabelas automáticos |
| [appsplit.py](back-end/appsplit.py) | `POST /split` | PyMuPDF | Divide um PDF em páginas individuais (todas ou um subconjunto), devolvidas como .zip |

## Rodando com Docker (recomendado)

Só precisa do [Docker Desktop](https://www.docker.com/products/docker-desktop/) (no Windows, com WSL2). Python, Tesseract (com português), Poppler e todas as dependências já vão dentro da imagem, nas versões testadas.

```bash
docker compose up -d --build
```

A API fica em `http://127.0.0.1:8000` (Swagger em `/docs`), e o [painel de testes](#painel-de-testes) funciona igual.

- O primeiro build demora alguns minutos e a imagem tem cerca de 2,4 GB, por causa do PaddlePaddle.
- A primeira chamada a `/extract` ou `/extract_fields` baixa os modelos do PaddleOCR (cerca de 1 minuto). Eles ficam guardados no volume `paddle-modelos` e não são baixados de novo.
- Para parar: `docker compose down`. Para ver os logs: `docker compose logs -f`.
- Depois de mudar o código, rode `docker compose up -d --build` de novo.
- Para usar outra porta (ex.: se a 8000 já estiver ocupada), mude só o número da esquerda em `ports` no [docker-compose.yml](docker-compose.yml): `"8080:8000"` deixa a API em `http://127.0.0.1:8080`. No painel, ajuste o campo **Servidor** para o mesmo endereço.

As seções abaixo (Requisitos, Instalação, Como rodar) são para rodar sem Docker.

## Requisitos

- Python **3.10 a 3.13** (recomendado: **3.13**, a versão testada). Python 3.14 ou mais novo não funciona: o `paddlepaddle` ainda não tem pacote para essas versões.
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) instalado (usado pelo `/transcribe`), com o pacote de idioma **por** (`tesseract --list-langs` deve listar `por`).
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows) instalado (usado para converter PDF em imagem).

No Windows, o padrão é `C:\Program Files\Tesseract-OCR\tesseract.exe` e `C:\poppler\Library\bin`. Se instalou em outro lugar, informe pelas variáveis de ambiente `TESSERACT_CMD` e `POPPLER_PATH` antes de subir o servidor (ex.: `set POPPLER_PATH=D:\poppler\Library\bin`). No Linux/macOS, os dois são encontrados pelo PATH e não precisam de configuração.

## Instalação

```bash
cd back-end
py -3.13 -m venv venv        # Windows: cria o venv com o Python 3.13
venv\Scripts\activate
python --version             # confira: deve mostrar Python 3.13.x
pip install -r requirements.txt
```

Se `py -3.13` der erro, o Python 3.13 não está instalado: baixe em [python.org](https://www.python.org/downloads/). No Linux/macOS, use `python3.13 -m venv venv` e `source venv/bin/activate`.

Se já existir um `venv` criado com outra versão do Python, apague a pasta `venv` e crie de novo.

O `requirements.txt` lista os pacotes do projeto, e o [constraints.txt](back-end/constraints.txt) fixa a versão exata de todas as dependências, inclusive as indiretas (numpy, protobuf...). O `pip install -r requirements.txt` já aplica os dois. Ao atualizar um pacote, gere o `constraints.txt` de novo com `pip freeze` a partir de um `venv` testado.

## Como rodar

```bash
cd back-end
uvicorn main:app --port 8000
```

Todos os endpoints ficam em `http://127.0.0.1:8000`, com documentação interativa (Swagger) em `http://127.0.0.1:8000/docs`. O PaddleOCR é carregado na primeira chamada a `/extract` ou `/extract_fields`, então essa primeira chamada demora alguns segundos a mais.

Cada arquivo também pode subir sozinho, se precisar de só um endpoint (ex.: `uvicorn appall:app --port 8002`).

## Painel de testes

Com o servidor no ar (Docker ou `uvicorn main:app`), abra **`http://127.0.0.1:8000/painel`** no navegador para testar qualquer endpoint sem precisar do front-end nem de curl: escolha o endpoint, selecione o arquivo, preencha os campos opcionais e envie. O endereço raiz (`http://127.0.0.1:8000/`) também leva ao painel.

O painel é o arquivo [back-end/painel.html](back-end/painel.html). Dá para abri-lo direto (duplo clique), mas não é recomendado: navegadores recentes pedem permissão para uma página aberta como arquivo acessar `127.0.0.1`, e se o aviso for bloqueado ou fechado a chamada falha com "Failed to fetch" sem chegar ao servidor. Aberto pelo servidor, isso não acontece.

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

Recebe um PDF ou imagem e uma lista de campos desejados, e devolve o valor de cada um. Usa o mesmo parser por posição do `/extract`, com os campos pedidos adicionados aos rótulos conhecidos. Assim um campo é encontrado com ou sem `:`, com o valor ao lado (`Nome: Fulano`) ou embaixo (como no RG), mesmo que não esteja na lista `ROTULOS`. A comparação ignora maiúsculas e acentos e tolera pequenos erros de OCR. Se o campo aparecer mais de uma vez, vale a primeira ocorrência.

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF, JPG ou PNG
- `fields`: nomes dos campos separados por vírgula (ex.: `"Nome,CPF"`)

**Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8000/extract_fields" \
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

Recebe um PDF ou imagem, faz OCR (páginas de PDF são processadas em paralelo) e estrutura o documento automaticamente, sem precisar informar os campos antecipadamente. Usa a posição de cada texto na página para:

- **Pares chave:valor**: linhas `chave: valor`, e também rótulos sem `:` (como no RG: `NOME` com o valor logo abaixo). Cada rótulo é ligado ao texto mais próximo abaixo ou à direita dele; rótulos escritos com `:` preferem o valor à direita e só usam o de baixo quando não há nada ao lado. Rótulos sem `:` precisam estar na lista `ROTULOS` de [appall.py](back-end/appall.py); a comparação tolera erros de OCR e acentos.
- **Tabelas**: uma linha com 3 ou mais títulos da lista `COLUNAS_TABELA` (ex.: `DISCIPLINA`, `ANO`, `MF`, `SF`) vira cabeçalho. Títulos de grupo por cima de outros (ex.: `CH` acima de Teórica/Prática/EAD) são ignorados. As colunas são separadas pelos espaços em branco verticais do corpo da tabela.
  - Cada linha da tabela é uma linha com texto em mais da metade das colunas. Textos de uma coluna só entre elas (ex.: nome da disciplina quebrado em duas linhas, "Docentes: ...") entram na linha mais próxima.
  - A tabela acaba num espaço vertical grande ou num salto bem maior que o espaçamento normal entre as linhas (ex.: o rodapé). Textos soltos logo depois da última linha (ex.: "Continua...") só entram se a tabela tiver células de várias linhas.
  - Tabelas com as mesmas colunas em páginas seguidas são unidas. Em colunas numéricas, um `o` sozinho (erro comum do OCR) vira `0`; células só com símbolos (`--`, `.`, `*`) ficam vazias.

Textos que não se encaixam em nenhum dos dois viram `campo_N`. Chaves repetidas ganham sufixo (`nome_2`).

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF, JPG ou PNG

**Exemplo:**
```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@historico.pdf"
```

**Resposta:**
```json
{
  "documento": "historico.pdf",
  "extraido": {
    "matrícula": "20020352",
    "nome": "FULANO DE TAL",
    "curso": "GEOGRAFIA",
    "campo_4": "UNIVERSIDADE ESTADUAL DE GOIÁS"
  },
  "tabelas": [
    {
      "colunas": ["disciplina", "ano", "mf", "che", "chc", "sf"],
      "linhas": [
        {"disciplina": "Estatística", "ano": "2002", "mf": "6,8", "che": "064", "chc": "064", "sf": "AP"}
      ]
    }
  ]
}
```

> **Limitações conhecidas:** o resultado depende de como o OCR separa as caixas de texto. Documentos tortos podem misturar linhas; colunas com títulos fora de `COLUNAS_TABELA` não são reconhecidas (acrescente-os à lista).

### `POST /split` — appsplit.py

Recebe um PDF e devolve um `.zip` com uma página por arquivo PDF. Não depende de OCR nem de Poppler — usa só o PyMuPDF.

**Parâmetros** (`multipart/form-data`):
- `file`: arquivo PDF
- `paginas` (opcional): páginas a extrair, 1-based, ex.: `"1,3,5-7"`. Vazio (padrão) extrai todas as páginas.

**Exemplo — páginas específicas:**
```bash
curl -X POST "http://127.0.0.1:8000/split" \
  -F "file=@documento.pdf" \
  -F "paginas=1,3" \
  -o paginas.zip
```

**Exemplo — PDF inteiro:**
```bash
curl -X POST "http://127.0.0.1:8000/split" \
  -F "file=@documento.pdf" \
  -o paginas.zip
```

**Resposta:** arquivo `.zip` (`application/zip`) contendo `documento_pagina_1.pdf`, `documento_pagina_3.pdf`, etc.

## Erros

Todos os endpoints retornam `{"error": "..."}` (HTTP 200) quando o formato do arquivo não é suportado (`app.py`, `appfield.py` e `appall.py` aceitam `.pdf`, `.jpg`, `.jpeg`, `.png`; `appsplit.py` aceita apenas `.pdf`).
