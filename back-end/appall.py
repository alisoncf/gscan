from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import re
import bisect
import difflib
import unicodedata
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

from comum import POPPLER_PATH, caminho_temporario, paddle_predict, salvar_upload

router = APIRouter()

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
    temp_path = caminho_temporario(".png")
    img.save(temp_path)
    try:
        result = paddle_predict(temp_path)
    finally:
        os.remove(temp_path)
    items = []
    for page in result:
        for text, box in zip(page["rec_texts"], page["rec_boxes"]):
            items.append((text.strip(), [float(v) for v in box]))
    return items

def ocr_pdf(pdf_path, dpi=200):
    """Processa PDF multipágina em paralelo; devolve uma lista de itens por página"""
    #pages = convert_from_path(pdf_path, dpi=dpi)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    def process_page(page):
        preprocessed = preprocess_image(page)
        return ocr_image(preprocessed)

    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(process_page, pages))

FORMATOS_ACEITOS = [".jpg", ".jpeg", ".png", ".pdf"]

async def ler_documento(file: UploadFile):
    """OCR do upload (PDF ou imagem); devolve as caixas de cada página"""
    ext = os.path.splitext(file.filename)[1].lower()
    temp_file = await salvar_upload(file)
    try:
        if ext == ".pdf":
            return ocr_pdf(temp_file, dpi=200)
        with Image.open(temp_file) as img:
            return [ocr_image(preprocess_image(img))]
    finally:
        os.remove(temp_file)

def separar_tabelas(pages):
    """Tira as tabelas de cada página; devolve (tabelas, caixas que sobraram por página)"""
    tabelas = []
    restantes = []
    for items in pages:
        tabelas_pagina, sobra = extrair_tabelas(items)
        tabelas.extend(tabelas_pagina)
        restantes.append(sobra)
    return juntar_tabelas(tabelas), restantes

@router.post("/extract")
async def extract(file: UploadFile = File(...)):
    if os.path.splitext(file.filename)[1].lower() not in FORMATOS_ACEITOS:
        return {"error": "Formato não suportado. Use PDF ou imagem."}

    pages = await ler_documento(file)
    # tabelas saem primeiro; o que sobra de cada página vai para o parser chave:valor
    tabelas, restantes = separar_tabelas(pages)

    return {
        "documento": file.filename,
        "extraido": parse_key_values(restantes),
        "tabelas": tabelas,
    }

# App próprio, para rodar só este endpoint: uvicorn appall:app --port 8002
app = FastAPI(title="GScan", description="API OCR otimizada para PDF e imagens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

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

# Rótulos que podem ter vários valores empilhados (ex.: FILIACAO: pai embaixo, mãe logo abaixo).
# Os valores extras saem como <chave>_2, <chave>_3...
ROTULOS_MULTIPLOS = ["FILIACAO"]

# Títulos de coluna de tabelas (ex.: notas de histórico escolar). Sem acento, em maiúsculas.
# Uma linha com 3 ou mais destes títulos é tratada como cabeçalho de tabela.
COLUNAS_TABELA = [
    "DISCIPLINA", "COMPONENTE CURRICULAR", "CODIGO", "TURMA", "PROFESSOR",
    "ANO", "SEMESTRE", "PERIODO", "MF", "MEDIA", "MEDIA FINAL", "NOTA",
    "FREQUENCIA", "FREQ", "FALTAS", "CH", "CHE", "CHC", "CARGA HORARIA",
    "CREDITOS", "SF", "SITUACAO", "RESULTADO",
]

def normalizar(texto):
    """Maiúsculas, sem acento e sem pontuação"""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t).upper()
    return " ".join(t.split())

def vocabulario(termos):
    """Prepara uma lista de termos para busca com casar()"""
    compactos = {t.replace(" ", ""): t for t in termos}
    # termos curtos (RG, UF, MF...) só valem com acerto exato, para não pegar lixo do OCR
    aproximaveis = [c for c in compactos if len(c) >= 4]
    return compactos, aproximaveis

def casar(texto, vocab):
    """Devolve o termo do vocabulário que o texto representa (tolera erros do OCR) ou None"""
    compactos, aproximaveis = vocab
    compacto = normalizar(texto).replace(" ", "")
    if compacto in compactos:
        return compactos[compacto]
    if len(compacto) < 4:
        return None
    # 0.85: aceita "NATUIRALIDADE", mas não "DO CURSO" como "CURSO"
    match = difflib.get_close_matches(compacto, aproximaveis, n=1, cutoff=0.85)
    return compactos[match[0]] if match else None

_VOCAB_ROTULOS = vocabulario(ROTULOS)
_VOCAB_COLUNAS = vocabulario(COLUNAS_TABELA)

def para_chave(termo):
    """'DATA DE EXPEDICAO' -> 'data_de_expedicao'"""
    return termo.lower().replace(" ", "_")

def agrupar_linhas(items):
    """Agrupa índices de caixas que estão na mesma altura da página, de cima para baixo"""
    ordem = sorted(range(len(items)), key=lambda i: (items[i][1][1] + items[i][1][3]) / 2)
    linhas = []
    for i in ordem:
        _, y1, _, y2 = items[i][1]
        centro = (y1 + y2) / 2
        if linhas:
            ultima = linhas[-1]
            centro_linha = sum((items[j][1][1] + items[j][1][3]) / 2 for j in ultima) / len(ultima)
            altura = sum(items[j][1][3] - items[j][1][1] for j in ultima) / len(ultima)
            if abs(centro - centro_linha) <= altura * 0.5:
                ultima.append(i)
                continue
        linhas.append([i])
    return linhas

def extrair_tabelas(items):
    """Encontra tabelas pelo cabeçalho e devolve (tabelas, caixas que não fazem parte delas)"""
    items = [(t, b) for t, b in items if re.search(r"\w", t)]
    linhas = agrupar_linhas(items)
    usados = set()
    tabelas = []
    k = 0
    while k < len(linhas):
        # cabeçalho: linha com pelo menos 3 títulos de coluna conhecidos
        cabecalho = [(i, casar(items[i][0], _VOCAB_COLUNAS)) for i in linhas[k]]
        cabecalho = sorted([(i, c) for i, c in cabecalho if c], key=lambda p: items[p[0]][1][0])
        k += 1
        if len(cabecalho) < 3:
            continue
        usados.update(i for i, _ in cabecalho)
        nomes = [para_chave(c) for _, c in cabecalho]
        caixas = [items[i][1] for i, _ in cabecalho]
        # fronteira entre colunas: meio do espaço entre um título e o seguinte
        fronteiras = [(a[2] + b[0]) / 2 for a, b in zip(caixas, caixas[1:])]
        altura = sum(b[3] - b[1] for b in caixas) / len(caixas)
        fundo = max(b[3] for b in caixas)

        registros = []
        while k < len(linhas):
            linha = sorted(linhas[k], key=lambda i: items[i][1][0])
            topo = min(items[i][1][1] for i in linha)
            if topo - fundo > altura * 3:
                break  # espaço grande: a tabela acabou
            celulas = {}
            for i in linha:
                x1, _, x2, _ = items[i][1]
                coluna = nomes[bisect.bisect(fronteiras, (x1 + x2) / 2)]
                celulas.setdefault(coluna, []).append(items[i][0])
            if len(celulas) < 2:
                break  # texto numa coluna só (ex.: "Continua..."): a tabela acabou
            registros.append({n: " ".join(celulas.get(n, [])) for n in nomes})
            usados.update(linha)
            fundo = max(items[i][1][3] for i in linha)
            k += 1
        tabelas.append({"colunas": nomes, "linhas": registros})

    restantes = [item for i, item in enumerate(items) if i not in usados]
    return tabelas, restantes

def juntar_tabelas(tabelas):
    """Junta tabelas seguidas com as mesmas colunas (tabela que continua na página seguinte)"""
    juntas = []
    for t in tabelas:
        if juntas and juntas[-1]["colunas"] == t["colunas"]:
            juntas[-1]["linhas"].extend(t["linhas"])
        else:
            juntas.append(t)
    return juntas

def dividir_rotulos_grudados(texto, vocab=_VOCAB_ROTULOS):
    """Posições onde começam rótulos grudados no meio de uma caixa do OCR.
    Ex.: "20020352 Nome:" -> [9]; "Local: Água Grande - São Nacionalidade:" -> [25]"""
    cortes = []
    for m in SEPARADOR.finditer(texto):
        inicio_trecho = cortes[-1] if cortes else 0
        palavras = [p for p in re.finditer(r"\S+", texto[:m.start()]) if p.start() >= inicio_trecho]
        inicio = None
        # 1º: rótulo conhecido exato, do mais longo ao mais curto ("Data de nascimento" antes de "nascimento")
        for n in range(min(4, len(palavras)), 0, -1):
            candidato = texto[palavras[-n].start():m.start()]
            if normalizar(candidato).replace(" ", "") in vocab[0]:
                inicio = palavras[-n].start()
                break
        # 2º: rótulo aproximado, do mais curto ao mais longo ("Nacionalidade" antes de "São Nacionalidade")
        if inicio is None:
            for n in range(1, min(4, len(palavras)) + 1):
                if casar(texto[palavras[-n].start():m.start()], vocab):
                    inicio = palavras[-n].start()
                    break
        # 3º: rótulo desconhecido, mas o que vem antes tem número, então é valor ("20020352 Fax:")
        if inicio is None and palavras and re.search(r"\d", texto[inicio_trecho:palavras[-1].start()]):
            inicio = palavras[-1].start()
        # só corta se sobrar texto antes do rótulo
        if inicio and texto[inicio_trecho:inicio].strip():
            cortes.append(inicio)
    return cortes

def dividir_caixas(items, vocab=_VOCAB_ROTULOS):
    """Separa caixas com rótulos grudados, estimando a posição de cada pedaço pelo nº de caracteres"""
    resultado = []
    for texto, (x1, y1, x2, y2) in items:
        limites = [0] + dividir_rotulos_grudados(texto, vocab) + [len(texto)]
        largura_char = (x2 - x1) / max(len(texto), 1)
        for a, b in zip(limites, limites[1:]):
            pedaco = texto[a:b].strip()
            if pedaco:
                resultado.append((pedaco, [x1 + a * largura_char, y1, x1 + b * largura_char, y2]))
    return resultado

def distancia_valor(rotulo, candidato, prefere_direita=False):
    """Distância do rótulo até uma caixa logo abaixo ou logo à direita dele; None se não for vizinha"""
    rx1, ry1, rx2, ry2 = rotulo
    cx1, cy1, cx2, cy2 = candidato
    altura = max(ry2 - ry1, 1)
    folga = altura * 0.5
    # abaixo: começa depois do rótulo e se sobrepõe a ele na horizontal
    if cy1 >= ry2 - folga and min(rx2, cx2) > max(rx1, cx1):
        d = cy1 - ry2
        if d > altura * 3:
            return None
        # em "Rótulo:" o valor costuma vir à direita; abaixo só se não houver nada perto à direita
        return max(d, 0) + (altura * 15 if prefere_direita else 0)
    # à direita: na mesma altura do rótulo
    if ry1 <= (cy1 + cy2) / 2 <= ry2 and cx1 >= rx2 - folga:
        d = cx1 - rx2
        return max(d, 0) if d <= altura * 15 else None
    return None

_MULTIPLOS = {r.replace(" ", "") for r in ROTULOS_MULTIPLOS}

def proximo_empilhado(items, box, ocupados):
    """Índice da caixa livre logo abaixo de box e alinhada à esquerda com ela; None se não houver"""
    x1, _, _, y2 = box
    altura = max(box[3] - box[1], 1)
    candidatos = [
        j for j, (_, (cx1, cy1, _, _)) in enumerate(items)
        if j not in ocupados
        and y2 - altura * 0.5 <= cy1 <= y2 + altura  # na linha seguinte, sem espaço grande
        and abs(cx1 - x1) <= altura * 2              # começa alinhada com o valor de cima
    ]
    return min(candidatos, key=lambda j: items[j][1][1], default=None)

def parse_key_values(pages, vocab=_VOCAB_ROTULOS):
    """Monta o dicionário chave:valor a partir das caixas do OCR de cada página.
    vocab: rótulos reconhecidos mesmo sem ':' (padrão: ROTULOS)"""
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
        items = dividir_caixas([(t, b) for t, b in items if re.search(r"\w", t)], vocab)
        diretos = {}  # índice -> (chave, valor) de linhas "chave: valor"
        rotulos = {}  # índice -> chave de rótulos que esperam valor em outra caixa
        com_dois_pontos = set()  # rótulos escritos "Rótulo:"
        for i, (texto, _) in enumerate(items):
            m = SEPARADOR.search(texto)
            chave = texto[:m.start()].strip().lower() if m else ""
            if chave:
                valor = texto[m.end():].strip()
                if valor:
                    diretos[i] = (chave, valor)
                else:
                    rotulos[i] = chave
                    com_dois_pontos.add(i)
                continue
            rotulo = casar(texto, vocab)
            if rotulo:
                rotulos[i] = para_chave(rotulo)

        # liga cada rótulo à caixa livre mais próxima, começando pelos pares mais próximos
        pares = []
        for i in rotulos:
            for j, (_, box) in enumerate(items):
                if j in rotulos or j in diretos:
                    continue
                d = distancia_valor(items[i][1], box, prefere_direita=i in com_dois_pontos)
                if d is not None:
                    pares.append((d, i, j))
        valor_de = {}
        usados = set()
        for _, i, j in sorted(pares):
            if i not in valor_de and j not in usados:
                valor_de[i] = j
                usados.add(j)

        # valores extras empilhados logo abaixo do primeiro valor (só em ROTULOS_MULTIPLOS)
        extras = {}
        for i, j in valor_de.items():
            if normalizar(rotulos[i]).replace(" ", "") not in _MULTIPLOS:
                continue
            extras[i] = []
            atual = items[j][1]
            while True:
                j = proximo_empilhado(items, atual, ocupados=usados | set(rotulos) | set(diretos))
                if j is None:
                    break
                extras[i].append(items[j][0])
                usados.add(j)
                atual = items[j][1]

        for i, (texto, _) in enumerate(items):
            if i in diretos:
                add(*diretos[i])
            elif i in rotulos:
                add(rotulos[i], items[valor_de[i]][0] if i in valor_de else "")
                for extra in extras.get(i, []):
                    add(rotulos[i], extra)
            elif i not in usados:
                add(f"campo_{len(data)+1}", texto)
    return data
