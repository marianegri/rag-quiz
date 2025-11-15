import sys
sys.dont_write_bytecode = True
# mcq_toolkit/ingest.py
import re
import unicodedata
import fitz
import pandas as pd

def clean_text(text: str) -> str:
    """
    Limpa caracteres, normaliza e remove quebras desnecessárias.
    """
    if not text:
        return ""

    # Remove caracteres invisíveis e espaços não separáveis
    text = re.sub(r'[\u200B-\u200F\uFEFF\xa0]', ' ', text)

    # Normaliza aspas, apóstrofos e travessões
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'•', '-', text)

    # Substitui quebras no meio de frases (sem ponto antes e sem número depois)
    text = re.sub(r'(?<![.:;?!])\n(?!\s*\d+\.)', ' ', text)

    # Corrige quebras duplas indevidas entre palavras (ex: "informações\n\nprovidas")
    text = re.sub(r'([a-zá-úÀ-Ú])\n{2,}([a-zá-úÀ-Ú])', r'\1 \2', text, flags=re.IGNORECASE)

    # Remove espaços múltiplos/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Converte 3+ quebras em apenas 2 (mantém blocos reais)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove espaços antes/depois de quebras
    text = re.sub(r' *\n *', '\n', text)

    # Normaliza Unicode 
    text = unicodedata.normalize('NFC', text).strip()

    return text


def extract_sections_grouped(pdf_path):
    """
    Lê PDF, detecta blocos que parecem regulamentos e divide em seções numeradas.
    Retorna um pandas.DataFrame com colunas:
      doc_name, section_number, section_title, page_start, page_end, content
    """
    # Lista de regulamentos válidos (ajuste se necessário)
    valid_docs = [
        "Regulamento Campanha ChaveTON",
        "Regulamento Ponto Ton",
        "Regulamento Ton na Mão",
        "Regulamento Pronta Entrega",
        "Regulamento Indique TapTon",
        "Regulamento Renda Ton",
        "Regulamento Renda Extra",
    ]

    # Regex para detectar regulamentos (case-insensitive)
    pattern_docs = [
        re.compile(rf"(?i)REGULAMENTO[\s\n]*{re.escape(name.split('Regulamento ')[-1])}")
        for name in valid_docs
    ]

    # ---- Lê e limpa todas as páginas ----
    doc = fitz.open(pdf_path)
    pages = [clean_text(p.get_text("text") or "") for p in doc]
    num_pages = len(pages)

    # ---- Detecta apenas as páginas que iniciam um novo regulamento ----
    page_docs = []
    for i, text in enumerate(pages, start=1):
        header = (text[:400] or "").upper()
        found_doc = None
        for pattern, name in zip(pattern_docs, valid_docs):
            if pattern.search(header):
                found_doc = name
                break

        has_section_restart = bool(re.search(r'(?m)^\s*1\.(?!\d)', text))

        # Só marca como novo regulamento se achou o nome E reiniciou numeração
        if found_doc and has_section_restart:
            page_docs.append((i, found_doc))

    if not page_docs:
        # fallback: tentar considerar todo o documento como um bloco sem nome
        page_docs = [(1, None)]

    # Adiciona marcador final (última página + 1)
    page_docs.append((num_pages + 1, None))

    # ---- Junta as páginas entre cada par de marcadores e extrai seções ----
    rows = []
    for (start_page, doc_name), (next_page, _) in zip(page_docs, page_docs[1:]):
        # Junta todas as páginas desse regulamento em um único bloco
        text_block = "\n\n".join(pages[start_page - 1: next_page - 1])

        # Cria mapa linear das posições pra calcular página inicial/final
        page_boundaries = []
        offset = 0
        for pi in range(start_page, next_page):
            page_boundaries.append((pi, offset))
            offset += len(pages[pi - 1]) + 2  # +2 por conta do "\n\n" usado ao juntar

        # --- Detecta seções dentro do bloco (capítulos numerados como 1., 2., ..., 12.)
        section_matches = list(re.finditer(r'(?m)^\s*(\d{1,2})\.(?!\d)', text_block))

        # Se não achar seções, salva o bloco inteiro como uma única seção sem número
        if not section_matches:
            rows.append({
                "doc_name": doc_name,
                "section_number": "",
                "section_title": "",
                "page_start": start_page,
                "page_end": next_page - 1,
                "content": text_block.strip()
            })
            continue

        for idx, m in enumerate(section_matches):
            section_number = m.group(1)
            start_pos = m.end()
            end_pos = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(text_block)

            # Título e conteúdo
            after = text_block[start_pos:].lstrip()
            next_line = after.split("\n", 1)[0].strip()
            section_title = next_line if next_line and not re.match(r"^\d", next_line) else ""
            section_text = text_block[start_pos:end_pos].strip()
            if section_title:
                section_text = re.sub(rf"^{re.escape(section_title)}[\s:–-]*", "", section_text).strip()

            # Calcula página inicial e final baseadas no mapa
            page_start = start_page
            page_end = next_page - 1
            for j in range(len(page_boundaries) - 1):
                if page_boundaries[j][1] <= start_pos < page_boundaries[j + 1][1]:
                    page_start = page_boundaries[j][0]
                if page_boundaries[j][1] <= end_pos < page_boundaries[j + 1][1]:
                    page_end = page_boundaries[j][0]
                    break

            rows.append({
                "doc_name": doc_name,
                "section_number": section_number,
                "section_title": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "content": section_text
            })

    return pd.DataFrame(rows)


def fix_double_newlines(text: str) -> str:
    """
    Corrige quebras de linha indevidas entre palavras, sem remover parágrafos reais.
    Exemplo: 'informações\\n\\nprovidas' -> 'informações providas'
    """
    if not isinstance(text, str):
        return text

    # Substitui múltiplas quebras entre letras por um espaço
    text = re.sub(
        r'([a-zá-úÀ-Ú])[\n\r\u2028\u2029]+\s*[\n\r\u2028\u2029]+([a-zá-úÀ-Ú])',
        r'\1 \2',
        text,
        flags=re.IGNORECASE,
    )

    # Remove espaços redundantes
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove espaços antes/depois de quebras verdadeiras
    text = re.sub(r' *\n+ *', '\n', text)
    # Normaliza Unicode
    text = unicodedata.normalize('NFC', text)
    return text.strip()
