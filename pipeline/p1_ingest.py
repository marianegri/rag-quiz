"""
Módulo de Ingestão de Documentos PDF
====================================

Este módulo é responsável por extrair e processar seções estruturadas de documentos PDF,
especificamente regulamentos da Ton. Realiza limpeza de texto, detecção de documentos
e divisão em seções numeradas.

Autor: Maria Negri
"""

import sys
sys.dont_write_bytecode = True

import re
import unicodedata
from typing import List, Tuple
import fitz
import pandas as pd


# ==================== CONSTANTES ====================

VALID_REGULATIONS = [
    "Regulamento Campanha ChaveTON",
    "Regulamento Ponto Ton",
    "Regulamento Ton na Mão",
    "Regulamento Pronta Entrega",
    "Regulamento Indique TapTon",
    "Regulamento Renda Ton",
    "Regulamento Renda Extra",
]

MIN_CHUNK_LENGTH = 50
PAGE_SEPARATOR = "\n\n"


# ==================== FUNÇÕES DE LIMPEZA ====================

def clean_text(text: str) -> str:
    """
    Limpa e normaliza texto extraído de PDF.
    
    Remove caracteres invisíveis, normaliza pontuação, corrige quebras de linha
    indevidas e garante formatação consistente.
    
    Args:
        text: Texto bruto extraído do PDF
        
    Returns:
        Texto limpo e normalizado
    """
    if not text:
        return ""

    try:
        # Remove caracteres invisíveis e espaços não separáveis
        text = re.sub(r'[\u200B-\u200F\uFEFF\xa0]', ' ', text)

        # Normaliza aspas, apóstrofos e travessões
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'•', '-', text)

        # Substitui quebras no meio de frases (sem ponto antes e sem número depois)
        text = re.sub(r'(?<![.:;?!])\n(?!\s*\d+\.)', ' ', text)

        # Corrige quebras duplas indevidas entre palavras
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
    
    except Exception as e:
        print(f"Aviso: Erro ao limpar texto: {e}")
        return text


def fix_double_newlines(text: str) -> str:
    """
    Corrige quebras de linha indevidas entre palavras, preservando parágrafos reais.
    
    Args:
        text: Texto a ser corrigido
        
    Returns:
        Texto com quebras corrigidas
    """
    if not isinstance(text, str):
        return text

    try:
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
    
    except Exception as e:
        print(f"Aviso: Erro ao corrigir quebras de linha: {e}")
        return text


# ==================== FUNÇÕES DE EXTRAÇÃO ====================

def _compile_document_patterns() -> List[Tuple[re.Pattern, str]]:
    """
    Compila padrões regex para detectar regulamentos válidos.
    
    Returns:
        Lista de tuplas (padrão_regex, nome_do_regulamento)
    """
    return [
        (
            re.compile(rf"(?i)REGULAMENTO[\s\n]*{re.escape(name.split('Regulamento ')[-1])}"),
            name
        )
        for name in VALID_REGULATIONS
    ]


def extract_sections_grouped(pdf_path: str) -> pd.DataFrame:
    """
    Extrai seções estruturadas de um PDF de regulamentos.
    
    Detecta múltiplos regulamentos no PDF, identifica suas seções numeradas,
    e retorna um DataFrame com metadados e conteúdo de cada seção.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Returns:
        DataFrame com colunas:
            - doc_name: Nome do regulamento
            - section_number: Número da seção
            - section_title: Título da seção
            - page_start: Página inicial da seção
            - page_end: Página final da seção
            - content: Conteúdo textual da seção
    """
    try:
        # Compila padrões de detecção de documentos
        pattern_docs = _compile_document_patterns()

        # Lê e limpa todas as páginas
        doc = fitz.open(pdf_path)
        pages = [clean_text(p.get_text("text") or "") for p in doc]
        num_pages = len(pages)

        # Detecta apenas as páginas que iniciam um novo regulamento
        page_docs = []
        for i, text in enumerate(pages, start=1):
            header = (text[:400] or "").upper()
            found_doc = None
            for pattern, name in pattern_docs:
                if pattern.search(header):
                    found_doc = name
                    break

            has_section_restart = bool(re.search(r'(?m)^\s*1\.(?!\d)', text))

            # Só marca como novo regulamento se achou o nome E reiniciou numeração
            if found_doc and has_section_restart:
                page_docs.append((i, found_doc))

        if not page_docs:
            page_docs = [(1, None)]

        # Adiciona marcador final (última página + 1)
        page_docs.append((num_pages + 1, None))

        # Junta as páginas entre cada par de marcadores e extrai seções
        rows = []
        for (start_page, doc_name), (next_page, _) in zip(page_docs, page_docs[1:]):
            # Junta todas as páginas desse regulamento em um único bloco
            text_block = PAGE_SEPARATOR.join(pages[start_page - 1: next_page - 1])

            # Cria mapa linear das posições pra calcular página inicial/final
            page_boundaries = []
            offset = 0
            for pi in range(start_page, next_page):
                page_boundaries.append((pi, offset))
                offset += len(pages[pi - 1]) + len(PAGE_SEPARATOR)

            # Detecta seções dentro do bloco
            section_matches = list(re.finditer(r'(?m)^\s*(\d{1,2})\.(?!\d)', text_block))

            # Se não achar seções, salva o bloco inteiro como uma única seção
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
                
        df = pd.DataFrame(rows)
        df["content"] = df["content"].apply(fix_double_newlines)
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
    except Exception as e:
        raise Exception(f"Erro ao processar PDF: {e}")
