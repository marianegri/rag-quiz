"""
Módulo de Chunking de Texto
===========================

Este módulo é responsável por dividir textos longos em chunks (pedaços) menores
com sobreposição, facilitando o processamento e embedding de documentos.

Autor: Maria Negri
"""

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path
from typing import List, Dict, Tuple
from uuid import uuid4


# ==================== CONSTANTES ====================

DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 200
MIN_CHUNK_LENGTH = 50

def chunk_text(
    text: str, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP
) -> List[Tuple[int, int, int, str]]:
    """
    Divide texto longo em chunks com sobreposição.
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho máximo de cada chunk em caracteres
        overlap: Número de caracteres de sobreposição entre chunks consecutivos
        
    Returns:
        Lista de tuplas (chunk_index, char_start, char_end, chunk_text)
    """
    if not isinstance(text, str) or not text:
        return []

    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    idx = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        
        # Ignora pedaços muito curtos
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append((idx, start, end, chunk))
            idx += 1
            
        if end == text_length:
            break
            
        start = end - overlap
        
    return chunks


def df_to_chunks_list(
    df, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP
) -> List[Dict]:
    """
    Converte DataFrame de seções em lista de chunks com metadados.
    
    Args:
        df: DataFrame com colunas ['doc_name', 'section_number', 'section_title', 
            'page_start', 'page_end', 'content']
        chunk_size: Tamanho máximo de cada chunk em caracteres
        overlap: Número de caracteres de sobreposição entre chunks
        
    Returns:
        Lista de dicionários contendo metadados e texto de cada chunk
    """
    out = []
    for _, row in df.iterrows():
        title = row.get('section_title') or row.get('doc_name') or ""
        text = row.get('content') or ""
        for idx, s, e, ch_text in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            out.append({
                "id": str(uuid4()),
                "doc_name": row.get('doc_name'),
                "section_number": row.get('section_number'),
                "section_title": title,
                "page_range": f"{row.get('page_start')}-{row.get('page_end')}",
                "chunk_index": idx,
                "char_start": int(s),
                "char_end": int(e),
                "text": ch_text
            })
    return out


def save_chunks_json(chunks_list: List[Dict], out_path: str) -> None:
    """
    Salva lista de chunks em arquivo JSON.
    
    Args:
        chunks_list: Lista de dicionários com chunks e metadados
        out_path: Caminho do arquivo JSON de saída
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(chunks_list, ensure_ascii=False, indent=2), encoding="utf-8")
