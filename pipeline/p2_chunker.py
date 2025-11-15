import sys
sys.dont_write_bytecode = True
# mcq_toolkit/chunker.py
from typing import List, Dict
from uuid import uuid4
from pathlib import Path
import json

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[tuple]:
    """
    Mesma lógica do seu notebook (desafio.py).
    Retorna lista de tuplas: (chunk_index, char_start, char_end, chunk_text)
    """
    if not isinstance(text, str) or not text:
        return []

    chunks = []
    start = 0
    L = len(text)
    idx = 0
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if len(chunk) >= 50:  # ignora pedaços muito curtos
            chunks.append((idx, start, end, chunk))
            idx += 1
        if end == L:
            break
        start = end - overlap
    return chunks


def df_to_chunks_list(df, chunk_size: int = 900, overlap: int = 200) -> List[Dict]:
    """
    Recebe o DataFrame com colunas esperadas:
      ['doc_name','section_number','section_title','page_start','page_end','content']
    (como o retorno de extract_sections_grouped).
    Retorna lista de dicionários (metadados + 'text') pronta para salvar/indexar.
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


def save_chunks_json(chunks_list: List[Dict], out_path: str):
    """
    Helper: salva a lista completa (com 'text') em JSON (utf-8).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(chunks_list, ensure_ascii=False, indent=2), encoding="utf-8")
