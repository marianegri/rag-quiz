"""
Módulo de Recuperação (Retriever)
====================================

Este módulo implementa busca semântica usando FAISS e embeddings.
Permite buscar chunks relevantes e construir contextos para LLMs.

Autor: Maria Negri
"""

import sys
sys.dont_write_bytecode = True

from typing import List, Dict, Any, Optional
import numpy as np


# ==================== CONSTANTES ====================

EPS = 1e-10  # Epsilon para evitar divisão por zero
DEFAULT_TOP_K = 5
DEFAULT_CHAR_LIMIT = 3000

def search_query(
    query: str,
    index,
    metas: List[Dict[str, Any]],
    embed_model: str,
    chunks_list: Optional[List[Dict[str, Any]]] = None,
    genai_client = None,
    k: int = DEFAULT_TOP_K,
    eps: float = EPS,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Busca chunks mais relevantes para uma query usando busca semântica.
    
    Args:
        query: Texto da consulta
        index: Índice FAISS previamente construído
        metas: Lista de metadados correspondentes aos vetores no índice
        chunks_list: Lista opcional com texto completo dos chunks
        genai_client: Cliente da API Google Generative AI
        k: Número de resultados a retornar
        embed_model: Modelo de embedding a usar
        eps: Epsilon para normalização
        verbose: Se True, imprime informações de debug
        
    Returns:
        Lista de dicionários com metadados, scores e texto dos chunks encontrados
    """

    if genai_client is None:
        raise ValueError("genai_client é obrigatório para gerar embedding da query")

    try:
        # Gera embedding da query
        r = genai_client.embed_content(model=embed_model, content=query)
        q = np.array(r["embedding"], dtype=np.float32)
        # Normaliza para busca por produto interno
        q = q / (np.linalg.norm(q) + eps)
        q = q.reshape(1, -1)

        # Busca no índice FAISS
        D_scores, I_indices = index.search(q, k)
        
        # Monta resultados com metadados e scores
        results = []
        for score, idx in zip(D_scores[0], I_indices[0]):
            idx = int(idx)
            meta = metas[idx].copy()
            chunk_text = None
            if chunks_list is not None and len(chunks_list) == len(metas):
                chunk_text = chunks_list[idx].get("text")
            meta.update({
                "_index": idx,
                "_score": float(score),
                "text": chunk_text
            })
            results.append(meta)

        if verbose:
            print(f"search_query: query={query[:60]!r} -> returned {len(results)} hits")

        return results
    
    except Exception as e:
        raise Exception(f"Erro na busca: {e}")


def build_context_from_results(
    results: List[Dict[str, Any]],
    char_limit: int = DEFAULT_CHAR_LIMIT,
    include_meta: bool = True
) -> str:
    """
    Constrói contexto textual a partir dos resultados de busca.
    
    Args:
        results: Lista de resultados do search_query
        char_limit: Limite máximo de caracteres para o contexto
        include_meta: Se True, inclui metadados de cada chunk no contexto
        
    Returns:
        String formatada com o contexto agregado
    """
    seen = set()
    parts = []
    total = 0

    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)

        header = ""
        if include_meta:
            header = (
                f"[Fonte: {r.get('doc_name','?')} | Páginas: {r.get('page_range','?')} "
                f"| Título: {r.get('section_title','?')} | score:{r.get('_score',0):.3f}]\n"
            )

        remaining = char_limit - total
        if remaining <= 0:
            break

        max_for_this = max(80, remaining - 50)
        if len(text) > max_for_this:
            cut = text[:max_for_this]
            if '.' in cut:
                # tenta cortar no fim da última sentença dentro do corte
                cut = cut.rsplit('.', 1)[0].strip() + '.'
        else:
            cut = text

        part = header + cut + "\n\n---\n\n"
        parts.append(part)
        total += len(part)

    context_text = "".join(parts).strip()
    return context_text


def build_queries_from_metas(
    metas: List[Dict[str, Any]],
    n: int = 10
) -> List[str]:
    """
    Gera queries para MCQ baseadas em títulos de seções.
    
    Args:
        metas: Lista de metadados dos chunks
        n: Número máximo de queries a gerar
        
    Returns:
        Lista de strings formatadas como prompts para geração de perguntas
    """
    titles = [m.get("section_title") for m in metas if m.get("section_title")]
    uniq = []
    for t in titles:
        if t and t not in uniq:
            uniq.append(t)

    queries: List[str] = []
    for t in uniq:
        if len(queries) >= n:
            break
        queries.append(f"Crie uma pergunta sobre: {t}")

    queries = queries[:n]

    return queries
