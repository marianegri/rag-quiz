"""
Módulo de Geração de Embeddings
==================================

Este módulo é responsável por gerar embeddings vetoriais de textos
usando a API do Google Generative AI, com normalização L2.

Autor: Maria Negri
"""

import sys
sys.dont_write_bytecode = True

from typing import List, Dict, Tuple, Any
import numpy as np


# ==================== CONSTANTES ====================

EPS = 1e-10  # Epsilon para evitar divisão por zero na normalização
DEFAULT_BATCH_SIZE = 50

def make_embeddings(
    genai,
    chunks_list: List[Dict[str, Any]],
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Gera embeddings normalizados para uma lista de chunks de texto.
    
    Args:
        genai: Cliente da API Google Generative AI
        chunks_list: Lista de dicionários contendo chunks com chave 'text'
        batch_size: Número de textos a processar por lote
        model: Nome do modelo de embedding a usar
        
    Returns:
        Tupla contendo:
            - embeddings: Array numpy (N, D) com vetores normalizados L2
            - metas: Lista de dicionários com metadados (sem campo 'text')
    """
    if not chunks_list:
        raise ValueError("chunks_list não pode estar vazia")
    
    try:
        texts = [c["text"] for c in chunks_list]
        embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = [genai.embed_content(model=model, content=t) for t in batch]

            for r in results:
                vec = np.array(r["embedding"], dtype=np.float32)
                # Normalização L2 para busca por produto interno
                vec = vec / (np.linalg.norm(vec) + EPS)
                embs.append(vec)

        embeddings = np.vstack(embs)

        # Extrai metadados (remove campo 'text' para economizar memória)
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks_list]

        return embeddings, metas
        
    except KeyError as e:
        raise ValueError(f"chunks_list deve conter chave 'text': {e}")
    except Exception as e:
        raise Exception(f"Erro ao gerar embeddings: {e}")
