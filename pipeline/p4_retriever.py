
import sys
sys.dont_write_bytecode = True
# 4_retriever.py
import numpy as np
from typing import List, Dict, Any, Optional

EPS = 1e-10

def search_query(
    query: str,
    index,
    metas: List[Dict[str, Any]],
    chunks_list: Optional[List[Dict[str, Any]]] = None,
    genai_client = None,
    k: int = 5,
    embed_model: str = "models/text-embedding-004",
    eps: float = EPS,
    verbose: bool = False,
) -> List[Dict[str, Any]]:

    if genai_client is None:
        raise ValueError("genai_client is required to embed the query")

    r = genai_client.embed_content(model=embed_model, content=query)
    q = np.array(r["embedding"], dtype=np.float32)
    q = q / (np.linalg.norm(q) + eps)
    q = q.reshape(1, -1)

    D_scores, I_indices = index.search(q, k)
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


def build_context_from_results(
    results: List[Dict[str, Any]],
    char_limit: int = 3000,
    include_meta: bool = True
) -> str:
    """
    Monta um contexto string a partir de results (retorno do search_query).
    Evita duplicatas simples (primeiros 120 chars) e tenta truncar por sentença.
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
    Cria até n queries baseadas em metas['section_title'].
    Se não houver títulos suficientes, usa trechos de chunks_list como fallback.
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
