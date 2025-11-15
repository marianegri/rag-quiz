import sys
sys.dont_write_bytecode = True
# embeddings.py — versão enxuta
import numpy as np

EPS = 1e-10

def make_embeddings(genai, chunks_list, batch_size=50, model="models/text-embedding-004"):
    texts = [c["text"] for c in chunks_list]
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = [genai.embed_content(model=model, content=t) for t in batch]

        for r in results:
            vec = np.array(r["embedding"], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + EPS)
            embs.append(vec)

    embeddings = np.vstack(embs)

    # metadados (sem texto)
    metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks_list]

    return embeddings, metas
