from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parent
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

def encoder():
    return SentenceTransformer(MODEL, revision=REVISION, device='cpu')

def vectors(model, rows):
    titles = [row['title'] for row in rows]
    combined = [f"{row['title']}. {row['lead'][:600]}" for row in rows]
    return [model.encode(texts, normalize_embeddings=True, batch_size=16) for texts in (titles, combined)]

if __name__ == '__main__':
    rows = json.loads((ROOT/'dataset.json').read_text())
    model = encoder()
    title, full = vectors(model, rows)
    np.savez_compressed(ROOT/'embeddings.npz', ids=[row['id'] for row in rows], title=title, full=full)
    print('Encoded', len(rows), 'documents. Max tokens:', model.max_seq_length, 'Frozen parameters:',sum(p.numel() for p in model.parameters()))
