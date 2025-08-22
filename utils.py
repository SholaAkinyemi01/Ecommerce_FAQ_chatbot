# utils.py

import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

def load_faq_dataset(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']

def embed_questions(questions: List[str], model) -> np.ndarray:
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index