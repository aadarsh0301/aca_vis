import re
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_by_sections(text):
    pattern = r'(Sec\.|SEC\.|Section)\s*\d+[A-Z]?\.*'
    matches = list(re.finditer(pattern, text))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[start:end].strip())
    return chunks

def create_embeddings(chunks):
    embeddings = {f"chunk_{i}": model.encode(chunk).tolist() for i, chunk in enumerate(chunks)}
    return embeddings

def search_bills(query, embeddings, chunks, top_n=5):
    query_embedding = model.encode(query)
    similarities = {
        chunk_id: np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        for chunk_id, emb in embeddings.items()
    }
    sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    temp = [(chunk_id, chunks[int(chunk_id.split('_')[1])]) for chunk_id, _ in sorted_chunks]
    return temp

def extract_bill_id(chunk_text):
    match = re.search(r"Sec\.\s*(\d+)", chunk_text)
    return match.group(1) if match else None
