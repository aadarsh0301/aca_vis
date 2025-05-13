import re
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

headers = {
    "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
}
# Constants for Hugging Face Inference API
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

# Send a request to Hugging Face Inference API
def query(sentences):
    payload = {"inputs": {"sentences": sentences}}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise error if request failed
    return response.json()

# Encode text using remote API
def encode(texts):
    response = query(texts)
    return np.array(response)

# Chunk the input text by sections (Sec./SEC./Section)
def chunk_by_sections(text):
    pattern = r'(Sec\.|SEC\.|Section)\s*\d+[A-Z]?\.*'
    matches = list(re.finditer(pattern, text))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[start:end].strip())
    return chunks

# Generate embeddings for text chunks
def create_embeddings(chunks):
    embeddings = {f"chunk_{i}": emb for i, emb in enumerate(encode(chunks))}
    return embeddings

# Search top-n similar chunks for a given query
def search_bills(query, embeddings, chunks, top_n=5):
    if len(query) > 3:
        query_embedding = encode([query])[0]
        similarities = {
            chunk_id: np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for chunk_id, emb in embeddings.items()
        }
        sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        temp = [(chunk_id, chunks[int(chunk_id.split('_')[1])]) for chunk_id, _ in sorted_chunks]
        return temp
    else :
        return []

# Extract bill ID from a text chunk
def extract_bill_id(chunk_text):
    match = re.search(r"Sec\.\s*(\d+)", chunk_text)
    return match.group(1) if match else None
