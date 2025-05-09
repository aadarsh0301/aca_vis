import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def encode(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return embeddings.numpy()

def chunk_by_sections(text):
    pattern = r'(Sec\.|SEC\.|Section)\s*\d+[A-Z]?\.*'
    matches = list(re.finditer(pattern, text))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(text) else len(text)
        chunks.append(text[start:end].strip())
    return chunks

def create_embeddings(chunks):
    embeddings = {f"chunk_{i}": emb for i, emb in enumerate(encode(chunks))}
    return embeddings

def search_bills(query, embeddings, chunks, top_n=5):
    query_embedding = encode([query])[0]
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
