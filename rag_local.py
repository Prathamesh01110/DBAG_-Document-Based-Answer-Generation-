import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === SETTINGS ===
DOCUMENT_PATH = "SJCEM_Document.txt"  # Replace with your file
CHUNK_SIZE = 200  # words
MODEL = SentenceTransformer("./llama.cpp/models/all-MiniLM-L6-v2")
LLAMA_SERVER_URL = "http://localhost:8080/completion"

# === LOAD + CHUNK DOCUMENT ===
with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Simple chunking
words = text.split()
chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# === EMBEDDINGS ===
chunk_embeddings = MODEL.encode(chunks)
print(f"Indexed {len(chunks)} chunks from the document.")

# === ASK USER ===
question = input("Ask a question about the document: ")
question_embedding = MODEL.encode([question])

# === FIND BEST MATCH ===
similarities = cosine_similarity(question_embedding, chunk_embeddings)
top_k = 3
top_k_idx = similarities[0].argsort()[-top_k:][::-1]
context = "\n\n".join([chunks[i] for i in top_k_idx])


# === PREPARE PROMPT ===
prompt = f"""You are an assistant. Use the following context from a document to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

# === SEND TO LOCAL LLM ===
response = requests.post(LLAMA_SERVER_URL, json={
    "prompt": prompt,
    "n_predict": 150,
    "temperature": 0.7
})

print("\n--- Answer ---\n")
print(response.json()["content"])
