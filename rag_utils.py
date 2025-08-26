from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
import faiss
import os

# Model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# LLM pipeline for answering
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Vectorstore paths
VECTORSTORE_FOLDER = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_FOLDER, "faiss_index.bin")
CHUNKS_FILE = os.path.join(VECTORSTORE_FOLDER, "chunks.npy")

# 1. Function to get embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# 2. Load FAISS + chunks
def load_vectorstore():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("Run run_once.py first to build vectorstore.")

    index = faiss.read_index(INDEX_FILE)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    return index, chunks

# 3. Retrieval + Generation
def retrieval_and_generation(query, top_k=2):
    index, chunks = load_vectorstore()
    query_embedding = get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)

    retrieved_texts = [chunks[i] for i in indices[0]]
    context = " ".join(retrieved_texts)

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_pipeline(prompt, max_length=200)
    return result[0]["generated_text"]