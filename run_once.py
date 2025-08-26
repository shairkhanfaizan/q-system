import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from rag_utils import get_embedding, model_name

# Paths
DOCS_FOLDER = "docs"
VECTORSTORE_FOLDER = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_FOLDER, "faiss_index.bin")
CHUNKS_FILE = os.path.join(VECTORSTORE_FOLDER, "chunks.npy")

# Create folder if not exists
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# Step 1: Read PDFs from docs/
def load_pdfs(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
    return texts

# Step 2: Chunk text (simple split by 500 chars)
def chunk_text(texts, chunk_size=500):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks

# Step 3: Create embeddings & FAISS index
def build_faiss(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

if __name__ == "__main__":
    print("📥 Loading PDFs...")
    texts = load_pdfs(DOCS_FOLDER)
    print(f"Loaded {len(texts)} pages.")

    print("🔹 Creating chunks...")
    chunks = chunk_text(texts)
    print(f"Created {len(chunks)} chunks.")

    print("⚙ Building FAISS index...")
    index, embeddings = build_faiss(chunks)

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)

    # Save chunks separately
    np.save(CHUNKS_FILE, np.array(chunks))

    print("✅ Vectorstore created successfully!")