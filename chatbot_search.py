
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load FAISS index and document list
index = faiss.read_index("clinical_documents.index")
with open("clinical_documents_sources.txt", "r") as f:
    sources = [line.strip() for line in f.readlines()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ask a question
question = input("Ask your question: ")
query_embedding = model.encode([question], convert_to_numpy=True)

# Search the index
D, I = index.search(query_embedding, k=3)

# Show results
print("
Top matching documents:")
for idx in I[0]:
    print(f"- {sources[idx]}")
