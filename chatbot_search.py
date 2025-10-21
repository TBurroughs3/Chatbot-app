import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Set page config
st.set_page_config(page_title="Clinical Document Search", page_icon="🩺", layout="wide")

# Title and instructions
st.title("🩺 Clinical Document Search Assistant")
st.markdown("""
Welcome! Enter a question below to search clinical documents using semantic similarity.
""")

# Load FAISS index and document sources
index = faiss.read_index("clinical_documents.index")
with open("clinical_documents_sources.txt", "r") as f:
    sources = [line.strip() for line in f.readlines()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input from user
question = st.text_input("Ask your question:")

if question:
    query_embedding = model.encode([question], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=3)

    st.subheader("🔍 Top Matching Documents:")
    for idx in I[0]:
        st.write(f"- {sources[idx]}")
