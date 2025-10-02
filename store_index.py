# store_index.py

from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Please set PINECONE_API_KEY in your .env file")

# -----------------------------
# 2️⃣ Configure Pinecone
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Make sure this matches embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# -----------------------------
# 3️⃣ Configure embeddings
# -----------------------------
embedding = download_embeddings()  # From helper.py, HuggingFace embeddings

# -----------------------------
# 4️⃣ Load and process PDFs
# -----------------------------
data_dir = r"C:\Users\adity\Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS\data"
pdf_documents = load_pdf_files(data_dir)
minimal_docs = filter_to_minimal_docs(pdf_documents)
texts_chunk = text_split(minimal_docs)

# -----------------------------
# 5️⃣ Store documents in Pinecone
# -----------------------------
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

# Load existing index if needed
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

print("✅ Documents embedded and stored in Pinecone successfully!")
