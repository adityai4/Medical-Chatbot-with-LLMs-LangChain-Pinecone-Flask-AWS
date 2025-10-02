# helper.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# -----------------------------
# 1️⃣ Load PDFs from a folder
# -----------------------------
def load_pdf_files(data_dir: str) -> List[Document]:
    """
    Load all PDF files from a directory and return as a list of Document objects.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# -----------------------------
# 2️⃣ Filter documents to minimal info
# -----------------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Return a new list of Document objects containing only 'source' in metadata
    and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# -----------------------------
# 3️⃣ Split documents into chunks
# -----------------------------
def text_split(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

# -----------------------------
# 4️⃣ Optional: Download HuggingFace embeddings
# -----------------------------
def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embeddings object. Optional: use if not using Gemini embeddings.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
