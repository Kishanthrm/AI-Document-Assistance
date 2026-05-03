import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {file_path}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks

def add_metadata(chunks, source_name="document"):
    for chunk in chunks:
        if not chunk.metadata.get("source"):
            chunk.metadata["source"] = source_name
    return chunks

def store_in_chroma(chunks, persist_directory="./chroma_db",
                    collection_name="documents"):
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore