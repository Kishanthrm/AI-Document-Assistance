import os
import json
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests

def load_vectorstore(persist_directory="./chroma_db"):
    print("Loading ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("ChromaDB loaded successfully!")
    return vectorstore

def retrieve_chunks(vectorstore, query, k=4):
    print(f"Searching for: {query}")
    results = vectorstore.similarity_search(query, k=k)
    print(f"Retrieved {len(results)} chunks")
    return results

def generate_answer(query, chunks):
    context = "\n\n".join([
        f"Source: {chunk.metadata.get('source', 'document')}\n{chunk.page_content}"
        for chunk in chunks
    ])
    prompt = f"""You are a helpful assistant that answers questions based on the provided document.
If the answer is not in the document, say "I couldn't find this information in the uploaded document."
Always give clear and accurate answers based only on the document content.

Document Content:
{context}

Question: {query}

Answer:"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            return "Sorry I couldn't generate an answer. Please try again."
    except Exception as e:
        return f"Error connecting to AI model: {str(e)}"

def log_query(query, answer, chunks, filename="unknown"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "query": query,
        "answer": answer,
        "num_chunks": len(chunks)
    }
    log_file = "logs/queries.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    logs.append(log_entry)
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

def answer_question(query, vectorstore, filename="unknown"):
    chunks = retrieve_chunks(vectorstore, query)
    answer = generate_answer(query, chunks)
    log_query(query, answer, chunks, filename)
    return answer, chunks