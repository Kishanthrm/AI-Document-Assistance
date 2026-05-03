import os
import streamlit as st
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import requests
from datetime import datetime

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.caption("Upload any document and ask questions about it!")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

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
            return "Sorry, I couldn't generate an answer. Please try again."
    except Exception as e:
        return f"Error connecting to AI model: {str(e)}"

def log_query(query, answer, filename):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "query": query,
        "answer": answer
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

with st.sidebar:
    st.header("📂 Upload Document")
    st.caption("Supported formats: PDF, TXT")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Upload a PDF or text file to chat with it"
    )

    if uploaded_file:
        if st.button("📥 Process Document", use_container_width=True):
            with st.spinner("Reading and processing your document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if uploaded_file.name.endswith(".pdf"):
                    loader = PyMuPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")

                docs = loader.load()
                chunks = split_documents(docs)

                embeddings = get_embeddings()
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="user_document"
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.filename = uploaded_file.name
                st.session_state.num_chunks = len(chunks)
                st.session_state.messages = []
                os.unlink(tmp_path)

            st.success(
                f"✅ Ready! Processed {len(chunks)} chunks from "
                f"{uploaded_file.name}"
            )

    if st.session_state.filename:
        st.divider()
        st.caption("📄 Current document:")
        st.info(st.session_state.filename)
        st.caption(
            f"📊 {st.session_state.num_chunks} chunks indexed"
        )

    st.divider()
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.vectorstore:
    st.info(
        "👈 Upload a document from the sidebar to get started. "
        "You can ask any question about your document!"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 What can I do?")
        st.markdown("""
        - Answer questions from your document
        - Summarize sections
        - Find specific information
        - Explain complex content
        """)
    with col2:
        st.markdown("### 📁 Supported formats")
        st.markdown("""
        - PDF documents
        - Text files (.txt)
        - Research papers
        - Notes and reports
        """)
    with col3:
        st.markdown("### 💡 Example questions")
        st.markdown("""
        - What is the main topic?
        - Summarize chapter 2
        - What are the key findings?
        - Explain this concept
        """)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input(
        f"Ask anything about {st.session_state.filename}..."
    ):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reading your document..."):
                chunks = st.session_state.vectorstore.similarity_search(
                    user_input, k=4
                )
                answer = generate_answer(user_input, chunks)

            st.markdown(answer)

            with st.expander("📄 Source chunks used"):
                for i, chunk in enumerate(chunks):
                    st.caption(
                        f"Chunk {i+1}: ...{chunk.page_content[:200]}..."
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        log_query(
            user_input,
            answer,
            st.session_state.filename
        )