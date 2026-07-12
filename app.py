import os
import shutil
import time

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# CONFIG

DATA_PATH = "data"
VECTOR_DB = "vectorstore"

os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG Chatbot")
st.caption("Chat with your PDF knowledge base using Local RAG.")


# SIDEBAR

st.sidebar.title("📚 Enterprise RAG")

st.sidebar.success("Model : Gemma3 4B")

st.sidebar.info("Retriever : ChromaDB")

st.sidebar.info("Embedding : MiniLM-L6-v2")

st.sidebar.markdown("---")

st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            DATA_PATH,
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDF Uploaded Successfully!")

st.sidebar.markdown("---")

pdfs = [
    file
    for file in os.listdir(DATA_PATH)
    if file.endswith(".pdf")
]

st.sidebar.metric(
    "Total PDFs",
    len(pdfs)
)

for pdf in pdfs:

    st.sidebar.write(f"✅ {pdf}")

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Rebuild Knowledge Base"):

    if os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB)

    st.cache_resource.clear()

    st.sidebar.success("✅ Knowledge Base Rebuilt!")

    st.rerun() 

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# LOAD RAG
@st.cache_resource
def load_rag():

    # Embedding Model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # If Vector DB already exists
    if os.path.exists(VECTOR_DB):

        vectordb = Chroma(
            persist_directory=VECTOR_DB,
            embedding_function=embedding
        )

    else:

        # Get all PDF files
        pdf_files = [
            f for f in os.listdir(DATA_PATH)
            if f.endswith(".pdf")
        ]

        # No PDFs found
        if len(pdf_files) == 0:

            st.warning("📂 Please upload at least one PDF to continue.")
            st.stop()

        documents = []

        # Load PDFs
        for file in pdf_files:

            loader = PyPDFLoader(
                os.path.join(DATA_PATH, file)
            )

            documents.extend(loader.load())

        # Split Documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create Vector DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=VECTOR_DB
        )

        # Uncomment ONLY if your Chroma version supports it
        # vectordb.persist()

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # Local LLM
    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0
    )

    return retriever, llm

retriever, llm = load_rag()

# ASK QUESTION

def ask_question(question):

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # No documents retrieved
    if not docs:

        return (
            "I couldn't find this information in the uploaded documents.",
            [],
            0
        )

    # Use top 3 retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs[:3]]
    )

    prompt = f"""
You are an Enterprise AI Assistant.

Answer ONLY using the provided context.

STRICT RULES:

1. Never use your own knowledge.

2. Never guess.

3. If the answer is not present in the context,
reply exactly:

I couldn't find this information in the uploaded documents.

4. Answer in simple English.

5. Keep the answer concise.

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{question}

======================
ANSWER
======================
"""

    # Measure LLM response time
    start = time.time()
    try:

        response = llm.invoke(prompt)

    except Exception as e:

        return (
            f"⚠️ Ollama Error:\n\n{str(e)}",
            [],
            0
        )

    

    end = time.time()

    response_time = end - start

    # Collect Sources
    sources = []

    for doc in docs:

        filename = os.path.basename(
            doc.metadata["source"]
        )

        page = doc.metadata.get("page", 0) + 1

        sources.append(
            f"{filename} (Page {page})"
        )

    # Remove duplicate sources
    sources = list(dict.fromkeys(sources))

    return (
        response.content,
        sources,
        response_time
    )

# CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# CHAT INPUT

question = st.chat_input("Ask a question...")

if question:

    # User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Assistant Message
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, sources, response_time = ask_question(question)

        st.markdown(answer)

        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"⏱ {response_time:.2f} sec")

        with col2:
            st.caption(f"📄 {len(sources)} Source(s)")

        if sources:

            with st.expander("📄 View Sources"):

                for source in sources:

                    st.write(f"✅ {source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )