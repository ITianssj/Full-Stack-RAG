"""
Streamlit RAG Search Engine Application

This is the main Streamlit application for the Full-Stack RAG Search Engine.
It provides a web interface for document upload, ingestion, and question-answering
using retrieval-augmented generation with vector similarity search.

Features:
- Document upload and ingestion (PDF, DOCX, TXT)
- Real-time chat interface for Q&A
- Automatic data folder creation for cloud deployment
- Error handling with user-friendly messages
- Session state management for conversation history
- Responsive UI with sidebar for document management

The application integrates all components: configuration, logging, document
processing, vector search, and LLM generation to provide a complete RAG solution.
"""

import streamlit as st
import os
from config import settings
from logger import logger
from ingest import ingest_document
from models import IngestRequest, QueryRequest
from utils import query_rag

# Configure Streamlit page settings
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide"
)
st.title(settings.app_title)

# Ensure data folder exists (critical for cloud deployments)
if not os.path.exists(settings.data_folder):
    os.makedirs(settings.data_folder, exist_ok=True)
    logger.info(f"Created data folder: {settings.data_folder}")

# Sidebar for document ingestion
with st.sidebar:
    st.header("Ingest Document")

    # File uploader for document ingestion
    uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])
    if uploaded:
        # Construct file path in data directory
        filename = uploaded.name
        path = os.path.join(settings.data_folder, filename)

        # Save uploaded file to disk
        try:
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            logger.info(f"Saved uploaded file: {path}")
        except Exception as e:
            st.error(f"Save error: {e}")
            st.stop()

        # Ingestion button and processing
        if st.button("Ingest"):
            with st.spinner("Ingestion in progress..."):
                req = IngestRequest(file_path=path)
                ingest_document(req)
            st.success(f"Ingested {filename}! Ask questions below.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Validate and clean user input
            clean_prompt = prompt.strip()
            if len(clean_prompt) < 3:
                answer = "Please ask more than 2 characters so I can help you properly."
            else:
                try:
                    req = QueryRequest(question=clean_prompt)
                    answer = query_rag(req)
                except Exception as e:
                    answer = f"Sorry, something went wrong: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
