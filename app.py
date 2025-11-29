# app.py — FIXED FOR STREAMLIT CLOUD (creates data folder + graceful saves)
import streamlit as st
import os
from config import settings
from logger import logger
from ingest import ingest_document
from models import IngestRequest, QueryRequest
from utils import query_rag

st.set_page_config(page_title=settings.app_title, page_icon=settings.app_icon, layout="wide")
st.title(settings.app_title)

# FIXED: Create data folder if missing (Cloud needs this)
if not os.path.exists(settings.data_folder):
    os.makedirs(settings.data_folder, exist_ok=True)
    logger.info(f"Created data folder: {settings.data_folder}")

with st.sidebar:
    st.header("Ingest Document")
    uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])
    if uploaded:
        # FIXED: Build path + create if needed
        filename = uploaded.name
        path = os.path.join(settings.data_folder, filename)
        
        try:
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            logger.info(f"Saved uploaded file: {path}")
        except Exception as e:
            st.error(f"Save error: {e}")
            st.stop()
        
        if st.button("Ingest"):
            with st.spinner("Ingestion in progress..."):
                req = IngestRequest(file_path=path)
                ingest_document(req)
            st.success(f"Ingested {filename}! Ask questions below.")

# Chat (unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Smart handling: auto-fix short questions
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