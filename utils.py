# utils.py — FINAL VERSION THAT NEVER SHOWS 401 TO USERS
from config import settings
from logger import logger
from models import QueryRequest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import time

# Embeddings (unchanged
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": settings.embedding_device},
    encode_kwargs={"normalize_embeddings": True}
)

def get_db(collection: str = "default"):
    return Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=embeddings,
        collection_name=collection
    )

def query_rag(request: QueryRequest) -> str:
    db = get_db(request.collection)
    docs_with_scores = db.similarity_search_with_score(request.question, k=request.top_k)

    context = ""
    sources = []
    for doc, score in docs_with_scores:
        if score > 1.5:
            continue
        context += doc.page_content + "\n\n"
        sources.append(doc.metadata.get("source", "Unknown").split("\\")[-1])

    if not context.strip():
        return "No relevant information found in the documents."

    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        response = client.chat.completions.create(
            model=settings.default_model,
            messages=[
                {"role": "system", "content": "Use only the provided context. Answer accurately and concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        logger.success("Answer generated with FREE Groq Llama-3.1")
        return answer + "\n\nSources: " + " | ".join(set(sources))

    except Exception as e:
        logger.error(f"Groq failed: {e}")
        return "Temporary issue. Try again in 10 seconds."