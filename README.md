# Full-Stack RAG Search Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.38+-red.svg)](https://streamlit.io/)

**Private Google for your PDFs** — A powerful Retrieval-Augmented Generation (RAG) search engine built with Groq-4.1-fast, BGE embeddings, and Chroma vector database.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd full-stack-rag-search-engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`.

## 📋 Features

- **📄 Multi-format Document Support**: Upload and search through PDF, DOCX, and TXT files
- **🧠 Advanced AI**: Powered by Groq-4.1-fast LLM with o1-level reasoning capabilities
- **🔍 Semantic Search**: BGE embeddings for accurate document retrieval
- **💾 Persistent Storage**: Chroma vector database for efficient document storage
- **🎯 Smart Validation**: Pydantic models with automatic input cleaning and validation
- **📊 Professional Logging**: Comprehensive logging with Loguru for monitoring and debugging
- **☁️ Cloud Ready**: Optimized for Streamlit Cloud deployment
- **⚡ GPU Acceleration**: Automatic GPU detection for faster embedding generation
- **🛡️ Error Handling**: Graceful error handling with user-friendly messages

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   Document      │────│   Vector DB     │
│                 │    │   Processing    │    │   (Chroma)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   LLM Query     │
                    │   (Groq API)    │
                    └─────────────────┘
```

### Core Components

- **`app.py`**: Main Streamlit application with chat interface and document upload
- **`config.py`**: Application settings and environment configuration
- **`models.py`**: Pydantic data models for request validation
- **`utils.py`**: RAG query utilities and vector database operations
- **`ingest.py`**: Document ingestion and text chunking pipeline
- **`logger.py`**: Centralized logging configuration

## 🔧 Configuration

The application uses the following configuration options (can be overridden via `.env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | Required | API key for Groq LLM service |
| `DEFAULT_MODEL` | `llama-3.1-8b-instant` | LLM model for RAG queries |
| `EMBEDDING_MODEL` | Auto-detected | BGE embedding model (large/small based on GPU) |
| `EMBEDDING_DIM` | Auto-detected | Embedding vector dimensions |
| `CHUNK_SIZE` | `1000` | Text chunk size for document splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between text chunks |
| `TOP_K` | `8` | Number of similar documents to retrieve |

## 📖 Usage

### Document Ingestion

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Click "Ingest"**: Process the document into the vector database
3. **Wait for Confirmation**: The system will chunk and embed your document

### Question Answering

1. **Ask Questions**: Type your question in the chat input
2. **Get Answers**: The system retrieves relevant context and generates answers
3. **View Sources**: Answers include source document references

### Example Queries

```
"What are the main benefits of renewable energy?"
"Tell me about the company's financial performance in Q3"
"How does the algorithm handle edge cases?"
```

## 🔍 API Reference

### Core Functions

#### `query_rag(request: QueryRequest) -> str`
Performs a RAG query using vector similarity search and LLM generation.

**Parameters:**
- `request` (QueryRequest): Query request with question and parameters

**Returns:**
- `str`: Generated answer with source attribution

#### `ingest_document(request: IngestRequest) -> None`
Ingests a document into the vector database.

**Parameters:**
- `request` (IngestRequest): Ingestion request with file path

### Data Models

#### `QueryRequest`
```python
class QueryRequest(BaseModel):
    question: str          # User's search question
    collection: str = "default"  # Vector database collection
    top_k: int = 8        # Number of documents to retrieve (1-20)
```

#### `IngestRequest`
```python
class IngestRequest(BaseModel):
    file_path: str         # Path to document file
    collection: str = "default"  # Target collection

**Made with ❤️ for document search and Q&A By Sagar Patel**
