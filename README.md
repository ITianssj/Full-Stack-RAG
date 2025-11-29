# Full-Stack RAG Search Engine

Private Google for your PDFs — built with Grok-4.1-fast + bge embeddings + Chroma

## Installation
Install dependencies from the included requirements file:

```sh
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Features
- Upload PDF/DOCX/TXT → instant search
- GPU auto-detection 
- Grok-4.1-fast via OpenRouter (o1-level reasoning)
- Professional logging, Pydantic validation, .env secrets
- Graceful error handling
- Ready for Streamlit Cloud



### Tech Stack
- Streamlit • LangChain • Chroma • Grok-4.1-fast • bge embeddings • Pydantic • Loguru


