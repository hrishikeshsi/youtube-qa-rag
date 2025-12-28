# YouTube Q&A RAG Application

## Project Overview
This is a RAG (Retrieval Augmented Generation) application for YouTube video Q&A using:
- **LangChain** - RAG pipeline orchestration
- **FAISS** - Vector store for embeddings
- **Google Gemini 2.5 Flash** - LLM for generating answers
- **Google Embedding Model** - Text embeddings
- **Streamlit** - Web interface

## Development Guidelines
- Use Python 3.10+
- Follow PEP 8 style guidelines
- Keep API keys in .env file (never commit)
- Use type hints for function signatures

## Project Structure
```
RAG/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Actual environment variables (gitignored)
└── README.md           # Project documentation
```

## Key Components
1. **YouTube Transcript Loader** - Extracts transcript from YouTube videos
2. **Text Splitter** - Chunks transcript for better retrieval
3. **Embeddings** - Google's embedding model for vectorization
4. **FAISS Vector Store** - Stores and retrieves relevant chunks
5. **Gemini LLM** - Generates answers based on retrieved context
