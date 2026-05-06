# RAG Assistant

A RAG (Retrieval-Augmented Generation) system built with LangGraph, Weaviate, and Ollama. The assistant routes questions to the correct knowledge base using an LLM router and generates answers based on retrieved context.

## Architecture

- **Router** — LLM determines which index to search based on the question
- **Weaviate** — vector database with 4 separate indexes
- **Ollama** — local LLM (qwen2.5:7b) and embeddings (nomic-embed-text)
- **LangGraph** — manages state and graph flow between nodes

## Requirements

- Docker
- Ollama
- Python 3.11+

## Setup and launch

**1. Start Weaviate**
```bash
docker-compose up -d
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull Ollama models**
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

**4. Configure environment**
```bash
cp .env.example .env
```

```env
WEAVIATE_URL=http://localhost:8080
DATA_FOLDER=data
INDEX_GADGETS=Gadgets
INDEX_HEADPHONES=Headphones
INDEX_LAPTOPS=Laptops
INDEX_SMARTPHONES=Smartphones
```

**5. Start the server**
```bash
python -m uvicorn api:app --reload
```

**6. Index documents**
```bash
POST http://localhost:8000/init
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a question |
| POST | `/init` | Index all documents |

## Web interface

The interface will be available at: http://127.0.0.1:7860
