# 🧠 Multi-Utility RAG Chatbot

A locally-running AI chatbot that answers questions about uploaded PDFs using **Corrective RAG (CRAG)** — a pipeline that evaluates retrieved chunks, falls back to web search when the document isn't sufficient, and blends both sources when needed.

Built with **LangGraph**, **Ollama**, **FAISS**, and **Streamlit**.

---

## Architecture Overview

```
User Message
     │
     ▼
 Chat Node  ──► (no tool needed) ──► Direct LLM Response
     │
     ▼ (PDF question detected)
  rag_tool
     │
     ▼
 CRAG Subgraph
  ├── retrieve        → FAISS similarity search (top-4 chunks)
  ├── eval_each_doc   → LLM scores each chunk [0.0–1.0]
  │     ├── CORRECT   (any chunk > 0.7) → refine with PDF context
  │     ├── INCORRECT (all chunks < 0.3) → rewrite query → web search → refine
  │     └── AMBIGUOUS (mixed scores)    → rewrite query → web search → refine
  └── refine          → LLM synthesizes final answer
     │
     ▼
 Chat Node (2nd pass) → Final Response to User
```

Each user session gets its **own FAISS vector store**, keyed by `thread_id`. Conversation history is persisted to **SQLite** via LangGraph's checkpointer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | `llama3.1` via Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | FAISS (in-memory, per thread) |
| Web search | Tavily Search API |
| Orchestration | LangGraph (StateGraph) |
| Memory / checkpointing | SQLite (`chatbot.db`) |
| PDF loading | LangChain `PyPDFLoader` |
| Frontend | Streamlit |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- A [Tavily API key](https://tavily.com) (free tier available)

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Make sure Ollama is running on `http://localhost:11434` before starting the app.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the app

```bash
streamlit run frontend.py
```

---

## Requirements

Create a `requirements.txt` with:

```
streamlit
langchain
langchain-community
langchain-core
langchain-ollama
langchain-tavily
langchain-text-splitters
langgraph
faiss-cpu
pypdf
python-dotenv
pydantic
```

---

## Usage

1. **Start a chat** — the app opens with a fresh thread automatically.
2. **Upload a PDF** — use the sidebar uploader. The document is chunked and indexed into FAISS for that thread only.
3. **Ask questions** — the CRAG pipeline automatically decides:
   - Answer from PDF (chunks scored highly relevant)
   - Search the web (PDF had no useful info)
   - Blend both (mixed relevance)
4. **Switch chats** — click any past conversation in the sidebar to restore it, including its message history.
5. **New chat** — click **＋ New Chat** to start a fresh thread with a clean context.

The status indicator in the chat shows which path CRAG took for each response.

---

## Project Structure

```
├── backend.py        # LangGraph graphs, CRAG pipeline, FAISS ingestion, tools
├── frontend.py       # Streamlit UI, chat rendering, sidebar, thread management
├── chatbot.db        # SQLite file auto-created at runtime (conversation history)
├── .env              # API keys (not committed)
└── README.md
```

---

## How CRAG Works

The **Corrective RAG** pipeline grades retrieved chunks before using them:

- Each chunk is scored `0.0–1.0` by the LLM for relevance to the question.
- **CORRECT** — at least one chunk scores above `0.7`. The answer is synthesized purely from the PDF.
- **INCORRECT** — all chunks score below `0.3`. The question is rewritten into a web search query, and the answer comes from the web.
- **AMBIGUOUS** — mixed scores. Both PDF chunks and web results are combined into the final answer.

This prevents hallucination from low-quality retrievals and ensures the chatbot gracefully handles questions outside the document's scope.

---

## Notes

- Vector stores are **in-memory only** and reset when the server restarts. Re-upload your PDF after restarting.
- Conversation history is **persisted** in `chatbot.db` and survives restarts.
- Each chat thread has its own isolated vector store — uploading a PDF in one chat does not affect others.
