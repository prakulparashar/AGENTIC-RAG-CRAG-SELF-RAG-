from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langgraph.checkpoint.sqlite import SqliteSaver                  
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOllama(model="llama3.1", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -------------------
# 2. CRAG thresholds
# -------------------
UPPER_TH = 0.7
LOWER_TH = 0.3

# -------------------
# 3. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 4. CRAG — Scorer
# -------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str


_doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores.\n"
     "Also return a short reason.\n"
     "Output JSON only."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])

_doc_eval_chain = _doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def _eval_docs(question: str, docs: List[Document]) -> tuple[List[Document], str, str]:
    """
    Score each doc and return (good_docs, verdict, reason).

    Verdicts:
      CORRECT   — at least one doc > UPPER_TH  → use PDF only
      INCORRECT — all docs < LOWER_TH           → use web only
      AMBIGUOUS — in between                    → use PDF + web
    """
    scores: List[float] = []
    good_docs: List[Document] = []

    for doc in docs:
        result = _doc_eval_chain.invoke({"question": question, "chunk": doc.page_content})
        scores.append(result.score)
        if result.score > LOWER_TH:
            good_docs.append(doc)

    if any(s > UPPER_TH for s in scores):
        return good_docs, "CORRECT", f"At least one chunk scored > {UPPER_TH}."

    if scores and all(s < LOWER_TH for s in scores):
        return [], "INCORRECT", f"All chunks scored < {LOWER_TH}."

    return good_docs, "AMBIGUOUS", f"No chunk > {UPPER_TH}, but not all < {LOWER_TH}."


# -------------------
# 5. CRAG — Query rewriter
# -------------------
class WebQuery(BaseModel):
    query: str


_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- Keep it short (6–14 words).\n"
     "- If the question implies recency (recent/latest/last week), add a constraint like (last 30 days).\n"
     "- Do NOT answer the question.\n"
     "- Return JSON with a single key: query"),
    ("human", "Question: {question}"),
])

_rewrite_chain = _rewrite_prompt | llm.with_structured_output(WebQuery)


def _rewrite_query(question: str) -> str:
    return _rewrite_chain.invoke({"question": question}).query


# -------------------
# 6. CRAG — Web search (Tavily)
# -------------------
_tavily = TavilySearch(max_results=5)


def _web_search(query: str) -> List[Document]:
    results = _tavily.invoke({"query": query})
    web_docs = []
    for r in results or []:
        # newer langchain-tavily returns strings, older returns dicts
        if isinstance(r, str):
            web_docs.append(Document(page_content=r, metadata={"source": "web"}))
        else:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
    return web_docs


# -------------------
# 7. CRAG — Sentence-level knowledge refinement
# -------------------
def _decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool


_filter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter.\n"
     "Return keep=true only if the sentence directly helps answer the question.\n"
     "Use ONLY the sentence. Output JSON only."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])

_filter_chain = _filter_prompt | llm.with_structured_output(KeepOrDrop)


def _refine_context(question: str, docs: List[Document]) -> str:
    combined = "\n\n".join(d.page_content for d in docs).strip()
    sentences = _decompose_to_sentences(combined)
    kept = [
        s for s in sentences
        if _filter_chain.invoke({"question": question, "sentence": s}).keep
    ]
    return "\n".join(kept).strip()


# -------------------
# 8. CRAG-enhanced rag_tool
# -------------------
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF using Corrective RAG (CRAG).

    Pipeline:
      1. Retrieve top-k chunks from the PDF vector store
      2. Score each chunk 0-1 for relevance
      3. Verdict:
           CORRECT   → refine PDF chunks only
           INCORRECT → rewrite query + web search only
           AMBIGUOUS → refine PDF chunks + web search
      4. Sentence-level filter to remove irrelevant sentences
      5. Return refined context

    Always include the thread_id when calling this tool.
    """

    if not thread_id:
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
        except Exception:
            pass

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # Step 1 — Retrieve
    raw_docs = retriever.invoke(query)

    # Step 2+3 — Score & verdict
    good_docs, verdict, reason = _eval_docs(query, raw_docs)

    web_docs: List[Document] = []
    web_query: Optional[str] = None

    # Step 4 — Corrective action
    if verdict in ("INCORRECT", "AMBIGUOUS"):
        web_query = _rewrite_query(query)
        web_docs = _web_search(web_query)

    # Step 5 — Assemble docs for refinement
    if verdict == "CORRECT":
        docs_to_refine = good_docs
    elif verdict == "INCORRECT":
        docs_to_refine = web_docs
    else:  # AMBIGUOUS
        docs_to_refine = good_docs + web_docs

    # Step 6 — Sentence-level refinement
    refined_context = _refine_context(query, docs_to_refine)

    return {
        "query": query,
        "refined_context": refined_context,
        "verdict": verdict,
        "reason": reason,
        "web_query": web_query,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        "relevance_grades": {
            "total_chunks": len(raw_docs),
            "good_chunks": len(good_docs),
            "web_search_triggered": verdict != "CORRECT",
        },
    }


# -------------------
# 9. Remaining tools + LLM binding
# -------------------
# Add back your other tools (search_tool, get_stock_price, calculator) here
tools = [rag_tool]  # e.g. tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 10. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 11. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
           "You are a helpful assistant.\n\n"
        "RULE 1: For greetings, casual conversation, general knowledge, coding, math, "
        "or any question NOT about a document — answer directly. Do NOT call any tool.\n\n"
        "RULE 2: Only call `rag_tool` with thread_id `{thread_id}` when the user is "
        "clearly asking about the content of an uploaded PDF or document.\n\n"
        "Examples of when NOT to call the tool: 'hi', 'hello', 'how are you', "
        "'what is Python', 'explain machine learning'.\n\n"
        "Examples of when to call the tool: 'what is this PDF about', "
        "'summarize the document', 'what does it say about X'.\n\n"
        "When you do call rag_tool, use the verdict to frame your answer:\n"
        "  • CORRECT   → answer came from the PDF\n"
        "  • INCORRECT → used web search\n"
        "  • AMBIGUOUS → combined PDF and web search\n\n"
        "If the tool returns a no-document error, ask the user to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# -------------------
# 12. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# -------------------
# 13. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 14. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})