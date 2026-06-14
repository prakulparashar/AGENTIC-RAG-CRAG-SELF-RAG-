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
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

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
_THREAD_RETRIEVERS: Dict[str, Any] = {} #global dictionaries,stores the pointers to the vector stores , because we are assigning dedicated FAISS store to its dedicated thread:id (user). it is a dict. where every thread id (key) is associated with its respective retriever object, and each retriever object points to the FAISS store of THAT thread_id only. get retriever function just returns the retriever object of THAT thread id
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS: #suppose if user makes req. with thread_id_abc, this fxn looks for that ID in the dict we defined above. if it exists....
        return _THREAD_RETRIEVERS[thread_id] #...if it exists, then this returns the pointer pointing to the vector store for the thread_id_abc
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

        vector_store = FAISS.from_documents(chunks, embeddings)         #EMBEDDING being passed as arguments, FAISS now knows how to embedd even the queries. vector_store now becomes an object. in the below line, .as_retriever is the method
        retriever = vector_store.as_retriever(                          # retriever.invoke() can be used now . basically .invoke()is the method which actually does the retrieval, and as_retriever is used to create the retriever object
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever #at this moment, the retriever object is assigned to the respective thread_id and this key-value pair is stored INSIDE the _THREAD_RETRIEVERS dictionary, which is global.
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
# 4. CRAG Subgraph — State
# -------------------
class CRAGState(TypedDict):
    question: str
    thread_id: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    web_query: str
    web_docs: List[Document]
    refined_context: str


# -------------------
# 5. CRAG Subgraph — Scorer
# -------------------
class DocEvalScore(BaseModel): #basemodel manje pydantic
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

_doc_eval_chain = _doc_eval_prompt | llm.with_structured_output(DocEvalScore) #created an agent using langchain, prompt is _doc_eval_prompt  passed the pydantic format named "docevalscore". so the response will be in that format only.
#Notes: | is called the pipe operator in langchain. this means the take whatever is on the left and feed to the right.

# -------------------
# 6. CRAG Subgraph — Nodes
# -------------------
def retrieve_node(state: CRAGState) -> CRAGState: #outputs a CRAGState
    retriever = _get_retriever(state["thread_id"]) #as discussed earlier, get retriever function searches the _THREAD_RETRIEVERS dict and returns to retriever object of THAT thread_id
    if retriever is None:
        return {"docs": []} #if query is made before a PDF is uploaded, this line handles it by returning empty set of DOCS
    return {"docs": retriever.invoke(state["question"])}  #retriever.invoke retrieves the chunks from the vector store. the user's query is passed from the "question" state. the retrieved chunks are stored in "docs" state

#a good way to think about langgraph node -> return {"docs": retriever.invoke(state["question"])}......in this kind of syntax, the thing on the left is the value of the state which will be updated, and the thing on the right is the method whose return value will replace it, right?
def eval_each_doc_node(state: CRAGState) -> CRAGState:
    question = state["question"]
    scores: List[float] = []
    good_docs: List[Document] = []

    for doc in state["docs"]:
        result = _doc_eval_chain.invoke({"question": question, "chunk": doc.page_content})
        scores.append(result.score)
        if result.score > LOWER_TH:
            good_docs.append(doc)

    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good_docs, #good_docs used later in the refinement function (part 9)
            "verdict": "CORRECT",
            "reason": f"At least one chunk scored > {UPPER_TH}.",
        }
    if scores and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All chunks scored < {LOWER_TH}.",
        }
    return {
        "good_docs": good_docs,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk > {UPPER_TH}, but not all < {LOWER_TH}.",
    }


# -------------------
# 7. CRAG Subgraph — Query rewriter
# -------------------
class WebQuery(BaseModel):
    query: str


_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- Keep it short (6-14 words).\n"
     "- If the question implies recency (recent/latest/last week), add a constraint like (last 30 days).\n"
     "- Do NOT answer the question.\n"
     "- Return JSON with a single key: query"),
    ("human", "Question: {question}"),
])

_rewrite_chain = _rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state: CRAGState) -> CRAGState:
    result = _rewrite_chain.invoke({"question": state["question"]})    #accesing the original prompt by the user from the state and passing it in the .invoke() function. 
    return {"web_query": result.query} #updates the state


# -------------------
# 8. CRAG Subgraph — Web search
# -------------------
_tavily = TavilySearch(max_results=5)


def web_search_node(state: CRAGState) -> CRAGState:
    query = state.get("web_query") or state["question"]
    results = _tavily.invoke({"query": query})
    web_docs = []
    for r in results or []:
        if isinstance(r, str):
            web_docs.append(Document(page_content=r, metadata={"source": "web"}))
        else:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
    return {"web_docs": web_docs}


# -------------------
# 9. CRAG Subgraph — Refinement
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


def refine_node(state: CRAGState) -> CRAGState:
    question = state["question"]
    verdict = state.get("verdict", "INCORRECT")

    if verdict == "CORRECT":
        docs_to_use = state["good_docs"] #good_docs were updated above in part 6
    elif verdict == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_docs"]

    combined = "\n\n".join(d.page_content for d in docs_to_use).strip()
    sentences = _decompose_to_sentences(combined)
    kept = [
        s for s in sentences
        if _filter_chain.invoke({"question": question, "sentence": s}).keep
    ]
    return {"refined_context": "\n".join(kept).strip()}


# -------------------
# 10. CRAG Subgraph — Routing
# -------------------
def route_after_eval(state: CRAGState) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    return "rewrite_query"


# -------------------
# 11. Build CRAG subgraph
# -------------------
crag_graph = StateGraph(CRAGState)

crag_graph.add_node("retrieve", retrieve_node)
crag_graph.add_node("eval_each_doc", eval_each_doc_node)
crag_graph.add_node("rewrite_query", rewrite_query_node)
crag_graph.add_node("web_search", web_search_node)
crag_graph.add_node("refine", refine_node)

crag_graph.add_edge(START, "retrieve")
crag_graph.add_edge("retrieve", "eval_each_doc")
crag_graph.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "rewrite_query": "rewrite_query"},
)
crag_graph.add_edge("rewrite_query", "web_search")
crag_graph.add_edge("web_search", "refine")
crag_graph.add_edge("refine", END)

crag_pipeline = crag_graph.compile()    #CRAG pipeline/graph being binded/compiled. now this becomes an executable graph


# -------------------
# 12. rag_tool — calls the CRAG subgraph
# -------------------
@tool
def rag_tool(query: str = "summarize the document", thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF using Corrective RAG (CRAG).
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

    # CRAG subgraph
    result = crag_pipeline.invoke({ #CRAG pipeline/graph being binded, initial state being passed on every query
        "question": query,
        "thread_id": thread_id,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "web_query": "",
        "web_docs": [],
        "refined_context": "",
    })

    return {                                                #returns a normal python list
        "query": query,
        "refined_context": result["refined_context"],
        "verdict": result["verdict"],
        "reason": result["reason"],
        "web_query": result.get("web_query"),
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        "relevance_grades": {
            "total_chunks": len(result["docs"]),
            "good_chunks": len(result["good_docs"]),
            "web_search_triggered": result["verdict"] != "CORRECT",
        },
    }


# -------------------
# 13. Tools + LLM binding
# -------------------
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)   # {modename}_with_tools used to bind the tool with the model


# -------------------
# 14. Chat State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] #add_messages is the reducer function, by adding this, we are not rewriting, we are appending


# -------------------
# 15. Chat Node
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant. thread_id is `{thread_id}`.\n\n"
            "Rules:\n"
            "1. For greetings or casual chat — respond with a short friendly message. Never return empty.\n"
            "2. For questions about a document or PDF — call `rag_tool` with thread_id and the user question as query.\n"
            "3. For general questions — answer directly, no need to call `rag_tool`.\n"
            "4. Never return an empty response under any circumstance."
        )
    )

    messages = [system_message, *state["messages"]]  #* creates a temp chatstate, cuz the system message is only required for the current response generation , thats why its not stored in the global chatstate. only the AI message in appended in the global chatstate at line 392
    response = llm_with_tools.invoke(messages, config=config) #creates AImessage on the first pass, and final LLM response on the second pass. explanation on line 394

    if not response.content and not response.tool_calls:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]} #guardrail , if the node fails to create an AImessage

    return {"messages": [response]} #AI response being appended in the global state

#on the first pass of chat_node, the messages state consists of system message and human message. on the second pass it consists of ai message and tool message as well********** 

tool_node = ToolNode(tools) #predefined by langgraph. langgraph cant directly execute the rag+tool code, so in simple terms we wrap the rag_tool inside a tool node, and then we can add the tool node in the graph, basically tool node "executes" the rag_tool

#The pre-built LangGraph ToolNode intercepts this raw Python dictionary returned by the rag_tool. Automatically, behind the scenes, it converts that dictionary into a string and wraps it into a formal LangChain ToolMessage.
#Because ChatState has a built-in message history mechanism (add_messages), LangGraph automatically appends this new ToolMessage straight into your global state["messages"] array
#when the main graph reaches the chatnode for the second time , this tool message is used to generate the response

# -------------------
# 16. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# -------------------
# 17. Main chat graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 18. Helpers
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





#rag_tool hands over the CRAGState to the subgraph. subgraph gives the optimal context (using websearch amd all). and then the llm is binded with this tool. then we invoke the llm in the chatnode, after which the rag_tool returns us the context required by the llm. 

#now we create the main graph. is starts, reaches chat_node, chatnode executes the tool node(basically rag_tool). and then llm has the context and generates the response












#the main graph executed the chat_node, since we are invoking the llm inside this node, it return an AImessage which tells us that we have to call the rag_tool (since that is the tool bindede with our llm)
#then we reach the tool_node, which physically executes the rag_tool and runs the CRAG subgraph (FAISS retrieval, grading, web search fallback). this gives us the tool message which contains the refined context.
#then the graph again reached tha chatnode, which uses this refined context to generate the response.

#this is INDUSTRY standard procedure, one node to create ai message and another to create tool message.









#1. MAIN GRAPH= chatnode(creates ai message by invvoking the llm. ai message informs the llm which tool it needs to call)
#2. MAINGRAPH REACHES TOOL NODE= tool node calls the rag_tool, which basically implements our subgraph, and then the rag_tool returns a python dicitonary consisting of our refined_context. when we do tools = [rag_tool], llm_with_tools = llm.bind_tools(tools) , our toolmessage gets updated with the refined context
#3. MAIN GRAPH AGAIN REACHED CHAT_NODE= now chat_node uses this tool_message (which consists of our refined_context)and our final response is generated




