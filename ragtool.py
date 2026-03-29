import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragtool import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Page Config ===========================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def friendly_thread_label(thread_id, index):
    """Show a short label instead of the full UUID."""
    short = str(thread_id)[:8]
    return f"Chat {index + 1} — {short}…"


def verdict_status(tool_name: str, raw_content: str) -> tuple[str, str]:
    """
    Parse rag_tool result and return (label, state).
    Falls back to a generic label for other tools.
    """
    if tool_name == "rag_tool":
        try:
            result = json.loads(raw_content)
            verdict = result.get("verdict", "")
            label_map = {
                "CORRECT":   ("✅ Answered from PDF", "complete"),
                "INCORRECT": ("🌐 PDF had no relevant info — used web search", "complete"),
                "AMBIGUOUS": ("🔀 Combined PDF + web search", "complete"),
            }
            return label_map.get(verdict, (f"🔧 `{tool_name}` finished", "complete"))
        except Exception:
            pass
    return (f"✅ `{tool_name}` finished", "complete")


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("🧠 RAG Chatbot")
st.sidebar.markdown(f"**Thread:** `{thread_key[:8]}…`")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# --- PDF Upload ---
st.sidebar.subheader("📄 Document")
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"**{latest_doc.get('filename')}**\n\n"
        f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed yet for this chat.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.divider()

# --- Past Conversations ---
st.sidebar.subheader("🗂 Past conversations")
if not threads:
    st.sidebar.caption("No past conversations yet.")
else:
    for i, thread_id in enumerate(threads):
        label = friendly_thread_label(thread_id, len(threads) - 1 - i)
        is_active = str(thread_id) == thread_key
        button_label = f"▶ {label}" if is_active else label
        if st.sidebar.button(button_label, key=f"side-thread-{thread_id}", use_container_width=True):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")
st.caption("Ask questions about your uploaded PDF. The CRAG pipeline will automatically decide whether to use the document, the web, or both.")

st.divider()

# --- Chat history ---
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # markdown instead of st.text

# --- Chat input ---
user_input = st.chat_input("Ask about your document or anything else…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None, "last_tool": None, "last_content": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # --- Tool call started (before tool runs) ---
                if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                    for tc in message_chunk.tool_calls:
                        tool_name = tc.get("name", "tool")
                        status_holder["last_tool"] = tool_name
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Running `{tool_name}`…", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Running `{tool_name}`…",
                                state="running",
                                expanded=True,
                            )

                # --- Tool result returned ---
                if isinstance(message_chunk, ToolMessage):
                    status_holder["last_content"] = message_chunk.content
                    tool_name = getattr(message_chunk, "name", None) or status_holder.get("last_tool", "tool")
                    label, state = verdict_status(tool_name, message_chunk.content)
                    if status_holder["box"] is not None:
                        status_holder["box"].update(
                            label=label, state=state, expanded=False
                        )
                        status_holder["box"] = None  # reset for next tool

                # --- AI text response ---
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # --- Document metadata footer ---
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 **{doc_meta.get('filename')}** — "
            f"{doc_meta.get('chunks')} chunks · {doc_meta.get('documents')} pages"
        )

# ============================ Thread Switch =======================
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()