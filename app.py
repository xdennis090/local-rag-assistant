import re
import urllib.request
import json
from typing import Dict, List, Tuple

import streamlit as st

from src.config import (
    AVAILABLE_CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_HISTORY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    SAMPLE_DOCS_DIR,
)
from src.ingestion import (
    get_indexed_stats,
    ingest_directory,
    ingest_uploaded_files,
    reset_vector_store,
)
from src.rag_engine import retrieve_and_stream

st.set_page_config(
    page_title="Local Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def fetch_local_ollama_models() -> Tuple[bool, List[str]]:
    """Prüft, ob Ollama erreichbar ist und liest lokal installierte Chat-Modelle aus."""
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [
                m["name"]
                for m in data.get("models", [])
                if "embed" not in m["name"].lower()
            ]
            return True, models if models else AVAILABLE_CHAT_MODELS
    except Exception:
        return False, AVAILABLE_CHAT_MODELS


def split_reasoning_and_answer(raw_text: str) -> Tuple[str, str]:
    """Trennt optionale <think>...</think>-Blöcke (z. B. DeepSeek-R1) von der finalen Antwort."""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return reasoning, answer
    return "", raw_text.strip()


def serialize_sources(docs) -> List[Dict]:
    """Wandelt LangChain-Documents in ein kompaktes Dict-Format für den Session-State um."""
    serialized = []
    for doc in docs:
        serialized.append(
            {
                "source_file": doc.metadata.get("source_file", "Unbekannt"),
                "page_display": doc.metadata.get("page_display"),
                "content": doc.page_content.strip(),
            }
        )
    return serialized


def render_sources_expander(sources: List[Dict], search_query: str = "") -> None:
    """Rendert die verwendeten Dokumenten-Quellen sauber unter einer Antwort."""
    if not sources:
        return

    with st.expander(f"Verwendete Quellen ({len(sources)} Chunks)", expanded=False):
        if search_query:
            st.caption(f"**Retrieval-Suchanfrage:** `{search_query}`")
        for idx, src in enumerate(sources, start=1):
            page_info = f" · Seite {src['page_display']}" if src.get("page_display") else ""
            st.markdown(f"**{idx}. {src['source_file']}{page_info}**")
            st.info(src["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Konfiguration")

    ollama_online, installed_models = fetch_local_ollama_models()
    if ollama_online:
        st.success("Ollama Server: Online", icon="🟢")
    else:
        st.error("Ollama Server nicht erreichbar (Port 11434)", icon="🔴")

    default_idx = 0
    for i, m in enumerate(installed_models):
        if m.startswith(DEFAULT_CHAT_MODEL):
            default_idx = i
            break

    selected_model = st.selectbox(
        "Chat-Modell (Ollama)",
        options=installed_models,
        index=default_idx,
        help="Lokal über Ollama verfügbare Sprachmodelle.",
    )

    with st.expander("Erweiterte Parameter", expanded=False):
        top_k = st.slider(
            "Top-K Chunks (Retrieval)",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Anzahl der relevantesten Textabschnitte, die an das LLM übergeben werden.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="Niedrigere Werte (0.0 - 0.3) sorgen für faktengetreuere Antworten.",
        )
        num_ctx = st.select_slider(
            "Context Window (Tokens)",
            options=[2048, 4096, 8192, 16384],
            value=DEFAULT_CONTEXT_SIZE,
        )
        max_history = st.slider(
            "Chat-Verlauf (Turns)",
            min_value=0,
            max_value=10,
            value=DEFAULT_MAX_HISTORY,
            help="Anzahl vorheriger Frage-Antwort-Paare für Rückfragen.",
        )

    st.divider()
    st.subheader("Wissensbasis (ChromaDB)")

    stats = get_indexed_stats()
    col1, col2 = st.columns(2)
    col1.metric("Dokumente", len(stats["sources"]))
    col2.metric("Chunks", stats["chunk_count"])

    if stats["sources"]:
        with st.expander("Indexierte Dateien anzeigen", expanded=False):
            for src_name in stats["sources"]:
                st.markdown(f"- `{src_name}`")
    else:
        st.warning("Noch keine Dokumente indexiert.")

    uploaded_files = st.file_uploader(
        "Dokumente hinzufügen (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Hochgeladene Dateien indexieren", use_container_width=True, type="primary"):
            with st.spinner("Verarbeite und indexiere Dokumente..."):
                try:
                    f_count, c_count = ingest_uploaded_files(
                        uploaded_files, embedding_model=DEFAULT_EMBEDDING_MODEL
                    )
                    st.success(f"{f_count} Datei(en) ({c_count} Chunks) hinzugefügt!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Fehler bei der Indexierung: {exc}")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Beispieldaten", use_container_width=True, help="Indexiert Dateien aus data/sample_docs"):
            with st.spinner("Lade Beispieldokumente..."):
                try:
                    f_count, c_count = ingest_directory(
                        SAMPLE_DOCS_DIR, embedding_model=DEFAULT_EMBEDDING_MODEL
                    )
                    if f_count > 0:
                        st.success(f"{f_count} Datei(en) geladen.")
                        st.rerun()
                    else:
                        st.info("Keine Dateien in data/sample_docs.")
                except Exception as exc:
                    st.error(f"Fehler: {exc}")

    with btn_col2:
        if st.button("DB leeren", use_container_width=True, help="Löscht alle indexierten Dokumente"):
            reset_vector_store()
            st.session_state.messages = []
            st.rerun()

    if st.button("Chatverlauf löschen", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Local Document RAG Assistant")
st.caption(
    "100 % lokaler Dokumenten-Assistent auf Basis von Ollama, LangChain & ChromaDB · "
    "Keine Daten verlassen deinen Rechner."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("reasoning"):
            with st.expander("Gedankengang (Reasoning)", expanded=False):
                st.markdown(message["reasoning"])
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources_expander(message["sources"], message.get("search_query", ""))

if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
    if not ollama_online:
        st.error("Bitte starte zuerst den lokalen Ollama-Server (`ollama serve`).")
        st.stop()

    if stats["chunk_count"] == 0:
        st.warning("Die Wissensbasis ist noch leer. Bitte lade links Dokumente hoch oder klicke auf 'Beispieldaten'.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    history_before_current = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Durchsuche Dokumente..."):
                token_stream, retrieved_docs, used_search_query = retrieve_and_stream(
                    query=prompt,
                    chat_messages=history_before_current,
                    model_name=selected_model,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    top_k=top_k,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    max_history=max_history,
                )

            raw_response = st.write_stream(token_stream)
            reasoning, clean_answer = split_reasoning_and_answer(raw_response)

            serialized_sources = serialize_sources(retrieved_docs)
            render_sources_expander(serialized_sources, used_search_query)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": clean_answer,
                    "reasoning": reasoning,
                    "sources": serialized_sources,
                    "search_query": used_search_query,
                }
            )
            if reasoning:
                st.rerun()

        except Exception as exc:
            st.error(f"Fehler bei der Antwortgenerierung: {exc}")