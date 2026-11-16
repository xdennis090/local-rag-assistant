from typing import Dict, Generator, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from src.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_HISTORY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)
from src.ingestion import get_vector_store

CONTEXTUALIZE_SYSTEM_PROMPT = """Formuliere die letzte Nutzerfrage unter Berücksichtigung des bisherigen Gesprächsverlaufs so um, dass sie als eigenständige Suchanfrage ohne Vorwissen aus dem Chatverlauf verständlich ist.
Beantworte die Frage NICHT, sondern gib ausschließlich eine kurze, präzise Suchanfrage (maximal 1 Satz) zurück. Falls die Frage bereits eigenständig verständlich ist, gib sie unverändert zurück."""

QA_SYSTEM_PROMPT = """Du bist ein präziser, faktenbasierter Dokumenten-Assistent.
Beantworte die Frage des Nutzers ausschließlich auf Grundlage der unten bereitgestellten Kontext-Auszüge aus den Dokumenten.

Regeln:
1. Nutze NUR Informationen aus dem bereitgestellten Kontext. Erfinde keine Fakten hinzu.
2. Wenn der Kontext keine ausreichenden Informationen zur Beantwortung enthält, sage klar und ehrlich, dass die hochgeladenen Dokumente dazu keine Auskunft geben.
3. Antworte in derselben Sprache, in der die Frage gestellt wurde (z. B. Deutsch bei deutschen Fragen, Englisch bei englischen Fragen), auch wenn die Quelldokumente in einer anderen Sprache verfasst sind.
4. Verweise bei wichtigen Fakten gerne auf das jeweilige Quelldokument und die Seitenzahl, falls angegeben.

Kontext aus den Dokumenten:
{context}"""


def build_chat_history(messages: List[Dict[str, str]], max_turns: int = DEFAULT_MAX_HISTORY) -> List[BaseMessage]:
    """Wandelt Streamlit-Chat-Nachrichten in LangChain-Messages um (begrenzt auf max_turns)."""
    if max_turns <= 0 or not messages:
        return []

    recent_messages = messages[-(max_turns * 2) :]
    history: List[BaseMessage] = []
    for msg in recent_messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def format_docs_with_metadata(docs: List[Document]) -> str:
    """Formatiert abgerufene Chunks inklusive Dateiname und Seitenzahl für den LLM-Kontext."""
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "Unbekanntes Dokument")
        page = doc.metadata.get("page_display")
        location = f"{source}, Seite {page}" if page else source
        formatted_chunks.append(f"[Quelle {i}: {location}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted_chunks)


def get_llm(
    model_name: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_CONTEXT_SIZE,
) -> ChatOllama:
    """Initialisiert das lokale Ollama-Chatmodell mit allen konfigurierten Parametern."""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
    )


def contextualize_query(
    query: str,
    chat_history: List[BaseMessage],
    llm: ChatOllama,
) -> str:
    """Löst Rückfragen mithilfe der Chat-History in eine eigenständige Retrieval-Query auf."""
    if not chat_history:
        return query

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = contextualize_prompt | llm | StrOutputParser()
    standalone_query = chain.invoke({"chat_history": chat_history, "input": query}).strip()

    if not standalone_query or len(standalone_query) > 250 or "\n" in standalone_query:
        last_human = next((m.content for m in reversed(chat_history) if isinstance(m, HumanMessage)), "")
        return f"{last_human} {query}".strip()

    return standalone_query


def retrieve_and_stream(
    query: str,
    chat_messages: List[Dict[str, str]],
    model_name: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_CONTEXT_SIZE,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> Tuple[Generator[str, None, None], List[Document], str]:
    """Führt Retrieval (inkl. Query-Contextualization) durch und gibt einen Token-Stream + Quellen zurück.

    Returns:
        Tuple (Token-Generator für st.write_stream, Liste der verwendeten Chunks, genutzte Suchanfrage)
    """
    llm = get_llm(model_name=model_name, temperature=temperature, num_ctx=num_ctx)
    chat_history = build_chat_history(chat_messages, max_turns=max_history)
    search_query = contextualize_query(query, chat_history, llm)

    vector_store = get_vector_store(embedding_model=embedding_model)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 3, 12)},
    )
    retrieved_docs = retriever.invoke(search_query)

    context_text = format_docs_with_metadata(retrieved_docs)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    rag_chain = qa_prompt | llm | StrOutputParser()

    def token_generator() -> Generator[str, None, None]:
        for chunk in rag_chain.stream(
            {
                "context": context_text,
                "chat_history": chat_history,
                "input": query,
            }
        ):
            yield chunk

    return token_generator(), retrieved_docs, search_query