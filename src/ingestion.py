import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    SAMPLE_DOCS_DIR,
    UPLOADED_DOCS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Erstellt die lokale Ollama-Embedding-Instanz."""
    return OllamaEmbeddings(model=model_name)


def get_vector_store(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Chroma:
    """Gibt die persistente ChromaDB-Instanz zurück."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(embedding_model),
        persist_directory=str(CHROMA_DB_DIR),
    )


def delete_existing_source(source_filename: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Entfernt vorhandene Chunks einer Datei, um Duplikate bei Re-Indexierung zu vermeiden."""
    vector_store = get_vector_store(embedding_model)
    try:
        vector_store._collection.delete(where={"source_file": source_filename})
    except Exception:
        pass


def load_single_file(file_path: Path) -> List[Document]:
    """Lädt eine einzelne PDF-, TXT- oder MD-Datei und normalisiert deren Metadaten."""
    suffix = file_path.suffix.lower()
    docs: List[Document] = []

    try:
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "source_file": file_path.name,
                                "page": idx,
                                "page_display": idx + 1,
                            },
                        )
                    )
        elif suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "source_file": file_path.name,
                            "page_display": None,
                        },
                    )
                )
        else:
            logger.warning(f"Nicht unterstütztes Dateiformat übersprungen: {file_path.name}")
            return []

        logger.info(f"Geladen: {file_path.name} ({len(docs)} Seiten/Abschnitte)")
        return docs

    except Exception as exc:
        logger.error(f"Fehler beim Laden von {file_path.name}: {exc}")
        raise RuntimeError(f"Datei '{file_path.name}' konnte nicht verarbeitet werden: {exc}") from exc


def split_documents(documents: List[Document]) -> List[Document]:
    """Zerlegt Dokumente mit Überlappung in semantisch sinnvolle Chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    logger.info(f"{len(documents)} Seiten/Dokumente in {len(valid_chunks)} Chunks aufgeteilt.")
    return valid_chunks


def ingest_directory(
    directory: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[int, int]:
    """Indexiert alle unterstützten Dateien eines Verzeichnisses idempotent in ChromaDB.

    Returns:
        Tuple (Anzahl verarbeiteter Dateien, Anzahl erzeugter Chunks)
    """
    if not directory.exists():
        logger.warning(f"Verzeichnis existiert nicht: {directory}")
        return 0, 0

    supported_files = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md"}
    ]

    if not supported_files:
        logger.info(f"Keine unterstützten Dateien in {directory} gefunden.")
        return 0, 0

    all_docs: List[Document] = []
    processed_files = 0

    for file_path in supported_files:
        docs = load_single_file(file_path)
        if docs:
            delete_existing_source(file_path.name, embedding_model)
            all_docs.extend(docs)
            processed_files += 1

    if not all_docs:
        return 0, 0

    chunks = split_documents(all_docs)
    vector_store = get_vector_store(embedding_model)
    vector_store.add_documents(chunks)

    logger.info(f"Indexierung abgeschlossen: {processed_files} Dateien, {len(chunks)} Chunks gespeichert.")
    return processed_files, len(chunks)


def ingest_uploaded_files(
    uploaded_files,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[int, int]:
    """Speichert hochgeladene Streamlit-Dateien lokal und indexiert sie idempotent in ChromaDB."""
    all_docs: List[Document] = []
    processed_files = 0

    for uploaded_file in uploaded_files:
        target_path = UPLOADED_DOCS_DIR / uploaded_file.name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_single_file(target_path)
        if docs:
            delete_existing_source(uploaded_file.name, embedding_model)
            all_docs.extend(docs)
            processed_files += 1

    if not all_docs:
        return 0, 0

    chunks = split_documents(all_docs)
    vector_store = get_vector_store(embedding_model)
    vector_store.add_documents(chunks)

    return processed_files, len(chunks)


def reset_vector_store() -> None:
    """Löscht die bestehende ChromaDB-Collection und alle hochgeladenen Dateien sauber."""
    if CHROMA_DB_DIR.exists():
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Chroma-Collection '{COLLECTION_NAME}' gelöscht.")
        except Exception:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

    if UPLOADED_DOCS_DIR.exists():
        for file_path in UPLOADED_DOCS_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()


def get_indexed_stats() -> dict:
    """Liefert Statistiken über aktuell indexierte Chunks und Quelldateien."""
    if not CHROMA_DB_DIR.exists():
        return {"chunk_count": 0, "sources": []}

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            return {"chunk_count": 0, "sources": []}

        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        sources = sorted({m.get("source_file", "Unbekannt") for m in metadatas if m})
        return {"chunk_count": count, "sources": sources}
    except Exception:
        return {"chunk_count": 0, "sources": []}


if __name__ == "__main__":
    logger.info("Starte initiale Indexierung der Beispieldokumente...")
    files_count, chunks_count = ingest_directory(SAMPLE_DOCS_DIR)
    logger.info(f"Fertig! {files_count} Dateien -> {chunks_count} Chunks in ChromaDB.")