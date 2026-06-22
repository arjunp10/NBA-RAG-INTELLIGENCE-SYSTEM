"""Embeds chunked Documents into ChromaDB — one collection per RetrievalConfig.

Collection names are nba_docs_{config.name} (e.g. nba_docs_default).
Re-running clears each collection before re-inserting to avoid duplicates.

Raw files are parsed once and split per config to avoid redundant disk I/O.
"""

import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingest.chunk import chunk_documents
from nba_types import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_CONFIGS,
    RetrievalConfig,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instantiate the Google embedding model from the environment API key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def _collection_name(config: RetrievalConfig) -> str:
    """Return the ChromaDB collection name for a given config."""
    return f"{CHROMA_COLLECTION_NAME}_{config.name}"


def embed_config(
    config: RetrievalConfig,
    documents: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> int:
    """Embed a list of Documents into the ChromaDB collection for one config.

    Clears the existing collection before inserting to prevent duplicates.

    Args:
        config: The RetrievalConfig whose chunk_size was used to produce documents.
        documents: Chunked Documents to embed.
        embeddings: Shared embedding model instance.

    Returns:
        Number of documents successfully embedded.

    Raises:
        RuntimeError: If embedding succeeds but the collection ends up empty.
    """
    collection = _collection_name(config)

    try:
        existing = Chroma(
            collection_name=collection,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
        existing.delete_collection()
        logger.info("Cleared existing collection '%s'", collection)
    except Exception as exc:
        logger.warning("Could not clear collection '%s': %s", collection, exc)

    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Use public API to count — avoids private _collection attribute
        count = len(vectorstore.get(include=[])["ids"])
        if count == 0:
            raise RuntimeError(
                f"Embed succeeded but collection '{collection}' is empty — "
                "check API quota or document content."
            )
        logger.info("Embedded %d chunks into collection '%s'", count, collection)
        return count
    except Exception as exc:
        logger.error("Failed to embed into collection '%s': %s", collection, exc)
        raise


def run_ingest() -> None:
    """Chunk and embed all raw documents into all ChromaDB collections.

    Files are read from disk once per config (chunking parameters differ),
    but a single embedding model instance is shared across all configs.
    """
    logger.info("Starting ingest across %d configs", len(RETRIEVAL_CONFIGS))

    # Build embeddings once — shared across all configs
    embeddings = _get_embeddings()

    for config in RETRIEVAL_CONFIGS:
        logger.info(
            "Processing config '%s' (chunk_size=%d tokens, overlap=%d tokens, k=%d)",
            config.name, config.chunk_size, config.chunk_overlap, config.k,
        )
        documents = chunk_documents(config)
        if not documents:
            logger.warning("No documents to embed for config '%s' — skipping", config.name)
            continue
        try:
            count = embed_config(config, documents, embeddings)
            logger.info("Config '%s' complete: %d vectors stored", config.name, count)
        except RuntimeError as exc:
            logger.error("Ingest failed for config '%s': %s", config.name, exc)

    logger.info("Ingest complete for all configs")
