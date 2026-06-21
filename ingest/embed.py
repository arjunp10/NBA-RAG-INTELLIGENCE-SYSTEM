"""Embeds chunked Documents into ChromaDB — one collection per RetrievalConfig.

Collection names are nba_docs_{config.name} (e.g. nba_docs_default).
Re-running clears each collection before re-inserting to avoid duplicates.
"""

import logging
import os
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
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def _collection_name(config: RetrievalConfig) -> str:
    """Return the ChromaDB collection name for a given config."""
    return f"{CHROMA_COLLECTION_NAME}_{config.name}"


def embed_config(config: RetrievalConfig, documents: List[Document]) -> int:
    """Embed a list of Documents into the ChromaDB collection for one config.

    Clears the existing collection before inserting to prevent duplicates.

    Args:
        config: The RetrievalConfig whose chunk_size was used to produce documents.
        documents: Chunked Documents to embed.

    Returns:
        Number of documents successfully embedded.
    """
    collection = _collection_name(config)
    embeddings = _get_embeddings()

    try:
        # Delete existing collection to avoid duplicates on re-ingest
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
        count = vectorstore._collection.count()
        logger.info(
            "Embedded %d chunks into collection '%s'", count, collection
        )
        return count
    except Exception as exc:
        logger.error("Failed to embed into collection '%s': %s", collection, exc)
        return 0


def run_ingest() -> None:
    """Chunk and embed all raw documents into all 4 ChromaDB collections."""
    logger.info("Starting ingest across %d configs", len(RETRIEVAL_CONFIGS))

    for config in RETRIEVAL_CONFIGS:
        logger.info(
            "Processing config '%s' (chunk_size=%d, overlap=%d, k=%d)",
            config.name, config.chunk_size, config.chunk_overlap, config.k,
        )
        documents = chunk_documents(config)
        if not documents:
            logger.warning("No documents to embed for config '%s' — skipping", config.name)
            continue
        count = embed_config(config, documents)
        logger.info("Config '%s' complete: %d vectors stored", config.name, count)

    logger.info("Ingest complete for all configs")
