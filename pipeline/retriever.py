"""Loads a ChromaDB collection and returns a LangChain retriever for a given config.

Each RetrievalConfig maps to its own collection (nba_docs_{config.name}).
"""

import logging
import os

from dotenv import load_dotenv
from langchain.schema.retriever import BaseRetriever
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from nba_types import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, RetrievalConfig

load_dotenv()

logger = logging.getLogger(__name__)


def get_retriever(config: RetrievalConfig) -> BaseRetriever:
    """Load the ChromaDB collection for the given config and return a retriever.

    Args:
        config: RetrievalConfig specifying which collection to load and how many chunks (k) to retrieve.

    Returns:
        A LangChain BaseRetriever configured for the given k.
    """
    collection_name = f"{CHROMA_COLLECTION_NAME}_{config.name}"

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": config.k})
        logger.info("Loaded collection '%s' with k=%d", collection_name, config.k)
        return retriever
    except Exception as exc:
        logger.error("Failed to load collection '%s': %s", collection_name, exc)
        raise
