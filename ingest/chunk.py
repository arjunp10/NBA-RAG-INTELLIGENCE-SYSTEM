"""Reads raw scraped text files from data/raw/ and returns chunked LangChain Documents.

Parses the header written by the scrapers (SOURCE_URL, SCRAPED_AT, SOURCE_TYPE)
to populate Document metadata. Does not embed or write anywhere.
"""

import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from nba_types import RAW_DATA_DIR, RetrievalConfig

logger = logging.getLogger(__name__)


def _parse_raw_file(filepath: Path) -> tuple[str, dict]:
    """Extract metadata header and body text from a scraped .txt file.

    Returns:
        Tuple of (body_text, metadata_dict).
    """
    raw = filepath.read_text(encoding="utf-8")
    lines = raw.splitlines()
    metadata = {}
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("SOURCE_URL:"):
            metadata["source"] = line.split("SOURCE_URL:", 1)[1].strip()
        elif line.startswith("SCRAPED_AT:"):
            metadata["scraped_at"] = line.split("SCRAPED_AT:", 1)[1].strip()
        elif line.startswith("SOURCE_TYPE:"):
            metadata["source_type"] = line.split("SOURCE_TYPE:", 1)[1].strip()
        elif line == "" and i > 0 and "source" in metadata:
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:]).strip()
    if "source" not in metadata:
        metadata["source"] = str(filepath)
    if "source_type" not in metadata:
        metadata["source_type"] = "unknown"
    if "scraped_at" not in metadata:
        metadata["scraped_at"] = ""

    return body, metadata


def chunk_documents(config: RetrievalConfig) -> List[Document]:
    """Read all raw .txt files and return Documents chunked per the given config.

    Args:
        config: RetrievalConfig whose chunk_size and chunk_overlap define the split.

    Returns:
        List of LangChain Document objects with source, source_type, scraped_at metadata.
    """
    raw_dir = Path(RAW_DATA_DIR)
    txt_files = list(raw_dir.rglob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in %s — run scrapers first", raw_dir)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    all_docs: List[Document] = []
    for filepath in txt_files:
        try:
            body, metadata = _parse_raw_file(filepath)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", filepath, exc)
            continue

        if not body:
            logger.warning("Empty body in %s — skipping", filepath)
            continue

        chunks = splitter.create_documents([body], metadatas=[metadata])
        all_docs.extend(chunks)
        logger.info(
            "Chunked %s → %d chunks (size=%d, overlap=%d)",
            filepath.name, len(chunks), config.chunk_size, config.chunk_overlap,
        )

    logger.info(
        "chunk_documents complete: %d files → %d total chunks for config '%s'",
        len(txt_files), len(all_docs), config.name,
    )
    return all_docs
