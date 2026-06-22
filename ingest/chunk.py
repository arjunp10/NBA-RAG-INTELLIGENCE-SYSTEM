"""Reads raw scraped text files from data/raw/ and returns chunked LangChain Documents.

Parses the header written by the scrapers (SOURCE_URL, SCRAPED_AT, SOURCE_TYPE)
to populate Document metadata. Does not embed or write anywhere.

chunk_size and chunk_overlap in RetrievalConfig are measured in TOKENS
(via tiktoken cl100k_base), not characters.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from nba_types import RAW_DATA_DIR, RetrievalConfig

logger = logging.getLogger(__name__)


def _token_len(text: str) -> int:
    """Count tokens using tiktoken cl100k_base (same tokenizer as text-embedding-004)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough 4-chars-per-token estimate if tiktoken not installed
        return len(text) // 4


def _parse_raw_file(filepath: Path) -> Tuple[str, dict]:
    """Extract metadata header and body text from a scraped .txt file.

    The scrapers always write a fixed 3-line header followed by a blank line:
        Line 0: SOURCE_URL: <url>
        Line 1: SCRAPED_AT: <iso timestamp>
        Line 2: SOURCE_TYPE: espn|reddit
        Line 3: (blank)
        Line 4+: body text

    Falls back to scanning for the blank line for any legacy files.

    Returns:
        Tuple of (body_text, metadata_dict).
    """
    raw = filepath.read_text(encoding="utf-8")
    lines = raw.splitlines()
    metadata: dict = {}

    # Fast path: fixed 3-line header
    if (
        len(lines) > 4
        and lines[0].startswith("SOURCE_URL:")
        and lines[1].startswith("SCRAPED_AT:")
        and lines[2].startswith("SOURCE_TYPE:")
        and lines[3] == ""
    ):
        metadata["source"] = lines[0].split("SOURCE_URL:", 1)[1].strip()
        metadata["scraped_at"] = lines[1].split("SCRAPED_AT:", 1)[1].strip()
        metadata["source_type"] = lines[2].split("SOURCE_TYPE:", 1)[1].strip()
        body = "\n".join(lines[4:]).strip()
    else:
        # Fallback: scan for the first blank line after all three header keys are found
        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith("SOURCE_URL:"):
                metadata["source"] = line.split("SOURCE_URL:", 1)[1].strip()
            elif line.startswith("SCRAPED_AT:"):
                metadata["scraped_at"] = line.split("SCRAPED_AT:", 1)[1].strip()
            elif line.startswith("SOURCE_TYPE:"):
                metadata["source_type"] = line.split("SOURCE_TYPE:", 1)[1].strip()
            elif line == "" and len(metadata) >= 3:
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()

    metadata.setdefault("source", str(filepath))
    metadata.setdefault("source_type", "unknown")
    metadata.setdefault("scraped_at", "")

    return body, metadata


def chunk_documents(config: RetrievalConfig) -> List[Document]:
    """Read all raw .txt files and return Documents chunked per the given config.

    chunk_size and chunk_overlap are treated as token counts.

    Args:
        config: RetrievalConfig whose chunk_size and chunk_overlap define the split.

    Returns:
        List of LangChain Document objects with source, source_type, scraped_at,
        chunk_index, total_chunks, and source_file metadata.
    """
    raw_dir = Path(RAW_DATA_DIR)
    txt_files = list(raw_dir.rglob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in %s — run scrapers first", raw_dir)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", " ", ""],
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

        # Inject positional metadata so retrieval failures are debuggable
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["source_file"] = filepath.name

        all_docs.extend(chunks)
        logger.info(
            "Chunked %s → %d chunks (size=%d tokens, overlap=%d tokens)",
            filepath.name, len(chunks), config.chunk_size, config.chunk_overlap,
        )

    logger.info(
        "chunk_documents complete: %d files → %d total chunks for config '%s'",
        len(txt_files), len(all_docs), config.name,
    )
    return all_docs
