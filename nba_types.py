"""Canonical dataclasses, constants, and config definitions for the NBA RAG system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class RetrievalConfig:
    name: str
    k: int
    chunk_size: int
    chunk_overlap: int


@dataclass
class RetrievedChunk:
    content: str
    source: str
    source_type: str
    score: float


@dataclass
class EvalScore:
    faithfulness: float
    answer_relevance: float
    passed: bool


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: List[RetrievedChunk]
    score: EvalScore
    config_used: RetrievalConfig
    retry_count: int
    low_confidence: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScrapedDocument:
    content: str
    source_url: str
    source_type: str
    scraped_at: str
    filename: str


# Canonical retrieval configs — self-correction iterates through these in order
RETRIEVAL_CONFIGS: List[RetrievalConfig] = [
    RetrievalConfig(name="default",        k=4, chunk_size=512,  chunk_overlap=50),
    RetrievalConfig(name="wider",          k=8, chunk_size=512,  chunk_overlap=50),
    RetrievalConfig(name="smaller_chunks", k=4, chunk_size=256,  chunk_overlap=25),
    RetrievalConfig(name="larger_chunks",  k=4, chunk_size=1024, chunk_overlap=100),
]

# Scoring thresholds
SCORE_THRESHOLD: float = 0.7
MAX_RETRIES: int = 4  # one attempt per RetrievalConfig

# Path constants
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "nba_docs"  # base name — suffixed per config
RAW_DATA_DIR: str = "./data/raw"
ESPN_RAW_DIR: str = "./data/raw/espn"
REDDIT_RAW_DIR: str = "./data/raw/reddit"
SQLITE_DB_PATH: str = "./eval/logs.db"

# Model constants
GEMINI_MODEL: str = "gemini-1.5-flash"
EMBEDDING_MODEL: str = "models/text-embedding-004"
