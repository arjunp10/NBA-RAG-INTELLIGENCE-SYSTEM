# NBA RAG System — Type Definitions

These are the canonical data shapes for the entire project.
Claude Code must use these exactly. Do not invent new shapes.

## Python Dataclasses

```python
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class RetrievalConfig:
    name: str                  # e.g. "default", "wider", "smaller_chunks"
    k: int                     # number of chunks to retrieve
    chunk_size: int            # tokens per chunk
    chunk_overlap: int         # overlap between chunks

@dataclass
class RetrievedChunk:
    content: str               # raw text of the chunk
    source: str                # filename or URL it came from
    source_type: str           # "espn" or "reddit"
    score: float               # cosine similarity score from ChromaDB

@dataclass
class EvalScore:
    faithfulness: float        # 0.0 - 1.0, RAGAS faithfulness score
    answer_relevance: float    # 0.0 - 1.0, RAGAS answer_relevancy score
    passed: bool               # True if both scores >= 0.7

@dataclass
class QueryResult:
    question: str              # original user question
    answer: str                # generated answer from Gemini
    chunks: List[RetrievedChunk]   # chunks used to generate the answer
    score: EvalScore           # RAGAS evaluation scores
    config_used: RetrievalConfig   # which config produced this result
    retry_count: int           # how many retries were needed (0 = first try worked)
    low_confidence: bool       # True if max retries hit and score still below threshold
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ScrapedDocument:
    content: str               # raw text content
    source_url: str            # where it was scraped from
    source_type: str           # "espn" or "reddit"
    scraped_at: str            # ISO 8601 timestamp string
    filename: str              # what it was saved as in data/raw/
```

## Retrieval Configs (canonical list)

```python
RETRIEVAL_CONFIGS = [
    RetrievalConfig(name="default",        k=4,  chunk_size=512,  chunk_overlap=50),
    RetrievalConfig(name="wider",          k=8,  chunk_size=512,  chunk_overlap=50),
    RetrievalConfig(name="smaller_chunks", k=4,  chunk_size=256,  chunk_overlap=25),
    RetrievalConfig(name="larger_chunks",  k=4,  chunk_size=1024, chunk_overlap=100),
]
```

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    config_name TEXT NOT NULL,
    k INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    faithfulness REAL NOT NULL,
    answer_relevance REAL NOT NULL,
    passed INTEGER NOT NULL,        -- 0 or 1
    retry_count INTEGER NOT NULL,
    low_confidence INTEGER NOT NULL, -- 0 or 1
    timestamp TEXT NOT NULL
);
```

## LangChain Document Metadata Schema

Every `Document` object passed into ChromaDB must have this metadata:

```python
{
    "source": str,        # original URL or filename
    "source_type": str,   # "espn" or "reddit"
    "scraped_at": str,    # ISO 8601 timestamp
}
```

## Scoring Thresholds

```python
SCORE_THRESHOLD = 0.7
MAX_RETRIES = 3
```

## Constants

```python
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "nba_docs"
RAW_DATA_DIR = "./data/raw"
ESPN_RAW_DIR = "./data/raw/espn"
REDDIT_RAW_DIR = "./data/raw/reddit"
SQLITE_DB_PATH = "./eval/logs.db"
GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
```
