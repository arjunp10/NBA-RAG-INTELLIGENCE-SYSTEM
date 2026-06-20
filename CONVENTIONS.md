# NBA RAG System — Conventions

Claude Code must follow these rules for every file it writes. No exceptions.

## General Rules

- Use Python 3.11+
- Every file must have a module-level docstring explaining what it does
- No hardcoded API keys anywhere — always load from `.env` via `python-dotenv`
- No hardcoded config values — always import from `TYPES.md` constants
- Every function must have type hints on all parameters and return values
- Every function must have a docstring

## Imports

- Standard library imports first
- Third party imports second
- Local imports third
- Separate each group with a blank line

```python
# correct
import os
from datetime import datetime

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingest.chunk import chunk_documents
from pipeline.retriever import get_retriever
```

## Error Handling

- All scraper functions must catch `requests.exceptions.RequestException` and log the error, not crash
- All ChromaDB operations must be wrapped in try/except
- All RAGAS scoring must be wrapped in try/except — if scoring fails, return `EvalScore(faithfulness=0.0, answer_relevance=0.0, passed=False)`
- Never use bare `except:` — always catch specific exceptions

## Logging

- Use Python's built-in `logging` module, not `print()`
- Log level: `INFO` for normal operations, `WARNING` for retries, `ERROR` for failures
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

```python
# correct
import logging
logger = logging.getLogger(__name__)
logger.info(f"Scraped {len(docs)} documents from ESPN")

# wrong
print(f"Scraped {len(docs)} documents from ESPN")
```

## File Naming

- All Python files: `snake_case.py`
- All raw scraped data: `{source_type}_{YYYYMMDD}_{index}.txt`
  - e.g. `espn_20240315_001.txt`, `reddit_20240315_042.txt`
- ChromaDB persist dir: `./chroma_db`
- SQLite DB: `./eval/logs.db`

## Data Flow Rules

- Scrapers only write to `data/raw/` — they do not return data to callers
- `chunk.py` reads from `data/raw/` and returns `List[Document]` — it does not write anywhere
- `embed.py` takes `List[Document]` and writes to ChromaDB — it does not return documents
- Never skip steps in the pipeline — always scrape → chunk → embed in that order

## Allowed Libraries

Only these libraries may be used. Do not install or import anything not on this list:

```
langchain
langchain-google-genai
langchain-chroma
langchain-community
chromadb
langgraph
ragas
streamlit
pandas
plotly
praw
requests
beautifulsoup4
python-dotenv
datasets
sqlite3 (stdlib)
logging (stdlib)
dataclasses (stdlib)
typing (stdlib)
datetime (stdlib)
os (stdlib)
pathlib (stdlib)
json (stdlib)
```

## What Claude Code Must NOT Do

- Do not add PyTorch, TensorFlow, or any ML training library
- Do not add Pinecone, Weaviate, or any paid vector database
- Do not change the project structure defined in ARCHITECTURE.md
- Do not change the dataclass shapes defined in TYPES.md
- Do not create new constants — use the ones in TYPES.md
- Do not import from `app/` in any pipeline or ingest file
- Do not use `print()` for logging
- Do not hardcode any URLs, API keys, or config values
- Do not use async/await unless explicitly asked
- Do not add authentication or user management

## Streamlit UI Rules

- Single page only
- No `st.set_page_config` calls beyond setting the title
- Input: `st.text_input` for the question
- Button: `st.button("Ask")` to trigger the query
- Always show: answer, faithfulness score, relevance score, config used, retry count
- Always show retrieved chunks in `st.expander("Retrieved Context")`
- If `low_confidence=True`, show `st.warning("Low confidence answer — consider rephrasing")`

## Environment Setup

Always include these instructions in any setup documentation:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# fill in .env with your keys
```
