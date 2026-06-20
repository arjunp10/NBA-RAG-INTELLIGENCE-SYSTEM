# NBA RAG System — Architecture

## Component Map

```
DATA COLLECTION
├── data/scrape_espn.py        → saves to data/raw/espn/
└── data/scrape_reddit.py      → saves to data/raw/reddit/

INGESTION
├── ingest/chunk.py            → reads data/raw/, chunks text into Documents
└── ingest/embed.py            → embeds Documents, writes to ChromaDB

PIPELINE
├── pipeline/retriever.py      → loads ChromaDB, exposes retriever interface
├── pipeline/chain.py          → builds LangChain RAG chain (retriever + Gemini)
└── pipeline/self_correct.py   → LangGraph graph that wraps chain with retry loop

EVALUATION
├── eval/scorer.py             → RAGAS scoring (faithfulness + answer relevance)
└── eval/logs.db               → SQLite log of every query, config, scores

APP
└── app/streamlit_app.py       → UI: text input → self_correct → display result + score
```

## Data Flow

```
User Question
      │
      ▼
pipeline/self_correct.py (LangGraph)
      │
      ├─── pipeline/chain.py (RAG chain)
      │         │
      │         ├─── pipeline/retriever.py (ChromaDB lookup)
      │         │         └─── returns top-k chunks
      │         │
      │         └─── Gemini 1.5 Flash (generates answer from chunks)
      │
      ├─── eval/scorer.py (RAGAS scores the answer)
      │         └─── faithfulness + answer_relevance
      │
      ├─── if score < 0.7 → retry with next config (max 3 retries)
      │
      └─── log to eval/logs.db → return result to Streamlit
```

## Module Responsibilities

### `data/scrape_espn.py`
- Input: list of ESPN game recap URLs
- Output: `.txt` files saved to `data/raw/espn/`
- Does NOT chunk or embed — raw text only

### `data/scrape_reddit.py`
- Input: subreddit `nba`, search query `"game thread"`
- Output: `.txt` files saved to `data/raw/reddit/`
- Does NOT chunk or embed — raw text only

### `ingest/chunk.py`
- Input: directory path to `data/raw/`
- Output: list of LangChain `Document` objects with metadata
- Uses `RecursiveCharacterTextSplitter`
- Metadata must include: `source`, `source_type` (espn or reddit), `scraped_at`

### `ingest/embed.py`
- Input: list of `Document` objects from chunk.py
- Output: populated ChromaDB collection
- Uses Google `text-embedding-004`
- Persist directory: `./chroma_db`
- Collection name: `nba_docs`

### `pipeline/retriever.py`
- Input: config dict with `k`, `chunk_size`, `chunk_overlap`
- Output: LangChain retriever object
- Loads existing ChromaDB collection
- Exposes `get_retriever(config: RetrievalConfig) -> BaseRetriever`

### `pipeline/chain.py`
- Input: retriever object
- Output: LangChain RAG chain
- Uses Gemini 1.5 Flash as LLM
- Uses `RetrievalQA.from_chain_type`

### `pipeline/self_correct.py`
- Input: user question string
- Output: `QueryResult` dataclass
- LangGraph graph with nodes: retrieve → score → check → retry or return
- Iterates through `RETRIEVAL_CONFIGS` from SPEC until score >= 0.7 or max retries hit

### `eval/scorer.py`
- Input: question, answer, retrieved contexts
- Output: `EvalScore` dataclass
- Uses RAGAS `faithfulness` and `answer_relevancy` metrics

### `app/streamlit_app.py`
- Single page UI
- Text input for question
- Displays: answer, faithfulness score, relevance score, config used, retry count
- Shows raw retrieved chunks in expander

## What Each File Must NOT Do
- Scrapers must not chunk or embed
- Chunker must not embed or call any LLM
- Retriever must not generate answers
- Scorer must not modify the pipeline
- No file should import from `app/`

## Config Flow
All retrieval configs are defined in `pipeline/self_correct.py` as a list of `RetrievalConfig` objects. Self-correction iterates through this list in order. Never hardcode config values anywhere else.
