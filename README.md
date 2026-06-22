# NBA RAG Intelligence System

A self-corrective Retrieval-Augmented Generation (RAG) CLI tool that answers qualitative questions about NBA games — the kind no box score can answer.

It scrapes ESPN game recaps and Reddit r/nba game threads, embeds everything into a local ChromaDB vector store, and answers natural language questions using Google Gemini 1.5 Flash. Every answer is automatically scored by RAGAS for faithfulness and relevance. If the score falls below 0.7, a LangGraph retry loop tries a different retrieval configuration (up to 3 retries). All queries and scores are logged to SQLite.

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 1.5 Flash (`langchain-google-genai`) |
| Embeddings | Google `text-embedding-004` |
| Vector Store | ChromaDB (local, persistent) |
| Orchestration | LangChain + LangGraph |
| Evaluation | RAGAS (faithfulness + answer relevance) |
| Data Collection | `requests` + BeautifulSoup (ESPN), PRAW (Reddit) |
| Storage | ChromaDB + SQLite |
| CLI | Python `argparse` + `rich` |

## CLI Commands

```bash
python -m app.cli scrape espn --url URL1 --url URL2   # scrape ESPN game recaps
python -m app.cli scrape reddit                        # scrape 25 recent r/nba game threads
python -m app.cli ingest                               # chunk + embed into ChromaDB
python -m app.cli ask "your question here"             # run the full RAG pipeline
python -m app.cli logs                                 # view past query log
python -m app.cli logs --last 10                       # view last 10 queries
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

### Required API Keys

```
GOOGLE_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
```

---

## Build Log

### 2026-06-20

**Project started.**

- Designed full system architecture: self-corrective RAG pipeline with LangGraph retry loop, 4 retrieval configs, RAGAS scoring, SQLite logging
- Created project scaffold: directory structure, `__init__.py` files, `.gitignore`, `.env.example`
- Wrote `nba_types.py` — canonical dataclasses (`RetrievalConfig`, `QueryResult`, `EvalScore`, `ScrapedDocument`) and all constants
- Wrote `requirements.txt` with full dependency list
- Implemented `data/scrape_espn.py` — scrapes ESPN game recap URLs passed via CLI args, saves `espn_YYYYMMDD_NNN.txt` to `data/raw/espn/`
- Implemented `data/scrape_reddit.py` — pulls 25 most recent r/nba game threads via PRAW, saves `reddit_YYYYMMDD_NNN.txt` to `data/raw/reddit/`
- Initialized GitHub repo and pushed M1 milestone

### 2026-06-21

- Implemented `ingest/chunk.py` — reads all scraped `.txt` files, parses scraper-written headers for metadata (`source`, `source_type`, `scraped_at`), splits text using `RecursiveCharacterTextSplitter` per config's `chunk_size` and `chunk_overlap`
- Implemented `ingest/embed.py` — loops all 4 `RetrievalConfigs`, clears and repopulates a dedicated ChromaDB collection per config (`nba_docs_default`, `nba_docs_wider`, `nba_docs_smaller_chunks`, `nba_docs_larger_chunks`); clears before re-inserting to prevent duplicates
- M2 complete: full ingest pipeline ready

### 2026-06-22

- Implemented `pipeline/retriever.py` — loads the correct ChromaDB collection (`nba_docs_{config.name}`) for a given `RetrievalConfig` and returns a LangChain retriever with `k` chunks
- Implemented `pipeline/chain.py` — builds a `RetrievalQA` chain backed by Gemini 1.5 Flash with a custom NBA analyst prompt
- Implemented `eval/scorer.py` — runs RAGAS `faithfulness` and `answer_relevancy` scoring; configured to use Gemini + Google embeddings as the judge (no OpenAI dependency)
- Implemented `eval/logger.py` — creates `eval/logs.db` SQLite on first run, writes every `QueryResult` to `query_logs`, exposes `fetch_logs(limit)` for the CLI
- Implemented `pipeline/self_correct.py` — LangGraph state graph with generate → score → decide → (retry or finalize) loop; iterates through all 4 `RETRIEVAL_CONFIGS`, tracks best result, marks `low_confidence=True` if threshold never reached
- M3–M4 complete: full self-corrective RAG pipeline ready

### 2026-06-22 (continued)

- Implemented `app/cli.py` — full CLI with all 5 commands using `argparse` + `rich`
  - `scrape espn --url ...` — passes URLs to scraper, reports count
  - `scrape reddit` — runs PRAW scraper
  - `ingest` — triggers full 4-collection embed pipeline
  - `ask "question"` — runs self-corrective RAG, displays answer in colored panel, scores, config used, retry count, and truncated source chunks
  - `logs` / `logs --last N` — renders a rich table of past queries from SQLite; scores colored green/red by threshold
- Added `app/__main__.py` so `python -m app.cli` dispatches correctly
- M5 complete: system is fully buildable end-to-end

### 2026-06-22 (post-review fixes)

Full review by 6 specialized agents (rag-architect, ingestion-engineer, retrieval-engineer, llm-engineer, rag-evaluator, code-reviewer). Applied all critical and high-priority findings:

**Critical fixes**
- `eval/scorer.py` — migrated to RAGAS v0.2+ API; LLM/embeddings now passed directly to `evaluate()` instead of mutating global metric singletons; added `_safe_float()` to handle list-or-scalar return values and NaN; pinned `ragas>=0.2,<0.3` in requirements.txt
- `nba_types.py` — `MAX_RETRIES` raised from 3 → 4 so all 4 `RetrievalConfig` entries are reachable
- `pipeline/self_correct.py` — `_app` no longer compiled at module import time (lazy-init in `run_query`); `finalize_node` now uses `best.retry_count` instead of `state["retry_count"]`; `decide_node` simplified to drive purely off `config_index`; retriever/chain instances cached via `@lru_cache`
- `eval/logger.py` — SQL injection fixed (`LIMIT ?` parameterized query); connections managed via `with` context manager to prevent leaks; `_get_connection` refactored into `_ensure_schema`
- `pipeline/chain.py`, `pipeline/retriever.py`, `ingest/embed.py`, `eval/scorer.py` — `os.environ["GOOGLE_API_KEY"]` replaced with `.get()` + explicit `RuntimeError` with actionable message

**High-priority fixes**
- `ingest/chunk.py` — switched to token-based splitting via `tiktoken cl100k_base`; added `chunk_index`, `total_chunks`, `source_file` metadata per chunk; header parser made positional (fixed 3-line header fast path + robust fallback)
- `ingest/embed.py` — replaced private `_collection.count()` with public `vectorstore.get(include=[])["ids"]`; raises `RuntimeError` on empty collection after embed; embedding model instantiated once and shared across all configs
- `data/scrape_espn.py` — filenames now use URL hash to prevent same-day collisions; added 1.5s rate-limit delay between requests; minimum content length guard (200 chars) to skip paywalled responses; removed dead `ScrapedDocument` usage; `datetime.utcnow()` → `datetime.now(timezone.utc)`
- `data/scrape_reddit.py` — filenames now use submission ID; fetch 2× limit then filter to posts with "game thread" in title; removed dead `ScrapedDocument` usage; `datetime.utcnow()` → `datetime.now(timezone.utc)`
- `pipeline/chain.py` — `temperature` lowered from 0.2 → 0.0 for deterministic factual answers

**Low-priority fixes**
- `app/cli.py` — `_noop_context` replaced with `contextlib.nullcontext`; hardcoded `0.7` thresholds replaced with `SCORE_THRESHOLD` from `nba_types`
- `requirements.txt` — added `tiktoken` for token-based chunking
