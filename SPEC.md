# NBA RAG System — Project Specification

## Purpose
A self-corrective RAG system that extracts qualitative insight from unstructured NBA media (game recaps, press conferences, Reddit threads) and answers questions no box score can answer.

## What This System Does
1. Scrapes NBA game recaps from ESPN and game threads from r/nba
2. Chunks and embeds all text into a local ChromaDB vector store
3. User asks a natural language question about NBA games, players, or trends
4. RAG pipeline retrieves relevant chunks and generates an answer via Gemini
5. RAGAS automatically scores the answer for faithfulness and relevance
6. If score is below threshold, LangGraph triggers a retry with different retrieval params
7. All queries, configs tried, and scores are logged to SQLite
8. Streamlit dashboard shows the answer, confidence score, and which config was used

## Stack

### LLM
- Model: `gemini-1.5-flash`
- Provider: Google AI Studio
- Library: `langchain-google-genai`

### Embeddings
- Model: `models/text-embedding-004`
- Provider: Google AI Studio
- Library: `langchain-google-genai`

### Vector Store
- ChromaDB (local, persistent)
- Persist directory: `./chroma_db`
- Collection name: `nba_docs`

### Orchestration
- LangChain — RAG chain construction
- LangGraph — self-corrective retry loop

### Evaluation
- RAGAS — faithfulness + answer relevance scoring
- Threshold for retry: 0.7 (scores below this trigger a retry)

### Data Collection
- `requests` + `BeautifulSoup` — ESPN game recap scraping
- `praw` — Reddit API for r/nba game threads

### Storage
- ChromaDB — vector store
- SQLite (`eval/logs.db`) — query + score logging
- `data/raw/` — raw scraped text files

### Frontend
- Streamlit — dashboard UI

### Utilities
- `pandas` — experiment result tracking
- `python-dotenv` — API key management

## API Keys Required
```
GOOGLE_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
```

## Data Sources
| Source | Method | What We Collect |
|--------|--------|-----------------|
| ESPN game recaps | BeautifulSoup scraper | Game narratives, analysis |
| r/nba game threads | PRAW Reddit API | Fan discussion, live reactions |

## Retrieval Configs (for self-correction)
| Config | k | chunk_size | chunk_overlap |
|--------|---|------------|---------------|
| default | 4 | 512 | 50 |
| wider | 8 | 512 | 50 |
| smaller_chunks | 4 | 256 | 25 |
| larger_chunks | 4 | 1024 | 100 |

## Scoring Thresholds
- Faithfulness < 0.7 → retry
- Answer relevance < 0.7 → retry
- Max retries: 3
- If still below threshold after 3 retries → return best result with low confidence flag

## Out of Scope
- No fine-tuning
- No PyTorch or TensorFlow
- No paid vector database (Pinecone etc)
- No authentication system
- No deployment (local only for now)
