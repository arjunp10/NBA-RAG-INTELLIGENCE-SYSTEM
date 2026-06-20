# Claude Code Bootstrap Prompt (CLI Version)

Paste this entire prompt into Claude Code to start the project.

---

I am building an NBA RAG Intelligence System as a CLI tool. Before writing any code, read these four anchor documents which are the single source of truth for this project:

- SPEC.md — what the system does, the full stack, all API keys needed
- ARCHITECTURE.md — how every component connects and what each file is responsible for
- TYPES.md — all dataclasses, constants, SQLite schema, and LangChain metadata schema
- CONVENTIONS.md — coding standards, allowed libraries, logging rules, what NOT to do

One change from the anchor docs: replace the Streamlit app with a CLI built using Python's `argparse` or `click` (your choice, pick whichever fits better). The CLI replaces `app/streamlit_app.py` with `app/cli.py`.

The CLI must support these commands:

```bash
python -m app.cli scrape espn        # runs data/scrape_espn.py
python -m app.cli scrape reddit      # runs data/scrape_reddit.py
python -m app.cli ingest             # runs chunk.py + embed.py, builds ChromaDB
python -m app.cli ask "your question here"   # runs the full self-corrective RAG pipeline
python -m app.cli logs               # prints a summary table of past queries from SQLite using pandas
python -m app.cli logs --last 10     # prints the last 10 queries
```

The `ask` command output must clearly show in the terminal:
- The question
- The final answer
- Faithfulness score and answer relevance score
- Which retrieval config was used
- How many retries it took
- A warning if low_confidence is True
- The retrieved chunks, truncated to first 150 characters each, under a "Sources" section

Use clean terminal formatting — consider using the `rich` library for colored output and tables (add it to requirements.txt and CONVENTIONS.md allowed libraries if you use it). Keep it readable, not cluttered.

All other documents (SPEC.md, ARCHITECTURE.md, TYPES.md, CONVENTIONS.md) apply exactly as written otherwise. Do not deviate from the dataclasses, constants, or pipeline logic defined there.

Once you have read all four documents, do the following in order:

1. Scaffold the full project structure as defined in ARCHITECTURE.md, with `app/cli.py` instead of `app/streamlit_app.py`
2. Create `requirements.txt` with all libraries from CONVENTIONS.md plus `rich` if used
3. Create `.env.example` with all keys from SPEC.md
4. Implement `data/scrape_espn.py` — scrapes ESPN NBA game recap URLs, saves raw text to `data/raw/espn/`
5. Implement `data/scrape_reddit.py` — uses PRAW to pull r/nba game threads, saves raw text to `data/raw/reddit/`

Use the exact dataclasses from TYPES.md. Follow all conventions from CONVENTIONS.md. After each file, confirm what you built and what comes next before proceeding.
