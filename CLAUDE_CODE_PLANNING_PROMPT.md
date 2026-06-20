# Claude Code Planning Prompt

Paste this into Claude Code BEFORE the build prompt. This makes Claude Code produce a plan first, so you can review/approve before any code gets written.

---

I am about to build an NBA RAG Intelligence System with you as a CLI tool. Before we write any code, I want you to read the four anchor documents in this folder and produce a plan — not code yet.

Read these in full:
- SPEC.md — what the system does, the full stack, all API keys needed
- ARCHITECTURE.md — how every component connects and what each file is responsible for
- TYPES.md — all dataclasses, constants, SQLite schema, and metadata schema
- CONVENTIONS.md — coding standards, allowed libraries, what NOT to do

Note: this will be a CLI tool, not Streamlit. The CLI replaces `app/streamlit_app.py` with `app/cli.py` and must support: `scrape espn`, `scrape reddit`, `ingest`, `ask "<question>"`, and `logs` commands.

Once you've read all four documents, produce a written plan with:

1. **Build order** — the exact sequence of files you'll create, in dependency order, with a one-line reason for each
2. **Open questions** — anything in the anchor docs that's ambiguous, underspecified, or that you'd need me to clarify before building (e.g. specific ESPN URLs to scrape, how many Reddit threads to pull, what counts as a "game recap" vs box score page)
3. **Risk flags** — anything in the spec you think could break, be rate-limited, or behave differently than expected (e.g. ESPN scraping fragility, Reddit API limits, RAGAS scoring against Gemini)
4. **Milestone checkpoints** — natural stopping points where you'll pause and show me working output before continuing to the next phase, instead of building the whole thing in one shot

Do not write any code yet. Do not scaffold any files yet. Just give me the plan so I can review and approve it first.
