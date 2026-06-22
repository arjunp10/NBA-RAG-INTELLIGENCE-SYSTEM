"""NBA RAG Intelligence System — CLI entry point.

Usage:
    python -m app.cli scrape espn --url URL1 --url URL2
    python -m app.cli scrape reddit
    python -m app.cli ingest
    python -m app.cli ask "your question here"
    python -m app.cli logs
    python -m app.cli logs --last 10
"""

import argparse
import logging
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()

# Configure logging to use rich handler for clean terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_scrape_espn(urls: list[str]) -> None:
    """Scrape ESPN game recap pages for the given URLs."""
    from data.scrape_espn import scrape_espn

    if not urls:
        console.print("[red]Error:[/red] At least one --url is required.")
        sys.exit(1)

    console.print(f"[bold cyan]Scraping {len(urls)} ESPN recap(s)...[/bold cyan]")
    scrape_espn(urls)
    console.print("[bold green]ESPN scrape complete.[/bold green]")


def cmd_scrape_reddit() -> None:
    """Scrape 25 most recent r/nba game threads."""
    from data.scrape_reddit import scrape_reddit

    console.print("[bold cyan]Scraping r/nba game threads...[/bold cyan]")
    scrape_reddit()
    console.print("[bold green]Reddit scrape complete.[/bold green]")


def cmd_ingest() -> None:
    """Chunk and embed all raw data into 4 ChromaDB collections."""
    from ingest.embed import run_ingest

    console.print("[bold cyan]Starting ingest across 4 retrieval configs...[/bold cyan]")
    console.print("[dim]This embeds into nba_docs_default, nba_docs_wider, nba_docs_smaller_chunks, nba_docs_larger_chunks[/dim]")
    run_ingest()
    console.print("[bold green]Ingest complete.[/bold green]")


def cmd_ask(question: str) -> None:
    """Run the self-corrective RAG pipeline and display results."""
    from pipeline.self_correct import run_query

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    console.print("[dim]Running self-corrective RAG pipeline...[/dim]")

    result = run_query(question)

    # --- Answer ---
    answer_style = "red" if result.low_confidence else "green"
    console.print(Panel(
        result.answer,
        title="[bold]Answer[/bold]",
        border_style=answer_style,
        padding=(1, 2),
    ))

    if result.low_confidence:
        console.print("[bold yellow]⚠ Low confidence answer — consider rephrasing your question.[/bold yellow]\n")

    # --- Scores & metadata ---
    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    meta.add_column("Field", style="bold")
    meta.add_column("Value")

    faith_color = "green" if result.score.faithfulness >= 0.7 else "red"
    rel_color = "green" if result.score.answer_relevance >= 0.7 else "red"

    meta.add_row("Faithfulness",     f"[{faith_color}]{result.score.faithfulness:.3f}[/{faith_color}]")
    meta.add_row("Answer Relevance", f"[{rel_color}]{result.score.answer_relevance:.3f}[/{rel_color}]")
    meta.add_row("Config used",      result.config_used.name)
    meta.add_row("Retries",          str(result.retry_count))
    meta.add_row("Timestamp",        result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))

    console.print(meta)

    # --- Sources ---
    if result.chunks:
        with console.pager() if len(result.chunks) > 6 else _noop_context():
            console.print("[bold]Sources[/bold]")
            for i, chunk in enumerate(result.chunks, 1):
                preview = chunk.content[:150].replace("\n", " ")
                if len(chunk.content) > 150:
                    preview += "..."
                source_label = f"[dim]{chunk.source_type}[/dim] {chunk.source}"
                console.print(f"  [cyan]{i}.[/cyan] {source_label}")
                console.print(f"     [dim]{preview}[/dim]")


def cmd_logs(last: int | None) -> None:
    """Print a summary table of past queries from SQLite."""
    from eval.logger import fetch_logs

    rows = fetch_logs(limit=last)

    if not rows:
        console.print("[yellow]No query logs found. Run [bold]ask[/bold] first.[/yellow]")
        return

    table = Table(
        title=f"Query Log{f' — last {last}' if last else ''}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID",         style="dim",    width=4)
    table.add_column("Question",                   width=36)
    table.add_column("Config",     style="cyan",   width=16)
    table.add_column("Faith",      justify="right", width=7)
    table.add_column("Relevance",  justify="right", width=9)
    table.add_column("Retries",    justify="right", width=7)
    table.add_column("Confidence", justify="center", width=10)
    table.add_column("Timestamp",  style="dim",    width=20)

    for row in rows:
        question_preview = row["question"][:33] + "..." if len(row["question"]) > 36 else row["question"]
        faith_color = "green" if row["faithfulness"] >= 0.7 else "red"
        rel_color = "green" if row["answer_relevance"] >= 0.7 else "red"
        confidence = "[red]LOW[/red]" if row["low_confidence"] else "[green]OK[/green]"

        table.add_row(
            str(row["id"]),
            question_preview,
            row["config_name"],
            f"[{faith_color}]{row['faithfulness']:.3f}[/{faith_color}]",
            f"[{rel_color}]{row['answer_relevance']:.3f}[/{rel_color}]",
            str(row["retry_count"]),
            confidence,
            row["timestamp"][:19],
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Context manager helper (avoids importing contextlib for a one-liner)
# ---------------------------------------------------------------------------

class _noop_context:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="NBA RAG Intelligence System CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    scrape_parser = sub.add_parser("scrape", help="Scrape raw data from ESPN or Reddit")
    scrape_sub = scrape_parser.add_subparsers(dest="source", required=True)

    espn_parser = scrape_sub.add_parser("espn", help="Scrape ESPN game recap URLs")
    espn_parser.add_argument(
        "--url", dest="urls", action="append", metavar="URL",
        help="ESPN game recap URL (repeatable)",
    )

    scrape_sub.add_parser("reddit", help="Scrape 25 most recent r/nba game threads")

    # ingest
    sub.add_parser("ingest", help="Chunk and embed raw data into ChromaDB")

    # ask
    ask_parser = sub.add_parser("ask", help="Ask a question about NBA games")
    ask_parser.add_argument("question", type=str, help="Natural language question")

    # logs
    logs_parser = sub.add_parser("logs", help="View past query logs")
    logs_parser.add_argument(
        "--last", type=int, default=None, metavar="N",
        help="Show only the last N queries",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scrape":
        if args.source == "espn":
            cmd_scrape_espn(args.urls or [])
        elif args.source == "reddit":
            cmd_scrape_reddit()

    elif args.command == "ingest":
        cmd_ingest()

    elif args.command == "ask":
        cmd_ask(args.question)

    elif args.command == "logs":
        cmd_logs(args.last)


if __name__ == "__main__":
    main()
