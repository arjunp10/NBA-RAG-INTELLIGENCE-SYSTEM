"""Scrapes r/nba game threads via PRAW and saves raw text to data/raw/reddit/.

Pulls the 25 most recent posts whose titles contain "game thread" from r/nba.
Saves title + top-level comments for each thread.
Output files are named reddit_{YYYYMMDD}_{submission_id}.txt to avoid same-day collisions.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import praw
from dotenv import load_dotenv

from nba_types import REDDIT_RAW_DIR

load_dotenv()

logger = logging.getLogger(__name__)

_THREAD_LIMIT = 25
_COMMENT_LIMIT = 200  # top-level comments per thread


def _build_reddit_client() -> praw.Reddit:
    """Instantiate a read-only PRAW Reddit client from environment variables."""
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def _serialize_thread(submission: praw.models.Submission) -> str:
    """Flatten a Reddit submission into a plain-text string."""
    lines = [
        f"TITLE: {submission.title}",
        f"SCORE: {submission.score}",
        f"URL: https://www.reddit.com{submission.permalink}",
        "",
        "--- TOP-LEVEL COMMENTS ---",
        "",
    ]

    submission.comments.replace_more(limit=0)
    for comment in list(submission.comments)[:_COMMENT_LIMIT]:
        if hasattr(comment, "body") and comment.body not in ("[deleted]", "[removed]"):
            lines.append(comment.body.strip())
            lines.append("")

    return "\n".join(lines)


def scrape_reddit() -> None:
    """Search r/nba for the 25 most recent game threads and save to data/raw/reddit/."""
    output_dir = Path(REDDIT_RAW_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        reddit = _build_reddit_client()
    except KeyError as exc:
        logger.error("Missing Reddit credential in environment: %s", exc)
        return

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    saved = 0

    try:
        results = reddit.subreddit("nba").search(
            "game thread", sort="new", limit=_THREAD_LIMIT * 2  # fetch extra to cover filter loss
        )
        # Filter to posts whose title actually contains "game thread"
        submissions = [
            s for s in results
            if "game thread" in s.title.lower()
        ][:_THREAD_LIMIT]
    except Exception as exc:
        logger.error("Reddit search failed: %s", exc)
        return

    for submission in submissions:
        try:
            content = _serialize_thread(submission)
        except Exception as exc:
            logger.error("Failed to serialize thread %s: %s", submission.id, exc)
            continue

        scraped_at = datetime.now(timezone.utc).isoformat()
        # Use submission ID in filename to prevent same-day collisions
        filename = f"reddit_{date_str}_{submission.id}.txt"
        filepath = output_dir / filename

        header = (
            f"SOURCE_URL: https://www.reddit.com{submission.permalink}\n"
            f"SCRAPED_AT: {scraped_at}\n"
            f"SOURCE_TYPE: reddit\n\n"
        )
        filepath.write_text(header + content, encoding="utf-8")

        logger.info("Saved %s (%d chars)", filename, len(content))
        saved += 1

    logger.info("Reddit scrape complete: %d/%d threads saved", saved, len(submissions))
