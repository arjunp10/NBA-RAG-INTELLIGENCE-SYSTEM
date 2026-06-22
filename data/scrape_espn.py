"""Scrapes ESPN NBA game recap pages and saves raw text to data/raw/espn/.

Usage (via CLI):
    python -m app.cli scrape espn --url URL1 --url URL2 ...

Each URL must be a direct link to an ESPN game recap article.
Output files are named espn_{YYYYMMDD}_{url_hash}.txt to avoid same-day collisions.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nba_types import ESPN_RAW_DIR

load_dotenv()

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_REQUEST_DELAY_SECONDS = 1.5   # polite delay between ESPN requests
_MIN_CONTENT_LENGTH = 200      # skip pages that returned less than this (paywalled etc.)


def _extract_article_text(html: str, url: str) -> str:
    """Parse article body text from ESPN game recap HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for selector in ("div.article-body", "div.story__body", "article", "main"):
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            if paragraphs:
                return "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    # Fallback: grab all <p> tags in the page body
    paragraphs = soup.find_all("p")
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if text:
        logger.warning("Used fallback paragraph extraction for %s", url)
        return text

    return soup.get_text(separator="\n", strip=True)


def scrape_espn(urls: List[str]) -> None:
    """Fetch each ESPN recap URL and save raw text to data/raw/espn/.

    Args:
        urls: List of ESPN game recap article URLs to scrape.
    """
    output_dir = Path(ESPN_RAW_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    saved = 0

    for idx, url in enumerate(urls):
        if idx > 0:
            time.sleep(_REQUEST_DELAY_SECONDS)

        try:
            response = requests.get(url, headers=_HEADERS, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch %s: %s", url, exc)
            continue

        content = _extract_article_text(response.text, url)
        if len(content.strip()) < _MIN_CONTENT_LENGTH:
            logger.warning(
                "Too little content from %s (%d chars) — skipping (possible paywall)",
                url, len(content.strip()),
            )
            continue

        scraped_at = datetime.now(timezone.utc).isoformat()
        url_hash = hashlib.sha1(url.encode()).hexdigest()[:8]
        filename = f"espn_{date_str}_{url_hash}.txt"
        filepath = output_dir / filename

        header = f"SOURCE_URL: {url}\nSCRAPED_AT: {scraped_at}\nSOURCE_TYPE: espn\n\n"
        filepath.write_text(header + content, encoding="utf-8")

        logger.info("Saved %s (%d chars)", filename, len(content))
        saved += 1

    logger.info("ESPN scrape complete: %d/%d URLs saved", saved, len(urls))
