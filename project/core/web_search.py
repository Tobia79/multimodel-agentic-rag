"""Web search fallback for low-confidence local retrieval (DuckDuckGo, no API key)."""

from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: Optional[int] = None) -> str:
    """Run a web search and return formatted snippets for the agent."""
    if not query or not query.strip():
        return "WEB_SEARCH_ERROR: query must not be empty"

    limit = max_results if max_results is not None else config.WEB_SEARCH_MAX_RESULTS
    limit = min(max(1, int(limit)), config.WEB_SEARCH_MAX_RESULTS)

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query.strip(), max_results=limit))

        if not results:
            return "NO_WEB_RESULTS"

        parts = []
        for i, row in enumerate(results, 1):
            title = (row.get("title") or "").strip()
            body = (row.get("body") or "").strip()
            href = (row.get("href") or row.get("link") or "").strip()
            parts.append(f"Result {i}:\nTitle: {title}\nURL: {href}\nSnippet: {body}")
        return "\n\n".join(parts)

    except ImportError:
        return (
            "WEB_SEARCH_ERROR: ddgs is not installed. "
            "Run: pip install ddgs"
        )
    except Exception as e:
        logger.exception("Web search failed for query=%r", query)
        return f"WEB_SEARCH_ERROR: {e}"
