import logging
import os
import time

import requests

_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,venue,abstract,url"
_logger = logging.getLogger("citation_search")


def _headers() -> dict:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def search_papers(query: str, limit: int = 3) -> list[dict]:
    if not query.strip():
        return []

    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(
                _API,
                params={"query": query, "limit": limit, "fields": _FIELDS},
                headers=_headers(),
                timeout=10,
            )
            if resp.status_code == 429:
                _logger.warning("semantic scholar rate-limited (attempt %d) for %r", attempt + 1, query)
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            _logger.warning("semantic scholar request failed for %r: %s", query, exc)
            return []
    else:
        return []

    data = resp.json().get("data", [])
    papers = []
    for p in data:
        authors = [a.get("name", "") for a in (p.get("authors") or [])]
        papers.append(
            {
                "title": p.get("title") or "(제목 없음)",
                "authors": authors,
                "year": p.get("year"),
                "venue": p.get("venue") or "",
                "abstract": p.get("abstract") or "",
                "url": p.get("url") or "",
            }
        )
    return papers
