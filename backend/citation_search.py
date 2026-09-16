import logging

import requests

_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,venue,abstract,url"
_logger = logging.getLogger("citation_search")


def search_papers(query: str, limit: int = 3) -> list[dict]:
    if not query.strip():
        return []
    try:
        resp = requests.get(
            _API,
            params={"query": query, "limit": limit, "fields": _FIELDS},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        _logger.warning("semantic scholar request failed for %r: %s", query, exc)
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
