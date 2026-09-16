import logging
import os
import xml.etree.ElementTree as ET

import requests

_OPENALEX_API = "https://api.openalex.org/works"
_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_logger = logging.getLogger("citation_search")


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))[:400]


def search_openalex(query: str, limit: int = 2) -> list[dict]:
    if not query.strip():
        return []
    params = {"search": query, "per_page": limit}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    try:
        resp = requests.get(_OPENALEX_API, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        _logger.warning("openalex request failed for %r: %s", query, exc)
        return []

    papers = []
    for w in resp.json().get("results", []):
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])
        ]
        source = w.get("primary_location") or {}
        papers.append(
            {
                "title": w.get("display_name") or "(제목 없음)",
                "authors": authors,
                "year": w.get("publication_year"),
                "venue": (source.get("source") or {}).get("display_name") or "",
                "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                "url": source.get("landing_page_url") or w.get("doi") or "",
                "provider": "OpenAlex",
            }
        )
    return papers


def search_arxiv(query: str, limit: int = 2) -> list[dict]:
    if not query.strip():
        return []
    params = {"search_query": f"all:{query}", "max_results": limit}
    try:
        resp = requests.get(_ARXIV_API, params=params, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        _logger.warning("arxiv request failed for %r: %s", query, exc)
        return []

    papers = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el = entry.find("atom:title", _ARXIV_NS)
        summary_el = entry.find("atom:summary", _ARXIV_NS)
        published_el = entry.find("atom:published", _ARXIV_NS)
        id_el = entry.find("atom:id", _ARXIV_NS)
        authors = [
            a.findtext("atom:name", default="", namespaces=_ARXIV_NS)
            for a in entry.findall("atom:author", _ARXIV_NS)
        ]
        papers.append(
            {
                "title": (title_el.text or "").strip().replace("\n", " ") if title_el is not None else "(제목 없음)",
                "authors": authors,
                "year": int(published_el.text[:4]) if published_el is not None and published_el.text else None,
                "venue": "arXiv",
                "abstract": (summary_el.text or "").strip().replace("\n", " ")[:400] if summary_el is not None else "",
                "url": id_el.text if id_el is not None else "",
                "provider": "arXiv",
            }
        )
    return papers


def search_papers(query: str, limit: int = 4) -> list[dict]:
    per_source = max(1, (limit + 1) // 2)
    papers = search_openalex(query, per_source) + search_arxiv(query, per_source)
    return papers[:limit]
