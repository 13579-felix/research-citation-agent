import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

_OPENALEX_API = "https://api.openalex.org/works"
_CROSSREF_API = "https://api.crossref.org/works"
_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,venue,abstract,url,externalIds"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_TIMEOUT = 10
_logger = logging.getLogger("citation_search")


class SearchError(Exception):
    """A provider failed; the message is shown to the user as-is."""


def _contact_email() -> str:
    return os.environ.get("OPENALEX_MAILTO", "")


def _user_agent() -> str:
    email = _contact_email()
    return f"research-citation-agent/1.0 (mailto:{email})" if email else "research-citation-agent/1.0"


def _get(url: str, provider: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": _user_agent(), **kwargs.pop("headers", {})}
    try:
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise SearchError(f"{provider} 요청 실패: {exc.__class__.__name__}") from exc
    if resp.status_code != 200:
        raise SearchError(f"{provider} HTTP {resp.status_code}: {resp.text[:120]}")
    return resp


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, limit: int) -> list[dict]:
    params = {"search": query, "per_page": limit}
    if email := _contact_email():
        params["mailto"] = email
    if api_key := os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = api_key
    resp = _get(_OPENALEX_API, "OpenAlex", params=params)

    papers = []
    for w in resp.json().get("results", []):
        source = w.get("primary_location") or {}
        papers.append(
            {
                "title": w.get("display_name") or "(제목 없음)",
                "authors": [
                    a.get("author", {}).get("display_name", "")
                    for a in w.get("authorships", [])
                ],
                "year": w.get("publication_year"),
                "venue": (source.get("source") or {}).get("display_name") or "",
                "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                "url": w.get("doi") or source.get("landing_page_url") or "",
                "doi": (w.get("doi") or "").removeprefix("https://doi.org/"),
                "provider": "OpenAlex",
            }
        )
    return papers


_JATS_TAG = re.compile(r"<[^>]+>")


def search_crossref(query: str, limit: int) -> list[dict]:
    params = {
        "query.bibliographic": query,
        "rows": limit,
        "select": "DOI,title,author,issued,container-title,abstract,URL",
    }
    if email := _contact_email():
        params["mailto"] = email
    resp = _get(_CROSSREF_API, "Crossref", params=params)

    papers = []
    for item in resp.json().get("message", {}).get("items", []):
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        doi = item.get("DOI", "")
        papers.append(
            {
                "title": " ".join(item.get("title") or []) or "(제목 없음)",
                "authors": [
                    " ".join(filter(None, [a.get("given"), a.get("family")]))
                    for a in item.get("author", [])
                ],
                "year": issued[0][0] if issued and issued[0] else None,
                "venue": " ".join(item.get("container-title") or []),
                "abstract": _JATS_TAG.sub("", item.get("abstract") or "").strip(),
                "url": f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                "doi": doi,
                "provider": "Crossref",
            }
        )
    return papers


def search_arxiv(query: str, limit: int) -> list[dict]:
    # "all:FeFET all:HZO" would OR the terms; AND them so every term matters.
    search_query = " AND ".join(f"all:{t}" for t in query.split())
    resp = _get(_ARXIV_API, "arXiv", params={"search_query": search_query, "max_results": limit})
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise SearchError("arXiv 응답 파싱 실패") from exc

    papers = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        published = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS)
        papers.append(
            {
                "title": " ".join(entry.findtext("atom:title", default="", namespaces=_ARXIV_NS).split())
                or "(제목 없음)",
                "authors": [
                    a.findtext("atom:name", default="", namespaces=_ARXIV_NS)
                    for a in entry.findall("atom:author", _ARXIV_NS)
                ],
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "venue": "arXiv",
                "abstract": " ".join(entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS).split()),
                "url": entry.findtext("atom:id", default="", namespaces=_ARXIV_NS),
                "doi": "",
                "provider": "arXiv",
            }
        )
    return papers


def search_semantic_scholar(query: str, limit: int) -> list[dict]:
    """Only used when SEMANTIC_SCHOLAR_API_KEY is set: the keyless shared
    pool returns 429 too often to be useful (see git history)."""
    headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]}
    params = {"query": query, "limit": limit, "fields": _S2_FIELDS}
    for attempt in range(3):
        try:
            resp = _get(_S2_API, "Semantic Scholar", params=params, headers=headers)
            break
        except SearchError as exc:
            if "HTTP 429" not in str(exc) or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

    papers = []
    for p in resp.json().get("data", []):
        doi = (p.get("externalIds") or {}).get("DOI", "")
        papers.append(
            {
                "title": p.get("title") or "(제목 없음)",
                "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                "year": p.get("year"),
                "venue": p.get("venue") or "",
                "abstract": p.get("abstract") or "",
                "url": f"https://doi.org/{doi}" if doi else p.get("url") or "",
                "doi": doi,
                "provider": "Semantic Scholar",
            }
        )
    return papers


def _providers() -> dict:
    providers = {
        "OpenAlex": search_openalex,
        "Crossref": search_crossref,
        "arXiv": search_arxiv,
    }
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        providers["Semantic Scholar"] = search_semantic_scholar
    return providers


def _dedup_key(paper: dict) -> str:
    if paper.get("doi"):
        return paper["doi"].lower()
    return re.sub(r"\W+", "", paper["title"].lower())


def search_papers(query: str, limit: int = 6, per_source: int = 3) -> tuple[list[dict], dict[str, str]]:
    """Search every provider in parallel.

    Returns (papers, status) where status maps provider name to "ok (N건)"
    or an error message, so a silently failing provider (e.g. OpenAlex
    returning nothing on the deploy server) is visible in the UI and logs
    instead of looking like "no papers exist".
    """
    if not query.strip():
        return [], {}

    providers = _providers()
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {name: pool.submit(fn, query, per_source) for name, fn in providers.items()}

    results: dict[str, list[dict]] = {}
    status: dict[str, str] = {}
    for name, fut in futures.items():
        try:
            results[name] = fut.result()
            status[name] = f"ok ({len(results[name])}건)"
        except Exception as exc:  # one provider failing must not sink the rest
            results[name] = []
            status[name] = str(exc) if isinstance(exc, SearchError) else f"{name} 오류: {exc}"
            _logger.warning("%s search failed for %r: %s", name, query, exc)
    _logger.info("search %r -> %s", query, status)

    # Round-robin across providers so one source can't crowd out the others.
    merged, seen = [], set()
    for i in range(per_source):
        for name in providers:
            if i < len(results[name]):
                paper = results[name][i]
                key = _dedup_key(paper)
                if key not in seen:
                    seen.add(key)
                    merged.append(paper)
    return merged[:limit], status
