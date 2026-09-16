import json
import os

from claim_extractor import build_query, split_sentences
from citation_search import search_papers

_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
_anthropic_client = None


def _get_client():
    """Lazily build an Anthropic client if an API key is configured.

    The app runs fine without one (falls back to a heuristic judge) so a
    missing key never blocks the demo.
    """
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from anthropic import Anthropic

    _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _refine_query_with_claude(sentence: str, fallback_query: str) -> str:
    client = _get_client()
    if client is None:
        return fallback_query
    try:
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=60,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "다음은 연구 글쓰기 초안의 한 문장이다. 이 문장의 주장을 "
                        "뒷받침할 선행연구를 학술 검색엔진(Semantic Scholar)에서 "
                        "찾기 위한 영어 검색 쿼리를 3~6개 키워드로만 출력하라. "
                        "다른 설명 없이 쿼리만 출력할 것.\n\n"
                        f"문장: {sentence}"
                    ),
                }
            ],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return text or fallback_query
    except Exception:
        return fallback_query


def _judge_with_claude(sentence: str, papers: list[dict]) -> dict:
    client = _get_client()
    if client is None or not papers:
        return _heuristic_judge(papers)

    paper_summaries = "\n".join(
        f"- {p['title']} ({p['year']}): {p['abstract'][:200]}" for p in papers
    )
    try:
        msg = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "아래 주장 문장과 후보 논문 목록을 보고, 이 주장을 뒷받침하는 "
                        "근거로 충분한 논문이 있는지 판단하라. "
                        'JSON만 출력: {"sufficient": true/false, "reason": "한 문장 이유"}\n\n'
                        f"주장: {sentence}\n\n후보 논문:\n{paper_summaries}"
                    ),
                }
            ],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        return {
            "sufficient": bool(parsed.get("sufficient")),
            "reason": parsed.get("reason", ""),
        }
    except Exception:
        return _heuristic_judge(papers)


def _heuristic_judge(papers: list[dict]) -> dict:
    if papers:
        return {"sufficient": True, "reason": "관련 논문이 검색되었습니다 (자동 판별 미사용)."}
    return {"sufficient": False, "reason": "관련 선행연구를 찾지 못했습니다."}


def analyze_draft(text: str) -> list[dict]:
    results = []
    for sentence in split_sentences(text):
        fallback_query = build_query(sentence)
        query = _refine_query_with_claude(sentence, fallback_query)
        papers = search_papers(query)
        if not papers and query != fallback_query:
            papers = search_papers(fallback_query)
        verdict = _judge_with_claude(sentence, papers)
        results.append(
            {
                "sentence": sentence,
                "query": query,
                "papers": papers,
                "sufficient": verdict["sufficient"],
                "reason": verdict["reason"],
            }
        )
    return results


def agentic_mode_enabled() -> bool:
    return _get_client() is not None
