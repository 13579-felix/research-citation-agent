"""Per-sentence citation analysis.

Every sentence ends in exactly one status:

- supported     근거 확인: a candidate paper's abstract backs the claim
- contradicted  근거와 모순: a candidate paper reports something that conflicts
- insufficient  근거 부족: candidates don't back the claim (or none were found)
- not_needed    인용 불필요: the sentence describes this work, not prior findings
- undetermined  미판정: nothing judged the evidence (heuristic mode)
- error         판단 실패: the judge call failed

The app must never report "supported" unless a judge actually said so;
failures and heuristic mode map to undetermined/error, never supported.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from citation_search import search_papers
from claim_extractor import build_query, looks_self_referential, split_sentences

_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
_MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "30"))
_SENTENCE_WORKERS = 4
_anthropic_client = None
_logger = logging.getLogger("agent")


class CitationPlan(BaseModel):
    needs_citation: bool = Field(
        description="True if the sentence states a fact/result from prior literature that a reader would expect to be cited."
    )
    reason: str = Field(description="한국어 한 문장 이유")
    query: str = Field(description="English search query, 3-6 specific technical keywords. Empty if needs_citation is false.")


class Judgement(BaseModel):
    verdict: Literal["supported", "contradicted", "insufficient"]
    reason: str = Field(description="한국어 1~2문장. 어떤 논문의 어떤 내용 때문인지 구체적으로.")
    supporting: list[int] = Field(description="1-based indices of papers that support the claim")
    contradicting: list[int] = Field(description="1-based indices of papers that contradict the claim")


def _get_client():
    """Lazily build an Anthropic client if an API key is configured.

    The app runs without one (heuristic mode) so a missing key never
    blocks the demo, but in that mode nothing is ever marked "supported".
    """
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def agentic_mode_enabled() -> bool:
    return _get_client() is not None


def _parse(client, schema: type[BaseModel], prompt: str, effort: str):
    response = client.messages.parse(
        model=_ANTHROPIC_MODEL,
        max_tokens=4000,
        output_config={"effort": effort},
        output_format=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("모델이 응답을 거부했습니다")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise RuntimeError(f"구조화된 응답을 받지 못했습니다 (stop_reason={response.stop_reason})")
    return response.parsed_output


def _plan_with_claude(client, sentence: str) -> CitationPlan:
    return _parse(
        client,
        CitationPlan,
        "다음은 연구 글(논문/연구계획서) 초안의 한 문장이다.\n"
        "1) 이 문장이 선행연구 인용이 필요한 주장인지 판단하라. 기존 문헌의 사실·결과·수치를 "
        "서술하면 인용이 필요하다. 본 연구의 목적·방법·계획을 서술하거나, 글의 구성을 안내하는 "
        "문장은 인용이 필요 없다.\n"
        "2) 인용이 필요하면, OpenAlex/Crossref/arXiv에서 이 주장을 검증할 논문을 찾기 위한 "
        "영어 검색 쿼리(구체적인 재료·소자·공정 용어 3~6개)를 만들어라. 'Si', 'Zr' 같은 "
        "단일 원소 기호만으로 된 일반적인 쿼리는 피하라.\n\n"
        f"문장: {sentence}",
        effort="low",
    )


def _judge_with_claude(client, sentence: str, papers: list[dict]) -> Judgement:
    paper_block = "\n\n".join(
        f"[{i}] {p['title']} ({p['year'] or '연도 미상'}, {p['venue'] or p['provider']})\n"
        f"초록: {p['abstract'] or '(초록 없음 — 제목만으로는 근거로 인정하지 말 것)'}"
        for i, p in enumerate(papers, 1)
    )
    return _parse(
        client,
        Judgement,
        "너는 연구 글의 인용 근거를 검증하는 심사자다. 아래 주장 문장과 후보 논문(초록 전문)을 대조하라.\n\n"
        "판정 기준:\n"
        "- supported: 적어도 한 논문의 초록이 이 주장의 핵심 내용(대상 재료·소자, 조건, 방향, "
        "수치)을 직접 뒷받침한다.\n"
        "- contradicted: 한 논문이라도 주장과 상충하는 결과를 보고한다. 예: 주장은 '5 nm에서도 "
        "동일한 2Pr'인데 초록은 두께 감소에 따른 2Pr 감소를 보고함. 수치·경향·조건이 다르면 "
        "모순으로 본다. 뒷받침하는 논문이 함께 있어도 모순이 있으면 contradicted.\n"
        "- insufficient: 주제는 관련 있어도 주장의 구체적 내용(특히 수치·조건)을 확인할 수 없다. "
        "키워드만 겹치는 논문, 다른 재료계 논문, 초록이 없는 논문은 근거가 아니다.\n"
        "확신이 없으면 supported가 아니라 insufficient로 판정하라.\n\n"
        f"주장: {sentence}\n\n후보 논문:\n{paper_block}",
        effort="medium",
    )


def _result(sentence, query, papers, status, reason, search_status=None, supporting=(), contradicting=()):
    return {
        "sentence": sentence,
        "query": query,
        "papers": [
            {
                **{k: v for k, v in p.items() if k != "abstract"},
                "role": "supporting" if i in supporting else "contradicting" if i in contradicting else None,
            }
            for i, p in enumerate(papers, 1)
        ],
        "status": status,
        "reason": reason,
        "search_status": search_status or {},
    }


def _no_papers(sentence: str, query: str, search_status: dict) -> dict:
    if search_status and not any(s.startswith("ok") for s in search_status.values()):
        return _result(
            sentence, query, [], "error",
            "모든 검색 소스 요청이 실패해 근거를 확인하지 못했습니다.", search_status,
        )
    return _result(sentence, query, [], "insufficient", "관련 후보 논문을 찾지 못했습니다.", search_status)


def _analyze_heuristic(sentence: str) -> dict:
    if looks_self_referential(sentence):
        return _result(sentence, "", [], "not_needed", "본 연구 서술로 보여 검색하지 않았습니다 (휴리스틱).")
    query = build_query(sentence)
    if not query:
        return _result(
            sentence, "", [], "undetermined",
            "검색어로 쓸 구체적인 영문 전문용어가 없어 검색하지 않았습니다. "
            "ANTHROPIC_API_KEY를 설정하면 Claude가 검색어를 생성합니다.",
        )
    papers, search_status = search_papers(query)
    if not papers:
        return _no_papers(sentence, query, search_status)
    return _result(
        sentence, query, papers, "undetermined",
        "후보 논문만 검색했고 근거 여부는 판단하지 않았습니다 (휴리스틱 모드). 직접 확인이 필요합니다.",
        search_status,
    )


def _analyze_agentic(client, sentence: str) -> dict:
    try:
        plan = _plan_with_claude(client, sentence)
    except Exception as exc:
        _logger.warning("citation planning failed for %r: %s", sentence, exc)
        plan = None

    if plan is not None and not plan.needs_citation:
        return _result(sentence, "", [], "not_needed", plan.reason)

    query = (plan.query.strip() if plan else "") or build_query(sentence)
    if not query:
        return _result(sentence, "", [], "error", "검색어를 생성하지 못해 판단할 수 없습니다.")

    papers, search_status = search_papers(query)
    fallback = build_query(sentence)
    if not papers and fallback and fallback != query:
        query = fallback
        papers, search_status = search_papers(query)
    if not papers:
        return _no_papers(sentence, query, search_status)

    try:
        judgement = _judge_with_claude(client, sentence, papers)
    except Exception as exc:
        _logger.warning("judgement failed for %r: %s", sentence, exc)
        return _result(
            sentence, query, papers, "error",
            f"근거 판단에 실패했습니다 ({exc.__class__.__name__}). 후보 논문을 직접 확인하세요.",
            search_status,
        )
    return _result(
        sentence, query, papers, judgement.verdict, judgement.reason, search_status,
        supporting=set(judgement.supporting), contradicting=set(judgement.contradicting),
    )


def analyze_draft(text: str) -> tuple[list[dict], bool]:
    """Returns (results, truncated)."""
    sentences = split_sentences(text)
    truncated = len(sentences) > _MAX_SENTENCES
    sentences = sentences[:_MAX_SENTENCES]

    client = _get_client()
    if client is None:
        worker = _analyze_heuristic
    else:
        def worker(sentence):
            return _analyze_agentic(client, sentence)

    with ThreadPoolExecutor(max_workers=_SENTENCE_WORKERS) as pool:
        return list(pool.map(worker, sentences)), truncated
