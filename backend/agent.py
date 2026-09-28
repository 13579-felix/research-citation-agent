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
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from citation_search import search_papers
from claim_extractor import build_query, looks_self_referential, split_sentences

# "gemini-flash-latest" tracks Google's current Flash model (free tier
# available), so the app keeps working when older model versions retire.
# Override with GEMINI_MODEL to pin a specific version.
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# Tried when the main model is overloaded (503) or out of free quota (429).
# Free-tier quota is counted per model, so this also buys extra requests.
# Set GEMINI_FALLBACK_MODEL to an empty string to disable.
_GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
_MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "30"))
_SEARCH_WORKERS = 4
# The Gemini free tier allows ~5 requests/minute, so sentences are sent to
# the model in batches (1 planning call + 1 judging call per batch) instead
# of 2 calls per sentence.
_JUDGE_BATCH = 10
_gemini_client = None
_logger = logging.getLogger("agent")


class CitationPlan(BaseModel):
    index: int = Field(description="The sentence number given in the input")
    needs_citation: bool = Field(
        description="True if the sentence states a fact/result from prior literature that a reader would expect to be cited."
    )
    reason: str = Field(description="한국어 한 문장 이유")
    query: str = Field(description="English search query, 3-6 specific technical keywords. Empty if needs_citation is false.")


class CitationPlans(BaseModel):
    plans: list[CitationPlan]


class ConditionCheck(BaseModel):
    condition: str = Field(description="한국어. 주장을 이루는 구체적 조건 하나 (예: 'HZO 두께 5 nm', '2Pr이 10 nm 시료와 같은 수준')")
    status: Literal["confirmed", "contradicted", "not_mentioned"]
    paper: int = Field(description="Paper number within this claim whose abstract confirms/contradicts it; 0 if not_mentioned")
    quote: str = Field(description="Verbatim span copied from that paper's abstract (English, unchanged). Empty if not_mentioned.")


class Judgement(BaseModel):
    index: int = Field(description="The claim number given in the input")
    conditions: list[ConditionCheck]


class Judgements(BaseModel):
    judgements: list[Judgement]


def _get_client():
    """Lazily build a Gemini client if an API key is configured.

    The app runs without one (heuristic mode) so a missing key never
    blocks the demo, but in that mode nothing is ever marked "supported".
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    _gemini_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60_000,  # ms
            # The free tier has a low requests-per-minute cap; back off on 429.
            retry_options=types.HttpRetryOptions(
                attempts=2, initial_delay=5.0, max_delay=30.0, http_status_codes=[429, 500, 503]
            ),
        ),
    )
    return _gemini_client


def agentic_mode_enabled() -> bool:
    return _get_client() is not None


def _generate(client, model: str, schema: type[BaseModel], prompt: str):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    if not isinstance(response.parsed, schema):
        finish = response.candidates[0].finish_reason if response.candidates else response.prompt_feedback
        raise RuntimeError(f"구조화된 응답을 받지 못했습니다 ({finish})")
    return response.parsed


def _parse(client, schema: type[BaseModel], prompt: str):
    try:
        return _generate(client, _GEMINI_MODEL, schema, prompt)
    except errors.APIError as exc:
        if exc.code not in (429, 503) or not _GEMINI_FALLBACK_MODEL:
            raise
        _logger.warning("%s unavailable (%s), falling back to %s", _GEMINI_MODEL, exc.code, _GEMINI_FALLBACK_MODEL)
        return _generate(client, _GEMINI_FALLBACK_MODEL, schema, prompt)


def _plan_with_llm(client, sentences: list[str]) -> dict[int, CitationPlan]:
    numbered = "\n".join(f"[{i}] {sent}" for i, sent in enumerate(sentences, 1))
    result = _parse(
        client,
        CitationPlans,
        "다음은 연구 글(논문/연구계획서) 초안의 문장 목록이다. 각 문장마다 번호(index)를 그대로 써서 답하라.\n"
        "1) 이 문장이 선행연구 인용이 필요한 주장인지 판단하라. 기존 문헌의 사실·결과·수치를 "
        "서술하면 인용이 필요하다. 본 연구의 목적·방법·계획을 서술하거나, 글의 구성을 안내하는 "
        "문장은 인용이 필요 없다.\n"
        "2) 인용이 필요하면, OpenAlex/Crossref/arXiv에서 이 주장을 검증할 논문을 찾기 위한 "
        "영어 검색 쿼리(구체적인 재료·소자·공정 용어 3~6개)를 만들어라. 'Si', 'Zr' 같은 "
        "단일 원소 기호만으로 된 일반적인 쿼리는 피하라.\n\n"
        f"문장 목록:\n{numbered}",
    )
    return {plan.index: plan for plan in result.plans}


def _judge_with_llm(client, claims: list[tuple[str, list[dict]]]) -> dict[int, Judgement]:
    blocks = []
    for i, (sentence, papers) in enumerate(claims, 1):
        paper_block = "\n".join(
            f"  ({j}) {p['title']} ({p['year'] or '연도 미상'}, {p['venue'] or p['provider']})\n"
            f"      초록: {p['abstract'] or '(초록 없음 — 제목만으로는 근거로 인정하지 말 것)'}"
            for j, p in enumerate(papers, 1)
        )
        blocks.append(f"### 주장 [{i}]: {sentence}\n후보 논문:\n{paper_block}")
    result = _parse(
        client,
        Judgements,
        "너는 연구 글의 인용 근거를 검증하는 엄격한 심사자다. 각 주장 문장을 그 주장에 딸린 후보 "
        "논문의 초록과 대조하라. 주장마다 번호(index)를 그대로 쓰고, 논문 번호는 그 주장 안에서의 "
        "번호를 쓴다.\n\n"
        "절차:\n"
        "1) 주장을 검증 가능한 구체적 조건으로 모두 나눠라: 대상 재료·소자, 두께 등 치수, 수치, "
        "비교 대상('~와 같은 수준', '~보다 높다'), 공정 조건(온도·방법), 연도·최초 여부 등. "
        "비교 표현은 그 비교 자체를 하나의 조건으로 둔다.\n"
        "2) 각 조건마다 초록에 그 조건이 명시돼 있으면 confirmed, 초록이 반대 결과(다른 수치, 반대 "
        "경향, 다른 조건에서만 성립)를 보고하면 contradicted, 명시돼 있지 않으면 not_mentioned.\n"
        "3) confirmed/contradicted에는 해당 초록에서 그대로 복사한 문구(quote)를 반드시 넣어라. "
        "문구를 바꾸거나 요약하거나 번역하지 마라. 복사할 문구가 없으면 not_mentioned다.\n"
        "추론·배경지식·키워드 겹침으로 confirmed 처리하지 마라. 초록에 직접 쓰여 있는 것만 인정한다.\n\n"
        + "\n\n".join(blocks),
    )
    return {j.index: j for j in result.judgements}


_NON_WORD = re.compile(r"[\W_]+")


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


_STATUS_LABEL = {"confirmed": "확인", "contradicted": "모순", "not_mentioned": "확인 안 됨"}


def _verdict_from(judgement: Judgement, papers: list[dict]):
    """Decide the verdict in code from quote-verified conditions.

    The model's word alone never makes a claim "supported": each
    confirmed/contradicted condition must carry a quote that actually
    occurs in the cited paper's abstract, otherwise it is downgraded to
    "not mentioned". Supported needs every condition confirmed.
    """
    checks, supporting, contradicting = [], set(), set()
    for c in judgement.conditions:
        status = c.status
        if status != "not_mentioned":
            abstract = papers[c.paper - 1]["abstract"] if 1 <= c.paper <= len(papers) else ""
            quote = _normalize(c.quote)
            if len(quote) < 8 or quote not in _normalize(abstract):
                status = "not_mentioned"
        if status == "confirmed":
            supporting.add(c.paper)
        elif status == "contradicted":
            contradicting.add(c.paper)
        checks.append({
            "condition": c.condition,
            "status": status,
            "label": _STATUS_LABEL[status],
            "paper": c.paper if status != "not_mentioned" else None,
            "quote": c.quote if status != "not_mentioned" else "",
        })

    def names(status):
        return ", ".join(ch["condition"] for ch in checks if ch["status"] == status)

    if contradicting:
        verdict = "contradicted"
        reason = f"초록과 상충하는 조건이 있습니다: {names('contradicted')}."
    elif checks and all(ch["status"] == "confirmed" for ch in checks):
        verdict = "supported"
        reason = "주장의 모든 조건이 후보 논문 초록에서 원문 인용으로 확인되었습니다."
    else:
        verdict = "insufficient"
        missing = names("not_mentioned") or "주장의 구체적 조건"
        reason = f"초록에서 확인되지 않은 조건이 있습니다: {missing}. 이 부분의 근거를 보완하거나 표현을 수정하세요."
    return verdict, reason, checks, supporting, contradicting


def _result(sentence, query, papers, status, reason, search_status=None, supporting=(), contradicting=(), conditions=()):
    return {
        "sentence": sentence,
        "query": query,
        "papers": [
            {
                **{k: v for k, v in p.items() if k != "abstract"},
                "role": "contradicting" if i in contradicting else "supporting" if i in supporting else None,
            }
            for i, p in enumerate(papers, 1)
        ],
        "status": status,
        "reason": reason,
        "search_status": search_status or {},
        "conditions": list(conditions),
    }


def _no_papers(sentence: str, query: str, search_status: dict) -> dict:
    if search_status and not any(s.startswith("ok") for s in search_status.values()):
        return _result(
            sentence, query, [], "error",
            "모든 검색 소스 요청이 실패해 근거를 확인하지 못했습니다.", search_status,
        )
    return _result(sentence, query, [], "insufficient", "관련 후보 논문을 찾지 못했습니다.", search_status)


def _llm_error(exc: Exception) -> str:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text[:5]:
        return "Gemini 무료 요청 한도를 초과했습니다. 1분쯤 뒤 다시 시도하세요"
    if "UNAVAILABLE" in text or "503" in text[:5]:
        return "Gemini 서버가 혼잡합니다. 잠시 뒤 다시 시도하세요"
    return text[:150]


def _analyze_heuristic(sentence: str) -> dict:
    if looks_self_referential(sentence):
        return _result(sentence, "", [], "not_needed", "본 연구 서술로 보여 검색하지 않았습니다 (휴리스틱).")
    query = build_query(sentence)
    if not query:
        return _result(
            sentence, "", [], "undetermined",
            "검색어로 쓸 구체적인 영문 전문용어가 없어 검색하지 않았습니다. "
            "GEMINI_API_KEY를 설정하면 Gemini가 검색어를 생성합니다.",
        )
    papers, search_status = search_papers(query)
    if not papers:
        return _no_papers(sentence, query, search_status)
    return _result(
        sentence, query, papers, "undetermined",
        "후보 논문만 검색했고 근거 여부는 판단하지 않았습니다 (휴리스틱 모드). 직접 확인이 필요합니다.",
        search_status,
    )


def _search_for(sentence: str, plan: CitationPlan | None):
    """Returns (query, papers, search_status)."""
    fallback = build_query(sentence)
    query = (plan.query.strip() if plan else "") or fallback
    if not query:
        return "", [], {}
    papers, search_status = search_papers(query)
    if not papers and fallback and fallback != query:
        query = fallback
        papers, search_status = search_papers(query)
    return query, papers, search_status


def _analyze_agentic(client, sentences: list[str]) -> list[dict]:
    try:
        plans = _plan_with_llm(client, sentences)
    except Exception as exc:
        # Keep going with heuristic queries; the judge step still decides.
        _logger.warning("citation planning failed: %s", exc)
        plans = {}

    results: list[dict | None] = [None] * len(sentences)
    to_search = []
    for i, sentence in enumerate(sentences):
        plan = plans.get(i + 1)
        if plan is not None and not plan.needs_citation:
            results[i] = _result(sentence, "", [], "not_needed", plan.reason)
        else:
            to_search.append((i, sentence, plan))

    with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as pool:
        searched = list(pool.map(lambda item: (item[0], item[1], *_search_for(item[1], item[2])), to_search))

    to_judge = []
    for i, sentence, query, papers, search_status in searched:
        if not query:
            results[i] = _result(sentence, "", [], "error", "검색어를 생성하지 못해 판단할 수 없습니다.")
        elif not papers:
            results[i] = _no_papers(sentence, query, search_status)
        else:
            to_judge.append((i, sentence, query, papers, search_status))

    for start in range(0, len(to_judge), _JUDGE_BATCH):
        batch = to_judge[start:start + _JUDGE_BATCH]
        try:
            judgements = _judge_with_llm(client, [(sentence, papers) for _, sentence, _, papers, _ in batch])
            failure = None
        except Exception as exc:
            _logger.warning("judgement failed: %s", exc)
            judgements, failure = {}, _llm_error(exc)
        for n, (i, sentence, query, papers, search_status) in enumerate(batch, 1):
            j = judgements.get(n)
            if j is None:
                reason = f"근거 판단에 실패했습니다 ({failure or '모델 응답에 이 문장이 빠짐'}). 후보 논문을 직접 확인하세요."
                results[i] = _result(sentence, query, papers, "error", reason, search_status)
            else:
                verdict, reason, checks, supporting, contradicting = _verdict_from(j, papers)
                results[i] = _result(
                    sentence, query, papers, verdict, reason, search_status,
                    supporting=supporting, contradicting=contradicting, conditions=checks,
                )
    return results


def analyze_draft(text: str) -> tuple[list[dict], bool]:
    """Returns (results, truncated)."""
    sentences = split_sentences(text)
    truncated = len(sentences) > _MAX_SENTENCES
    sentences = sentences[:_MAX_SENTENCES]
    if not sentences:
        return [], truncated

    client = _get_client()
    if client is None:
        with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as pool:
            return list(pool.map(_analyze_heuristic, sentences)), truncated
    return _analyze_agentic(client, sentences), truncated
