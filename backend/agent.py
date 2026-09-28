"""Per-sentence citation analysis.

Every sentence ends in exactly one status:

- supported     근거 확인: every condition of the claim is confirmed by a verified quote
- partial       일부 확인: one paper confirms some conditions, none contradicted
- unverifiable  확인 불가: no candidate paper has an abstract to check against
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

from citation_search import fill_missing_abstracts, search_many, search_papers
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
    queries: list[str] = Field(
        description="1-3 English search queries (3-6 keywords each). Include one variant with chemical names spelled out "
        "(e.g. 'hafnium oxide' for HfO2). Empty if needs_citation is false."
    )
    year: int = Field(description="Publication/report year stated in the sentence (e.g. 2011), else 0")
    known_title: str = Field(
        description="Exact title of the well-known original paper for this claim, only if you are certain; else empty"
    )


class CitationPlans(BaseModel):
    plans: list[CitationPlan]


class Evidence(BaseModel):
    paper: int = Field(description="Paper number within this claim")
    quote: str = Field(description="Verbatim span copied from that paper's abstract or title (or just the publication year, e.g. 2011)")
    stance: Literal["supports", "contradicts"]
    special_condition: str = Field(
        description="한국어. 이 결과가 특수 처리·조건에서만 얻은 값이면 그 조건(예: '칼슘 캡핑 후'), 일반적인 값이면 빈 문자열"
    )


class ConditionCheck(BaseModel):
    condition: str = Field(description="한국어. 주장을 이루는 조건 하나")
    kind: Literal["subject", "fact", "qualifier"] = Field(
        description="subject: 주제·대상 이름만(예: 'a-IGZO'); fact: 구체적 사실·수치·공정 조건; "
        "qualifier: 일반성·최초·비교 표현(typically, generally, first, higher than LTPS 등)"
    )
    evidence: list[Evidence] = Field(description="All supporting AND contradicting evidence from every paper; empty if none")


class PaperMatch(BaseModel):
    paper: int = Field(description="Paper number within this claim")
    sentence: str = Field(description="The one abstract sentence most relevant to the claim, copied verbatim and whole. Empty if no abstract.")


class Judgement(BaseModel):
    index: int = Field(description="The claim number given in the input")
    conditions: list[ConditionCheck]
    overall: Literal["supported", "contradicted", "insufficient"] = Field(
        description="문장 전체(한정어·비교 포함)가 초록들로 성립하는지에 대한 종합 판정"
    )
    overall_reason: str = Field(description="한국어 1~2문장. 종합 판정 이유")
    matches: list[PaperMatch] = Field(default_factory=list)


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
        "영어 검색 쿼리를 1~3개 만들어라(각 3~6개 구체적 용어). 논문 제목은 화학식 대신 이름을 쓰는 "
        "경우가 많으니 하나는 이름으로 풀어 써라(HfO2 → hafnium oxide, HZO → hafnium zirconium oxide). "
        "'Si', 'Zr' 같은 단일 원소 기호만으로 된 쿼리는 피하라.\n"
        "3) 문장에 연도가 있으면 year에, 이 주장의 원 출처로 널리 알려진 논문 제목을 확실히 알면 "
        "known_title에 정확히 적어라. 확실하지 않으면 비워라.\n\n"
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
        "1) 주장을 조건으로 나누고 종류를 붙여라. subject: 주제·대상 이름만(예: 'a-IGZO'). fact: 구체적 "
        "사실·수치·공정 조건(두께, 온도, 이동도 값, 연도 등). qualifier: 일반성·최초·비교 표현"
        "('typically', 'generally', 'first', 'higher than LTPS', '~와 같은 수준'). 한정어와 비교는 반드시 "
        "별도 qualifier 조건으로 둔다.\n"
        "2) 각 조건마다 모든 후보 논문에서 지지 근거와 반대 근거를 **둘 다** 찾아 evidence에 넣어라. "
        "반대 근거를 빠뜨리지 마라:\n"
        "   - 같은 재료·물성에 대해 주장 범위 밖의 값(예: 주장 '>100 cm²/Vs'인데 초록 '18.1 cm²/Vs')은 contradicts.\n"
        "   - 같은 문장 안의 기준값·처리 전 값·비교군 값도 따로 평가하라(예: '12에서 160으로 증가'라면 "
        "'12'는 일반 a-IGZO 값으로 contradicts, '160'은 special_condition='칼슘 캡핑 후'인 supports).\n"
        "   - 특수 처리·특수 조건에서만 얻은 결과는 special_condition에 그 조건을 적어라.\n"
        "3) quote는 해당 초록(또는 제목)에서 그대로 복사하라. 바꾸거나 요약·번역하지 마라. 연도 근거는 "
        "그 논문의 출판 연도 숫자만(예: 2011). quote는 주장과 같은 재료·소자에 대한 내용이어야 한다.\n"
        "4) qualifier는 초록이 그 일반성·비교를 직접 말할 때만 supports다(예: 'typically', 'higher than', "
        "'first'). 예외적·특수 조건의 결과 하나로 '일반적으로'나 '최초'를 지지할 수 없다.\n"
        "5) overall: 한정어·비교를 포함한 문장 전체가 초록들로 성립하면 supported, 반대 근거가 있으면 "
        "contradicted, 판단할 근거가 부족하면 insufficient. 확신이 없으면 supported로 하지 마라.\n"
        "6) matches: 후보 논문마다, 그 초록에서 주장과 가장 관련 깊은 문장 하나를 한 문장 전체 그대로 "
        "복사하라(바꾸거나 줄이지 말 것). 초록이 없으면 빈 문자열.\n\n"
        + "\n\n".join(blocks),
    )
    return {j.index: j for j in result.judgements}


_NON_WORD = re.compile(r"[\W_]+")


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


_ABSTRACT_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_MATCH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have",
    "been", "its", "their", "which", "than", "can", "also", "such", "into", "our", "these",
    "study", "paper", "work", "results", "show", "shows", "using", "based",
    "of", "as", "to", "in", "on", "by", "at", "is", "be", "an", "or", "it", "we",
}


def _terms(text: str) -> set[str]:
    return {t for t in _normalize(text).split() if len(t) >= 2 and t not in _MATCH_STOPWORDS}


def _keyword_best_sentence(abstract: str, claim: str, query: str) -> str:
    """Abstract sentence sharing the most terms with the claim/query; "" if none overlap."""
    wanted = _terms(claim) | _terms(query)
    best, best_score = "", 0
    for sent in _ABSTRACT_SENTENCE.split(abstract.strip()):
        score = len(_terms(sent) & wanted)
        if score > best_score:
            best, best_score = sent.strip(), score
    return best


def _matched_sentence(paper: dict, claim: str, query: str, llm_sentence: str = "") -> tuple[str, str]:
    """(sentence, source): the Gemini-picked sentence if it really is in the
    abstract, else a keyword-overlap pick, else nothing."""
    abstract = paper.get("abstract") or ""
    if not abstract:
        return "", ""
    if llm_sentence and len(_normalize(llm_sentence)) >= 8 and _normalize(llm_sentence) in _normalize(abstract):
        return llm_sentence.strip(), "gemini"
    sentence = _keyword_best_sentence(abstract, claim, query)
    return (sentence, "keyword") if sentence else ("", "")


_STATUS_LABEL = {
    "confirmed": "확인",
    "contradicted": "모순",
    "not_mentioned": "확인 안 됨",
    "other_paper": "다른 논문에서만 확인",
    "subject": "주제어 (판정 제외)",
}


# A qualifier ("typically", "first", "higher than LTPS") is only supported by a
# quote that itself states generality, priority or a comparison.
_QUALIFIER_MARKERS = re.compile(
    r"\b(typical\w*|general\w*|usual\w*|common\w*|widely|most|first|pioneer\w*|"
    r"higher|lower|greater|larger|smaller|superior|inferior|better|worse|exceed\w*|outperform\w*|"
    r"than|compared|comparable|similar|same|equal\w*|over|above|below)\b"
)


def _verify_quote(ev: Evidence, papers: list[dict]) -> tuple[bool, bool]:
    """(quote really occurs in that paper's abstract/title or is its year, is_year)."""
    paper = papers[ev.paper - 1] if 1 <= ev.paper <= len(papers) else {}
    quote = _normalize(ev.quote)
    is_year = quote.isdigit() and len(quote) == 4 and quote == str(paper.get("year"))
    in_text = len(quote) >= 8 and (
        quote in _normalize(paper.get("abstract") or "") or quote in _normalize(paper.get("title") or "")
    )
    return is_year or in_text, is_year


# Generality/priority words in the claim itself. If one is present, the
# claim cannot be (partially) supported unless a qualifier was confirmed.
_CLAIM_QUALIFIERS = re.compile(
    r"\b(typically|generally|usually|commonly|always|widely|first|in general)\b|일반적|대부분|항상|최초|처음",
    re.IGNORECASE,
)


def _verdict_from(judgement: Judgement, papers: list[dict], sentence: str = ""):
    """Decide the verdict in code from quote-verified evidence.

    Rules (each exists because a real run got it wrong):
    - Every quote must really occur in that paper's abstract/title (or be its
      publication year), otherwise the evidence is dropped.
    - Contradicting evidence is looked for, not just support; one verified
      contradiction makes the claim "contradicted".
    - "subject" conditions (just naming the material) never count.
    - A qualifier (typically/first/higher than) is only supported by
      evidence with no special condition whose quote states generality or a
      comparison; a Ca-capped 160 cm²/Vs cannot support "typically >100".
    - Support must come from one paper, and a year match alone never counts.
    - The model's whole-sentence verdict can only make the result more
      cautious, never upgrade it.
    """
    checks = []
    for c in judgement.conditions:
        supports, contradicts, notes = [], [], []
        for ev in c.evidence:
            ok, is_year = _verify_quote(ev, papers)
            if not ok:
                continue
            if ev.stance == "contradicts":
                contradicts.append(ev)
                continue
            if c.kind == "qualifier" and ev.special_condition.strip():
                notes.append(f"논문 ({ev.paper})의 값은 '{ev.special_condition.strip()}' 조건의 결과라 일반화 근거가 아님")
                continue
            if c.kind == "qualifier" and not _QUALIFIER_MARKERS.search(ev.quote.lower()):
                notes.append(f"논문 ({ev.paper}) 인용문에 일반성·비교 표현이 없음")
                continue
            supports.append((ev, is_year))
        checks.append({"c": c, "supports": supports, "contradicts": contradicts, "notes": notes})

    # The single paper confirming the most non-subject conditions by text.
    text_hits: dict[int, int] = {}
    for ch in checks:
        if ch["c"].kind == "subject":
            continue
        for p in {ev.paper for ev, is_year in ch["supports"] if not is_year}:
            text_hits[p] = text_hits.get(p, 0) + 1
    best = max(text_hits, key=text_hits.get) if text_hits else None

    rendered, supporting, contradicting = [], set(), set()
    for ch in checks:
        c = ch["c"]
        best_support = [ev for ev, _ in ch["supports"] if ev.paper == best]
        if ch["contradicts"] and c.kind != "subject":
            status, shown = "contradicted", ch["contradicts"]
            contradicting |= {ev.paper for ev in shown}
        elif best_support:
            status, shown = "confirmed", best_support
            supporting |= {ev.paper for ev in shown}
        elif ch["supports"]:
            status, shown = "other_paper", [ev for ev, _ in ch["supports"]]
        else:
            status, shown = "not_mentioned", []
        if c.kind == "subject" and status != "confirmed":
            status = "subject"
        rendered.append({
            "condition": c.condition,
            "kind": c.kind,
            "status": status,
            "label": _STATUS_LABEL[status],
            "evidence": [
                {"paper": ev.paper, "quote": ev.quote, "stance": ev.stance, "special_condition": ev.special_condition}
                for ev in shown
            ],
            "notes": ch["notes"],
        })

    counted = [r for r in rendered if r["kind"] != "subject"]

    def names(*statuses):
        return ", ".join(r["condition"] for r in counted if r["status"] in statuses)

    if any(r["status"] == "contradicted" for r in counted):
        verdict = "contradicted"
        reason = f"초록에 주장과 반대되는 근거가 있습니다: {names('contradicted')}."
    elif counted and best is not None and all(r["status"] == "confirmed" for r in counted):
        verdict = "supported"
        reason = f"주장의 모든 조건(한정어 포함)이 논문 ({best})에서 원문 인용으로 확인되었습니다."
    elif best is not None:
        verdict = "partial"
        reason = (
            f"논문 ({best})에서 일부 조건만 확인되었습니다. 확인됨: {names('confirmed')}. "
            f"확인 안 됨: {names('not_mentioned', 'other_paper')}."
        )
    elif not any(p.get("abstract") for p in papers):
        verdict = "unverifiable"
        reason = "후보 논문에 초록이 없어 내용을 확인할 수 없습니다. 논문을 직접 열어 확인하세요."
    else:
        verdict = "insufficient"
        reason = f"초록에서 확인되지 않은 조건이 있습니다: {names('not_mentioned', 'other_paper') or '주장의 구체적 조건'}."

    qualifier_confirmed = any(r["kind"] == "qualifier" and r["status"] == "confirmed" for r in rendered)
    if verdict in ("supported", "partial") and _CLAIM_QUALIFIERS.search(sentence) and not qualifier_confirmed:
        verdict = "insufficient"
        reason = (
            f"문장의 한정어('{_CLAIM_QUALIFIERS.search(sentence).group(0)}')를 뒷받침하는 일반적 근거가 초록에 없습니다. "
            f"개별 사례의 확인({names('confirmed') or '없음'})만으로는 이 표현을 지지할 수 없습니다."
        )

    # The whole-sentence judgement can only make things more cautious.
    overall = judgement.overall
    if verdict in ("supported", "partial") and overall == "contradicted":
        verdict = "insufficient"
        reason += f" 문장 전체로는 반대 근거가 있다고 판단되었으나 원문 인용으로 검증되지 않았습니다: {judgement.overall_reason}"
    elif verdict == "supported" and overall == "insufficient":
        verdict = "partial"
        reason += f" 다만 문장 전체로는 근거가 충분하지 않다고 판단되었습니다: {judgement.overall_reason}"
    elif verdict == "partial":
        reason += " 확인 안 된 부분의 근거를 보완하거나 표현을 조정하세요."
    if verdict == "insufficient" and overall != "contradicted":
        reason += " 이 부분의 근거를 보완하거나 표현을 수정하세요."
    return verdict, reason, rendered, supporting, contradicting


def _result(sentence, query, papers, status, reason, search_status=None, supporting=(), contradicting=(),
            conditions=(), llm_matches=None):
    llm_matches = llm_matches or {}
    rendered = []
    for i, p in enumerate(papers, 1):
        matched, source = _matched_sentence(p, sentence, query, llm_matches.get(i, ""))
        rendered.append({
            **{k: v for k, v in p.items() if k != "abstract"},
            "role": "contradicting" if i in contradicting else "supporting" if i in supporting else None,
            "matched_sentence": matched,
            "match_source": source,
        })
    return {
        "sentence": sentence,
        "query": query,
        "papers": rendered,
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
    """Returns (query label, papers, search_status)."""
    fallback = build_query(sentence)
    queries = [q for q in (plan.queries if plan else []) if q.strip()] or ([fallback] if fallback else [])
    if not queries:
        return "", [], {}
    year = plan.year if plan and 1900 < plan.year < 2100 else None
    known_title = plan.known_title if plan else ""
    papers, search_status = search_many(queries, year=year, known_title=known_title)
    if not papers and fallback and fallback not in queries:
        queries = [fallback]
        papers, search_status = search_many(queries)
    label = " | ".join(queries) + (f" | 제목: {known_title}" if known_title else "") + (f" | {year}년" if year else "")
    return label, papers, search_status


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

    fill_missing_abstracts([p for _, _, _, papers, _ in searched for p in papers])

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
                verdict, reason, checks, supporting, contradicting = _verdict_from(j, papers, sentence)
                results[i] = _result(
                    sentence, query, papers, verdict, reason, search_status,
                    supporting=supporting, contradicting=contradicting, conditions=checks,
                    llm_matches={m.paper: m.sentence for m in j.matches},
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
