# 연구 글쓰기 근거 보완 Agent

AI+X 1차 과제 프로젝트: **연구 글쓰기용 선행연구 인용 자동 매칭 및 근거 보완 Agentic AI**

연구 배경 문단이나 논문/연구계획서 초안을 입력하면 문장 단위로 주장을 추출하고,
[OpenAlex](https://openalex.org/), [Crossref](https://www.crossref.org/), [arXiv](https://arxiv.org/)
API에서 관련 선행연구를 검색해 근거를 매칭합니다 (모두 키 없이 사용 가능,
`SEMANTIC_SCHOLAR_API_KEY`를 설정하면 Semantic Scholar도 추가).

- **Agentic 모드** (`ANTHROPIC_API_KEY` 설정): Claude가 ① 인용이 필요한 문장인지 판별하고,
  ② 영어 검색어를 만들고, ③ 후보 논문 초록 전문과 대조해 **근거 확인 / 근거와 모순 / 근거 부족**을
  판정합니다. 수치·조건이 다르면 모순으로 봅니다.
- **휴리스틱 모드** (키 없음): 문장에서 구체적인 영문 전문용어(FeFET, HZO, TiN 등; "Si"·"nm" 같은
  일반 조각은 제외)만 뽑아 후보 논문을 검색하고, 근거 여부는 판단하지 않아 **미판정**으로 표시합니다.

판정 결과는 `근거 확인`, `근거와 모순`, `근거 부족`, `인용 불필요`, `미판정`, `판단 실패` 중 하나입니다.
판단을 하지 않았거나 판단에 실패한 문장은 절대 "근거 확인"으로 표시하지 않습니다. 카드 하단에는
검색 소스별 성공/실패가 표시되어, 특정 소스(예: OpenAlex)가 조용히 실패하는 경우를 바로 알 수 있습니다.

## 배경

IGZO 채널 FeFET(HZO 강유전체 게이트) 연구에서 TiN capping에 의한 상전이,
SUT(Simultaneous UV-Thermal) 처리 등 선행연구를 지속적으로 조사해야 하는데,
글을 쓸 때마다 주장에 맞는 인용을 수작업으로 찾아 삽입하는 과정이 반복적이고
근거 누락 위험이 있다는 문제에서 출발했습니다.

## 실행 방법

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Agentic 모드를 쓰려면 프로젝트 루트에 `.env` 파일을 만들고 (`.env.example` 참고)
본인의 `ANTHROPIC_API_KEY`를 넣습니다. 없어도 휴리스틱 모드로 바로 동작합니다.

```bash
uvicorn app:app --reload --port 8000
```

브라우저에서 `http://localhost:8000` 접속.

## 구조

```
backend/
  app.py               FastAPI 엔트리포인트, /api/analyze
  agent.py             인용 필요 판별 → 검색 → 근거/모순 판정 (Claude 또는 휴리스틱)
  claim_extractor.py   초안 텍스트를 문장 단위 주장으로 분리
  citation_search.py   OpenAlex + Crossref + arXiv (+ Semantic Scholar) 병렬 검색
frontend/
  index.html, app.js, style.css   단순 정적 웹 UI
```

## 실습 기록 (2부)

과제 문서의 "실습내용 / 실습결과" 항목은 실제로 이 앱을 자신의 FeFET 연구 관련
글에 사용해본 뒤 채워 넣습니다.
