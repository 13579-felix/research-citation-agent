# 연구 글쓰기 근거 보완 Agent

AI+X 1차 과제 프로젝트: **연구 글쓰기용 선행연구 인용 자동 매칭 및 근거 보완 Agentic AI**

연구 배경 문단이나 논문/연구계획서 초안을 입력하면 문장 단위로 주장을 추출하고,
[Semantic Scholar](https://www.semanticscholar.org/) API에서 관련 선행연구를 검색해
근거를 매칭합니다. `ANTHROPIC_API_KEY`를 설정하면 Claude가 검색어 생성과
"근거가 충분한지" 판단까지 수행하는 Agentic 모드로 동작하고, 키가 없으면
영문 전문용어 추출 기반 휴리스틱으로 동작합니다 (예: FeFET, HZO, TiN capping 같은
재료·소자 연구의 영문 전문용어를 문장에서 뽑아 검색어로 사용).

## 배경

IGZO 채널 FeFET(HZO 강유전체 게이트) 연구에서 TiN capping에 의한 상전이,
SUT(Simultaneous UV-Thermal) 처리 등 선행연구를 지속적으로 조사해야 하는데,
글을 쓸 때마다 주장에 맞는 인용을 수작업으로 찾아 삽입하는 과정이 반복적이고
근거 누락 위험이 있다는 문제에서 출발했습니다.

## 실행 방법

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
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
  agent.py             문장별 검색어 생성 + 근거 충분성 판단 (Claude 또는 휴리스틱)
  claim_extractor.py   초안 텍스트를 문장 단위 주장으로 분리
  citation_search.py   Semantic Scholar API 래퍼
frontend/
  index.html, app.js, style.css   단순 정적 웹 UI
```

## 실습 기록 (2부)

과제 문서의 "실습내용 / 실습결과" 항목은 실제로 이 앱을 자신의 FeFET 연구 관련
글에 사용해본 뒤 채워 넣습니다.
