const input = document.getElementById("draft-input");
const button = document.getElementById("analyze-btn");
const results = document.getElementById("results");
const modeBadge = document.getElementById("mode-badge");

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    modeBadge.textContent = data.agentic_mode
      ? "Agentic 모드 (Claude 사용)"
      : "휴리스틱 모드 (ANTHROPIC_API_KEY 미설정)";
  } catch {
    modeBadge.textContent = "상태 확인 실패";
  }
}

function paperHTML(p) {
  const authors = p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " 외" : "");
  return `
    <div class="paper">
      <a href="${p.url || "#"}" target="_blank" rel="noopener">${p.title}</a>
      <div class="paper-meta">${authors} · ${p.year ?? "연도 미상"} ${p.venue ? "· " + p.venue : ""} · ${p.provider}</div>
    </div>`;
}

function cardHTML(item) {
  const statusClass = item.sufficient ? "ok" : "warn";
  const statusText = item.sufficient ? "근거 매칭됨" : "근거 부족";
  const papers = item.papers.length
    ? item.papers.map(paperHTML).join("")
    : `<div class="paper-meta">검색된 선행연구가 없습니다. 검색어: "${item.query}"</div>`;

  return `
    <div class="claim-card">
      <span class="status-pill ${statusClass}">${statusText}</span>
      <div class="claim-sentence">${item.sentence}</div>
      <div class="reason">${item.reason}</div>
      ${papers}
    </div>`;
}

async function analyze() {
  const text = input.value.trim();
  if (!text) return;

  button.disabled = true;
  button.textContent = "분석 중...";
  results.innerHTML = `<div class="empty-state">문장별로 선행연구를 검색하고 있습니다...</div>`;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!data.results.length) {
      results.innerHTML = `<div class="empty-state">분석할 문장을 찾지 못했습니다.</div>`;
    } else {
      results.innerHTML = data.results.map(cardHTML).join("");
    }
  } catch (err) {
    results.innerHTML = `<div class="empty-state">오류가 발생했습니다: ${err}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "분석하기";
  }
}

button.addEventListener("click", analyze);
loadStatus();
