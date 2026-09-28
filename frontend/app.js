const input = document.getElementById("draft-input");
const button = document.getElementById("analyze-btn");
const results = document.getElementById("results");
const modeBadge = document.getElementById("mode-badge");

// Must match the statuses produced by backend/agent.py.
const STATUS = {
  supported: { text: "근거 확인", cls: "ok" },
  partial: { text: "일부 확인", cls: "caution" },
  contradicted: { text: "근거와 모순", cls: "warn" },
  insufficient: { text: "근거 부족", cls: "warn" },
  unverifiable: { text: "확인 불가 (초록 없음)", cls: "neutral" },
  not_needed: { text: "인용 불필요", cls: "neutral" },
  undetermined: { text: "미판정", cls: "caution" },
  error: { text: "판단 실패", cls: "caution" },
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function safeUrl(url) {
  return /^https?:\/\//i.test(url || "") ? url : "#";
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    modeBadge.textContent = data.agentic_mode
      ? "Agentic 모드 (Gemini가 인용 필요 여부·근거·모순 판단)"
      : "휴리스틱 모드 (GEMINI_API_KEY 미설정 — 근거 판단 안 함)";
  } catch {
    modeBadge.textContent = "상태 확인 실패";
  }
}

function matchedHTML(p) {
  if (!p.matched_sentence) {
    return `<div class="matched none">초록이 없어 매칭 문장을 찾을 수 없습니다.</div>`;
  }
  const source = p.match_source === "gemini" ? "Gemini 선택 · 원문 확인됨" : "키워드 일치 (자동)";
  return `
    <blockquote class="matched">
      “${esc(p.matched_sentence)}”
      <span class="matched-source">초록 원문 · ${source}</span>
    </blockquote>`;
}

function paperHTML(p, i) {
  const authors = p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " 외" : "");
  const role = p.role === "supporting" ? "근거" : p.role === "contradicting" ? "모순" : "";
  return `
    <div class="paper ${p.role || ""}">
      <span class="paper-num">(${i + 1})</span>
      ${role ? `<span class="paper-role">${role}</span>` : ""}
      <a href="${esc(safeUrl(p.url))}" target="_blank" rel="noopener">${esc(p.title)}</a>
      <div class="paper-meta">${esc(authors)} · ${esc(p.year ?? "연도 미상")} ${p.venue ? "· " + esc(p.venue) : ""} · ${esc(p.provider)}</div>
      ${matchedHTML(p)}
    </div>`;
}

function searchStatusHTML(status) {
  const entries = Object.entries(status || {});
  if (!entries.length) return "";
  const parts = entries.map(([name, s]) =>
    s.startsWith("ok")
      ? `${esc(name)} ${esc(s.slice(3))}`
      : `<span class="failed">${esc(name)} 실패</span>`
  );
  const failures = entries.filter(([, s]) => !s.startsWith("ok"));
  const title = failures.map(([, s]) => s).join("\n");
  return `<div class="search-status" title="${esc(title)}">검색: ${parts.join(" · ")}</div>`;
}

function conditionsHTML(conditions) {
  if (!conditions || !conditions.length) return "";
  const rows = conditions.map((c) => `
    <li class="cond ${esc(c.status)}">
      <span class="cond-label">${esc(c.label)}</span> ${esc(c.condition)}
      ${c.quote ? `<div class="cond-quote">“${esc(c.quote)}” — 논문 (${esc(c.paper)})</div>` : ""}
    </li>`).join("");
  return `<ul class="conditions">${rows}</ul>`;
}

function cardHTML(item) {
  const status = STATUS[item.status] || STATUS.error;
  let papers = "";
  if (item.papers.length) {
    papers = item.papers.map((p, i) => paperHTML(p, i)).join("");
  } else if (item.query) {
    papers = `<div class="paper-meta">검색된 선행연구가 없습니다. 검색어: "${esc(item.query)}"</div>`;
  }

  return `
    <div class="claim-card">
      <span class="status-pill ${status.cls}">${status.text}</span>
      <div class="claim-sentence">${esc(item.sentence)}</div>
      <div class="reason">${esc(item.reason)}</div>
      ${conditionsHTML(item.conditions)}
      ${papers}
      ${item.papers.length && item.query ? `<div class="paper-meta">검색어: ${esc(item.query)}</div>` : ""}
      ${searchStatusHTML(item.search_status)}
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
    if (!res.ok) throw new Error(`서버 오류 (HTTP ${res.status})`);
    const data = await res.json();
    if (!data.results.length) {
      results.innerHTML = `<div class="empty-state">분석할 문장을 찾지 못했습니다.</div>`;
    } else {
      const notice = data.truncated
        ? `<div class="notice">문장이 많아 앞부분 ${data.results.length}문장만 분석했습니다.</div>`
        : "";
      results.innerHTML = notice + data.results.map(cardHTML).join("");
    }
  } catch (err) {
    results.innerHTML = `<div class="empty-state">오류가 발생했습니다: ${esc(err.message || err)}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "분석하기";
  }
}

button.addEventListener("click", analyze);
loadStatus();
