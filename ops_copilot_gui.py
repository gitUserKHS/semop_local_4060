from __future__ import annotations

import argparse
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semop import BASELINE_SPECS, BaselineRunner, CopilotRequest, DomainCopilot, ReviewQueueStore


EXAMPLES_PATH = Path("examples/ops_sop_cases_ko.jsonl")
DOMAIN_LABELS = {
    "general": "일반 테스트",
    "warehouse_onboarding": "창고 온보딩",
    "warehouse_exception": "창고 예외 대응",
}
STATUS_LABELS = {
    "pending": "검토 대기",
    "approved": "승인",
    "rejected": "반려",
    "needs_followup": "추가 확인",
}
KPI_LABELS = {
    "invalid_advice_rate": "위험 답변 비율",
    "plan_executability": "실행 가능성",
    "missing_prerequisite_rate": "사전조건 누락 비율",
    "context_misread_rate": "문맥 오독 비율",
    "relation_recovery": "관계 복원력",
    "human_audit_usefulness": "감사 추적 유용성",
    "clarification_need_rate": "추가 질문 필요도",
}
LOW_IS_GOOD = {
    "invalid_advice_rate",
    "missing_prerequisite_rate",
    "context_misread_rate",
    "clarification_need_rate",
}
SCENARIO_HINTS = {
    "qa": "문서 질문에 답할 때 사용합니다.",
    "exception_response": "예외 상황에서 중단, 우회, 승인 절차를 확인할 때 적합합니다.",
    "onboarding": "신입 교육이나 작업 순서 설명에 적합합니다.",
}


def load_examples() -> list[dict]:
    if not EXAMPLES_PATH.exists():
        return []
    with open(EXAMPLES_PATH, "r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class OpsGuiApp:
    def __init__(
        self,
        mode: str,
        model_id: str,
        memory_store: str | None,
        memory_source: str | None,
        review_queue: str | None,
        feedback_rules: str | None,
        baseline_config: str | None,
    ):
        self.copilot = DomainCopilot(
            mode=mode,
            model_id=model_id,
            memory_store_path=memory_store,
            memory_source=memory_source,
            review_queue_path=review_queue,
            feedback_rules_path=feedback_rules,
        )
        self.review_store = ReviewQueueStore(review_queue) if review_queue else None
        self.examples = load_examples()
        self.feedback_rules = feedback_rules
        self.baseline_config = baseline_config

    def handle(self, form: dict[str, list[str]]) -> str:
        action = _first(form, "action", "run")
        flash_message = ""
        flash_tone = "neutral"
        if self.review_store is not None and action == "review_update":
            item_id = int(_first(form, "review_id", "0") or 0)
            status = _first(form, "review_status", "pending")
            note = _first(form, "resolution_note", "")
            if item_id > 0:
                self.review_store.update_status(item_id, status=status, resolution_note=note)
                flash_message = f"리뷰 항목 #{item_id} 상태를 '{STATUS_LABELS.get(status, status)}'로 저장했습니다."
                flash_tone = "success"

        domain = _first(form, "domain", "warehouse_exception")
        scenario = _first(form, "scenario", "qa")
        query = _first(form, "query", "")
        context = _first(form, "context", "")
        compare_baseline = _first(form, "compare_baseline", "0") == "1"
        baseline_name = _first(form, "baseline", "lexical_rag")
        selected_example = _first(form, "example_id", "")
        selected_review_id = int(_first(form, "selected_review_id", "0") or 0)

        if selected_example:
            matched = next((item for item in self.examples if item.get("id") == selected_example), None)
            if matched is not None:
                domain = matched.get("domain", domain)
                scenario = matched.get("scenario", scenario)
                query = matched.get("query", query)
                context = matched.get("context", context)
                if action == "load_example":
                    flash_message = f"예제 '{selected_example}'를 불러왔습니다. 아래에서 바로 실행하거나 문장을 수정해 볼 수 있습니다."
                    flash_tone = "neutral"

        copilot_result = None
        baseline_result = None
        if query.strip() and action in {"run", ""}:
            request = CopilotRequest(query=query, context=context, domain=domain, scenario=scenario)
            copilot_result = self.copilot.run(request)
            if compare_baseline:
                baseline_result = BaselineRunner(baseline_name, config_path=self.baseline_config).answer(query, context, domain=domain)
            flash_message = "테스트가 완료되었습니다. 아래에서 구조 추론 결과와 baseline 차이를 확인하세요."
            flash_tone = "success"

        review_stats = self.review_store.fetch_stats() if self.review_store is not None else {}
        pending = self.review_store.fetch_pending(limit=10) if self.review_store is not None else []
        recent = self.review_store.fetch_items(status=None, limit=10) if self.review_store is not None else []
        selected_review = self.review_store.fetch_item_detail(selected_review_id) if self.review_store is not None and selected_review_id else None
        return render_page(
            query=query,
            context=context,
            domain=domain,
            scenario=scenario,
            compare_baseline=compare_baseline,
            baseline_name=baseline_name,
            copilot_result=copilot_result,
            baseline_result=baseline_result,
            examples=self.examples,
            selected_example=selected_example,
            pending_reviews=pending,
            recent_reviews=recent,
            selected_review=selected_review,
            review_stats=review_stats,
            feedback_rules=self.feedback_rules,
            baseline_config=self.baseline_config,
            flash_message=flash_message,
            flash_tone=flash_tone,
        )


def render_page(
    *,
    query: str,
    context: str,
    domain: str,
    scenario: str,
    compare_baseline: bool,
    baseline_name: str,
    copilot_result,
    baseline_result,
    examples: list[dict],
    selected_example: str,
    pending_reviews,
    recent_reviews,
    selected_review,
    review_stats,
    feedback_rules: str | None,
    baseline_config: str | None,
    flash_message: str,
    flash_tone: str,
) -> str:
    domain_options = []
    for value, label in DOMAIN_LABELS.items():
        selected = " selected" if value == domain else ""
        domain_options.append(f'<option value="{value}"{selected}>{html.escape(label)}</option>')

    baseline_options = []
    for value, spec in BASELINE_SPECS.items():
        selected = " selected" if value == baseline_name else ""
        baseline_options.append(
            f'<option value="{value}"{selected}>{html.escape(value)} | {html.escape(spec.description)}</option>'
        )

    config_note = []
    if baseline_config:
        config_note.append(f"baseline config: {baseline_config}")
    if feedback_rules:
        config_note.append(f"feedback rules: {feedback_rules}")
    config_html = ""
    if config_note:
        config_html = "".join(f'<div class="mini-chip">{html.escape(note)}</div>' for note in config_note)

    scenario_hint = SCENARIO_HINTS.get(scenario, "시나리오 이름은 자유롭게 적을 수 있습니다.")
    result_html = render_results(copilot_result, baseline_result, baseline_name)
    review_html = render_review_panel(review_stats, pending_reviews, recent_reviews, selected_review)
    empty_state = render_empty_state(copilot_result is None)
    example_cards = render_example_cards(examples, selected_example)
    flash_html = render_flash(flash_message, flash_tone)

    return f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SemOp Ops Copilot</title>
<style>
:root {{
  --bg: #f4efe5;
  --ink: #16241d;
  --muted: #5b6d63;
  --card: rgba(255, 251, 245, 0.9);
  --line: #cbd7cc;
  --accent: #1d5a47;
  --accent-2: #d98f3d;
  --soft: #edf5ee;
  --warn: #b45e2f;
  --danger: #9d2f2f;
  --ok: #256347;
  --shadow: 0 16px 40px rgba(32, 50, 40, 0.10);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(217,143,61,0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(29,90,71,0.15), transparent 24%),
    linear-gradient(180deg, #f7f1e8 0%, #edf3ef 100%);
  font-family: "Segoe UI", "Malgun Gothic", sans-serif;
}}
main {{ max-width: 1320px; margin: 0 auto; padding: 24px 20px 40px; }}
a {{ color: inherit; }}
.hero {{
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 18px;
  margin-bottom: 20px;
}}
.hero-card, .side-card, .panel {{
  background: var(--card);
  border: 1px solid rgba(122, 147, 132, 0.24);
  box-shadow: var(--shadow);
  border-radius: 24px;
}}
.hero-card {{ padding: 28px; }}
.hero h1 {{ font-size: 34px; margin: 0 0 12px; letter-spacing: -0.03em; }}
.lead {{ color: var(--muted); font-size: 16px; line-height: 1.6; margin: 0 0 18px; }}
.hero-points, .mini-list {{ display: grid; gap: 10px; }}
.hero-point {{ display: flex; gap: 10px; align-items: flex-start; color: var(--ink); }}
.hero-point strong {{ min-width: 78px; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
.side-card {{ padding: 22px; display: grid; gap: 14px; }}
.side-card h2, .panel h2 {{ margin: 0; font-size: 19px; }}
.side-card p, .panel p, .small {{ color: var(--muted); line-height: 1.55; }}
.chips, .inline-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip, .mini-chip, .status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 12px;
  font-weight: 700;
}}
.chip {{ background: #f0f6f1; color: var(--accent); }}
.mini-chip {{ background: #f8f2e7; color: #84542f; }}
.status-pill {{ background: #eef4ef; color: var(--ok); }}
.flash {{ margin: 0 0 18px; padding: 14px 16px; border-radius: 16px; font-weight: 600; }}
.flash.success {{ background: #e8f5ed; color: var(--ok); border: 1px solid #b6d8bf; }}
.flash.neutral {{ background: #eef3f7; color: #31556f; border: 1px solid #cbd9e4; }}
.layout {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; align-items: start; }}
.panel {{ padding: 22px; }}
.section-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 14px; }}
.section-title h2 {{ font-size: 22px; }}
.helper {{ font-size: 13px; color: var(--muted); }}
form {{ display: grid; gap: 14px; }}
label {{ display: grid; gap: 7px; font-weight: 700; }}
.label-note {{ font-size: 12px; color: var(--muted); font-weight: 500; }}
input, select, textarea {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.9);
  color: var(--ink);
  font: inherit;
}}
textarea {{ min-height: 152px; resize: vertical; }}
textarea.context-box {{ min-height: 260px; }}
button {{
  border: 0;
  border-radius: 999px;
  padding: 12px 18px;
  background: var(--accent);
  color: white;
  font: inherit;
  font-weight: 800;
  cursor: pointer;
}}
button.secondary {{ background: #e7efe8; color: var(--accent); }}
button.ghost {{ background: white; color: var(--accent); border: 1px solid var(--line); }}
input[type="checkbox"] {{ width: auto; }}
.grid-2 {{ display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }}
.example-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
.example-card {{
  padding: 16px;
  border: 1px solid rgba(121, 141, 130, 0.25);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,246,242,0.9));
  display: grid;
  gap: 10px;
}}
.example-card.active {{ border-color: rgba(29,90,71,0.5); box-shadow: 0 10px 26px rgba(29,90,71,0.10); }}
.example-card h3 {{ margin: 0; font-size: 16px; }}
.example-card p {{ margin: 0; font-size: 13px; color: var(--muted); line-height: 1.5; }}
.example-card form {{ gap: 8px; }}
.hint-box {{ background: #f6efe4; border-radius: 16px; padding: 14px; color: #765338; font-size: 13px; line-height: 1.55; }}
.result-grid {{ display: grid; gap: 18px; grid-template-columns: 1.2fr 1fr; margin-top: 18px; }}
.metric-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.metric-card {{ padding: 14px; border-radius: 18px; background: #f5f8f5; border: 1px solid rgba(121, 141, 130, 0.15); display: grid; gap: 6px; }}
.metric-card.good {{ background: #e9f6ee; }}
.metric-card.caution {{ background: #fff3e6; }}
.metric-card.risk {{ background: #faecec; }}
.metric-name {{ font-size: 12px; font-weight: 800; color: var(--muted); }}
.metric-value {{ font-size: 26px; font-weight: 900; letter-spacing: -0.04em; }}
.metric-state {{ font-size: 12px; font-weight: 800; }}
.result-card {{ display: grid; gap: 14px; }}
.result-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }}
.result-head h3 {{ margin: 0; font-size: 20px; }}
.result-copy, pre {{
  margin: 0;
  background: #f5f8f5;
  border-radius: 18px;
  padding: 16px;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
  border: 1px solid rgba(121, 141, 130, 0.15);
}}
.callout {{ padding: 12px 14px; border-radius: 16px; background: #fff7ea; color: #7b5329; font-size: 13px; line-height: 1.55; }}
.chunk-list, .review-list {{ display: grid; gap: 10px; }}
.chunk-item, .review-item {{ border: 1px solid rgba(121, 141, 130, 0.18); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.75); }}
.chunk-item strong, .review-item strong {{ display: block; margin-bottom: 6px; }}
.review-item-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 8px; }}
.review-stat-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); margin-bottom: 14px; }}
.review-stat {{ padding: 14px; border-radius: 18px; background: #f5f8f5; display: grid; gap: 4px; }}
.review-stat span {{ font-size: 24px; font-weight: 900; }}
.inline-form {{ display: inline; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
.dual-panel {{ display: grid; gap: 18px; grid-template-columns: 1fr 1fr; }}
.empty-state {{
  padding: 22px;
  border-radius: 22px;
  border: 1px dashed rgba(121, 141, 130, 0.35);
  background: rgba(255,255,255,0.45);
  color: var(--muted);
  display: grid;
  gap: 10px;
}}
code {{ font-family: Consolas, monospace; font-size: 0.95em; }}
@media (max-width: 1080px) {{
  .hero, .layout, .result-grid, .dual-panel {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 720px) {{
  main {{ padding: 16px 14px 28px; }}
  .hero-card, .side-card, .panel {{ border-radius: 20px; }}
  .grid-2 {{ grid-template-columns: 1fr; }}
  .metric-grid {{ grid-template-columns: 1fr 1fr; }}
  .hero h1 {{ font-size: 28px; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div class="hero-card">
      <div class="chips">
        <div class="chip">초보 테스터용 GUI</div>
        <div class="chip">SemOp 구조 추론</div>
        <div class="chip">Baseline 비교</div>
      </div>
      <h1>현장 문맥을 읽는 SOP Copilot 테스트 화면</h1>
      <p class="lead">질문만 보는 챗봇이 아니라, 숨겨진 승인 조건, 막힌 경로, 대체 절차, 위험 요소를 같이 복원하는지 확인하는 화면입니다. 처음 써보는 테스터도 예제를 불러 바로 비교할 수 있게 정리했습니다.</p>
      <div class="hero-points">
        <div class="hero-point"><strong>1. 입력</strong><span>질문과 SOP 문맥을 넣거나 아래 예제를 불러옵니다.</span></div>
        <div class="hero-point"><strong>2. 실행</strong><span>SemOp 결과와 일반 baseline을 같이 비교합니다.</span></div>
        <div class="hero-point"><strong>3. 검토</strong><span>위험한 답변은 review queue에서 승인, 반려, 추가 확인으로 처리합니다.</span></div>
      </div>
    </div>
    <aside class="side-card">
      <h2>바로 확인할 포인트</h2>
      <div class="mini-list">
        <div>질문에 없지만 문맥에 숨어 있는 <strong>사전조건</strong>을 잡았는지</div>
        <div>막힌 통로, 승인 누락, 안전 문제 같은 <strong>차단 조건</strong>을 놓치지 않았는지</div>
        <div>답변 끝에 <strong>근거와 감사 추적</strong>이 남는지</div>
      </div>
      <div class="inline-chips">{config_html}</div>
    </aside>
  </section>

  {flash_html}

  <section class="layout">
    <div class="panel">
      <div class="section-title">
        <h2>Step 1. 예제로 시작하거나 직접 입력</h2>
        <span class="helper">처음이면 예제를 먼저 불러보는 편이 빠릅니다.</span>
      </div>
      {example_cards}
    </div>
    <div class="panel">
      <div class="section-title">
        <h2>Step 2. 테스트 설정</h2>
        <span class="helper">현재 시나리오: {html.escape(scenario_hint)}</span>
      </div>
      <form method="post">
        <input type="hidden" name="action" value="run">
        <input type="hidden" name="example_id" value="{html.escape(selected_example)}">
        <div class="grid-2">
          <label>도메인
            <span class="label-note">업무 유형에 맞는 프로필을 고릅니다.</span>
            <select name="domain">{''.join(domain_options)}</select>
          </label>
          <label>시나리오
            <span class="label-note">예: qa, exception_response, onboarding</span>
            <input name="scenario" value="{html.escape(scenario)}" placeholder="exception_response">
          </label>
        </div>
        <label>질문
          <span class="label-note">실무자가 실제로 물을 만한 문장으로 적어 주세요.</span>
          <textarea name="query" placeholder="예: 통로가 막혀 있고 승인도 없는데 지금 출고를 진행해도 되나요?">{html.escape(query)}</textarea>
        </label>
        <label>현장 문맥 또는 SOP 본문
          <span class="label-note">매뉴얼, 작업 절차, FAQ, 장애 대응 문서를 붙여 넣습니다.</span>
          <textarea class="context-box" name="context" placeholder="예: 승인 없는 출고는 금지. 통로가 막히면 작업 중단 후 안전 담당자 확인...">{html.escape(context)}</textarea>
        </label>
        <div class="grid-2">
          <label>비교 설정
            <span class="label-note">SemOp만 볼지, baseline도 같이 볼지 선택합니다.</span>
            <div class="hint-box">
              <label style="display:flex; gap:10px; align-items:center; font-weight:700;">
                <input type="checkbox" name="compare_baseline" value="1"{' checked' if compare_baseline else ''}>
                baseline 결과도 함께 비교하기
              </label>
            </div>
          </label>
          <label>Baseline 종류
            <span class="label-note">일반 RAG나 고객 정의 키워드 baseline과 비교합니다.</span>
            <select name="baseline">{''.join(baseline_options)}</select>
          </label>
        </div>
        <div class="actions">
          <button type="submit">SemOp 테스트 실행</button>
          <span class="helper">실행 후 아래에서 KPI, 근거, review queue 상태를 확인할 수 있습니다.</span>
        </div>
      </form>
    </div>
  </section>

  <section class="panel">
    <div class="section-title">
      <h2>Step 3. 결과 비교</h2>
      <span class="helper">SemOp이 숨은 조건과 실행 제약을 더 잘 복원하는지 확인합니다.</span>
    </div>
    {empty_state}
    {result_html}
  </section>

  {review_html}
</main>
</body>
</html>
"""


def render_example_cards(examples: list[dict], selected_example: str) -> str:
    if not examples:
        return '<div class="empty-state"><strong>예제 파일이 없습니다.</strong><span><code>examples/ops_sop_cases_ko.jsonl</code>를 확인하세요.</span></div>'
    cards: list[str] = ["<div class=\"example-grid\">"]
    for item in examples[:6]:
        example_id = str(item.get("id", ""))
        label = DOMAIN_LABELS.get(item.get("domain", "general"), item.get("domain", "general"))
        summary = (item.get("query", "") or "")[:90]
        is_active = " active" if example_id == selected_example else ""
        cards.append(
            """
<div class="example-card{active}">
  <div class="inline-chips">
    <div class="mini-chip">{id}</div>
    <div class="status-pill">{domain}</div>
  </div>
  <h3>{scenario}</h3>
  <p>{summary}</p>
  <form method="post">
    <input type="hidden" name="action" value="load_example">
    <input type="hidden" name="example_id" value="{id_attr}">
    <button class="secondary" type="submit">이 예제 불러오기</button>
  </form>
</div>
""".format(
                active=is_active,
                id=html.escape(example_id),
                id_attr=html.escape(example_id),
                domain=html.escape(label),
                scenario=html.escape(str(item.get("scenario", "qa"))),
                summary=html.escape(summary + ("..." if len(item.get("query", "")) > 90 else "")),
            )
        )
    cards.append("</div>")
    return "".join(cards)


def render_results(copilot_result, baseline_result, baseline_name: str) -> str:
    if copilot_result is None:
        return ""
    queue_note = ""
    if copilot_result.queued_for_review:
        reasons = ", ".join(copilot_result.review_reasons)
        queue_note = f'<div class="callout">이 결과는 review queue에 들어갔습니다. 이유: {html.escape(reasons)}</div>'
    notes_html = ""
    if copilot_result.kpis.notes:
        note_items = "".join(f"<li>{html.escape(note)}</li>" for note in copilot_result.kpis.notes)
        notes_html = f"<div><strong>운영 메모</strong><ul>{note_items}</ul></div>"
    audit_items = "".join(
        f'<li><strong>{html.escape(item.stage)}</strong>: {html.escape(item.detail)}</li>' for item in copilot_result.audit_items
    )
    semop_html = f"""
<div class="result-card">
  <div class="result-head">
    <div>
      <h3>SemOp 구조 추론 결과</h3>
      <p class="small">도메인: {html.escape(DOMAIN_LABELS.get(copilot_result.request.domain, copilot_result.request.domain))}</p>
    </div>
    <div class="status-pill">핵심 추론 + 제약 확인</div>
  </div>
  {render_kpi_cards(copilot_result.kpis.model_dump())}
  {queue_note}
  <div>
    <strong>답변 전문</strong>
    <pre>{html.escape(copilot_result.to_text())}</pre>
  </div>
  {notes_html}
  <div>
    <strong>감사 추적</strong>
    <ul>{audit_items}</ul>
  </div>
</div>
"""
    baseline_html = ""
    if baseline_result is not None:
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in baseline_result.warnings)
        warning_block = f"<div><strong>경고</strong><ul>{warnings}</ul></div>" if baseline_result.warnings else ""
        baseline_html = f"""
<div class="result-card">
  <div class="result-head">
    <div>
      <h3>Baseline 비교: {html.escape(baseline_name)}</h3>
      <p class="small">문맥 chunk를 단순 검색했을 때의 결과입니다.</p>
    </div>
    <div class="status-pill">confidence {baseline_result.confidence}</div>
  </div>
  <div>
    <strong>Baseline 답변</strong>
    <div class="result-copy">{html.escape(baseline_result.answer_text)}</div>
  </div>
  {render_chunks(baseline_result.retrieved_chunks)}
  {warning_block}
</div>
"""
    else:
        baseline_html = """
<div class="result-card">
  <div class="result-head">
    <div>
      <h3>Baseline 비교 미실행</h3>
      <p class="small">입력 화면에서 'baseline 결과도 함께 비교하기'를 켜면 일반 검색형 답변과 나란히 볼 수 있습니다.</p>
    </div>
    <div class="status-pill">선택 사항</div>
  </div>
  <div class="callout">초보 테스터라면 같은 질문으로 SemOp과 baseline을 같이 실행해 보는 편이 차이를 이해하기 쉽습니다.</div>
</div>
"""
    return f'<div class="result-grid">{semop_html}{baseline_html}</div>'


def render_kpi_cards(kpis: dict[str, object]) -> str:
    cards: list[str] = ['<div class="metric-grid">']
    for key in KPI_LABELS:
        value = float(kpis.get(key, 0.0) or 0.0)
        state, card_class = classify_kpi(key, value)
        cards.append(
            f'<div class="metric-card {card_class}">'
            f'<div class="metric-name">{html.escape(KPI_LABELS[key])}</div>'
            f'<div class="metric-value">{value:.2f}</div>'
            f'<div class="metric-state">{html.escape(state)}</div>'
            '</div>'
        )
    cards.append("</div>")
    return "".join(cards)


def classify_kpi(name: str, value: float) -> tuple[str, str]:
    if name in LOW_IS_GOOD:
        if value <= 0.15:
            return ("양호", "good")
        if value <= 0.4:
            return ("주의", "caution")
        return ("위험", "risk")
    if value >= 0.85:
        return ("양호", "good")
    if value >= 0.65:
        return ("주의", "caution")
    return ("위험", "risk")


def render_chunks(chunks) -> str:
    if not chunks:
        return '<div class="callout">검색된 supporting chunk가 없습니다.</div>'
    items = ['<div><strong>참고한 문맥 조각</strong><div class="chunk-list">']
    for chunk in chunks:
        items.append(
            f'<div class="chunk-item"><strong>{html.escape(chunk.chunk_id)}</strong><div>{html.escape(chunk.text)}</div></div>'
        )
    items.append("</div></div>")
    return "".join(items)


def render_review_panel(review_stats, pending_reviews, recent_reviews, selected_review) -> str:
    if not review_stats:
        return ""
    stats = []
    for key in ["total", "pending", "approved", "rejected", "needs_followup"]:
        stats.append(
            f'<div class="review-stat"><strong>{html.escape(STATUS_LABELS.get(key, key.title()))}</strong><span>{review_stats.get(key, 0)}</span></div>'
        )
    selected_block = render_selected_review(selected_review)
    pending_block = render_review_list("검토 대기 항목", pending_reviews, empty_label="현재 대기 항목이 없습니다.")
    recent_block = render_review_list("최근 리뷰 이력", recent_reviews, empty_label="최근 이력이 없습니다.")
    return f"""
<section class="panel">
  <div class="section-title">
    <h2>Step 4. 리뷰 큐 처리</h2>
    <span class="helper">위험하거나 애매한 답변은 여기서 처리합니다.</span>
  </div>
  <div class="review-stat-grid">{''.join(stats)}</div>
  <div class="dual-panel">
    <div>{pending_block}{recent_block}</div>
    <div>{selected_block}</div>
  </div>
</section>
"""


def render_review_list(title: str, items, empty_label: str) -> str:
    if not items:
        return f'<div class="panel" style="padding:16px;"><h3>{html.escape(title)}</h3><div class="empty-state">{html.escape(empty_label)}</div></div>'
    rows = [f'<div class="panel" style="padding:16px;"><h3>{html.escape(title)}</h3><div class="review-list">']
    for item in items:
        reasons = ", ".join(item.reasons)
        rows.append(
            f'<div class="review-item">'
            f'<div class="review-item-header"><strong>#{item.id} {html.escape(STATUS_LABELS.get(item.status, item.status))}</strong>'
            f'<span class="small">{html.escape(DOMAIN_LABELS.get(item.domain, item.domain))}</span></div>'
            f'<div class="small">{html.escape(item.query)}</div>'
            f'<div class="small">리뷰 이유: {html.escape(reasons)}</div>'
            f'<form method="post" class="actions" style="margin-top:10px;">'
            f'<input type="hidden" name="action" value="run">'
            f'<input type="hidden" name="selected_review_id" value="{item.id}">'
            f'<button class="ghost" type="submit">상세 보기</button>'
            f'</form>'
            f'</div>'
        )
    rows.append("</div></div>")
    return "".join(rows)


def render_selected_review(selected_review) -> str:
    if selected_review is None:
        return '<div class="panel" style="padding:16px;"><h3>선택된 리뷰 항목</h3><div class="empty-state">왼쪽 목록에서 항목을 눌러 상세 내용을 확인하세요.</div></div>'
    kpi_cards = render_kpi_cards(selected_review.get("kpis", {}))
    reasons = ", ".join(selected_review.get("reasons", []))
    audit_rows = "".join(
        f'<li><strong>{html.escape(str(item.get("stage", "")))}</strong>: {html.escape(str(item.get("detail", "")))}</li>'
        for item in selected_review.get("audit_items", [])
    )
    return f"""
<div class="panel" style="padding:16px;">
  <h3>선택된 리뷰 항목 #{selected_review['id']}</h3>
  <p class="small">상태: {html.escape(STATUS_LABELS.get(selected_review['status'], selected_review['status']))}</p>
  <div class="callout">리뷰 이유: {html.escape(reasons)}</div>
  <div>
    <strong>질문</strong>
    <div class="result-copy">{html.escape(selected_review['query'])}</div>
  </div>
  <div>
    <strong>답변</strong>
    <pre>{html.escape(selected_review.get('answer_text', ''))}</pre>
  </div>
  <div>
    <strong>당시 KPI</strong>
    {kpi_cards}
  </div>
  <div>
    <strong>감사 추적</strong>
    <ul>{audit_rows}</ul>
  </div>
  <form method="post">
    <input type="hidden" name="action" value="review_update">
    <input type="hidden" name="review_id" value="{selected_review['id']}">
    <div class="grid-2">
      <label>처리 상태
        <select name="review_status">
          {''.join(render_review_status_options(selected_review['status']))}
        </select>
      </label>
      <label>처리 메모
        <textarea name="resolution_note" style="min-height:120px;">{html.escape(selected_review.get('resolution_note', ''))}</textarea>
      </label>
    </div>
    <div class="actions">
      <button type="submit">리뷰 상태 저장</button>
    </div>
  </form>
</div>
"""


def render_review_status_options(current: str) -> list[str]:
    options: list[str] = []
    for value in ["pending", "approved", "rejected", "needs_followup"]:
        selected = " selected" if value == current else ""
        options.append(f'<option value="{value}"{selected}>{html.escape(STATUS_LABELS.get(value, value))}</option>')
    return options


def render_empty_state(show: bool) -> str:
    if not show:
        return ""
    return """
<div class="empty-state">
  <strong>아직 실행 결과가 없습니다.</strong>
  <span>위에서 예제를 불러오거나 질문과 SOP 문맥을 넣은 뒤 <code>SemOp 테스트 실행</code> 버튼을 누르세요.</span>
  <span>처음 테스트라면 <code>창고 예외 대응</code> 도메인과 baseline 비교를 같이 켜는 편이 차이를 보기 쉽습니다.</span>
</div>
"""


def render_flash(message: str, tone: str) -> str:
    if not message:
        return ""
    return f'<div class="flash {html.escape(tone)}">{html.escape(message)}</div>'


def _first(form: dict[str, list[str]], key: str, default: str) -> str:
    values = form.get(key)
    if not values:
        return default
    return values[0]



def make_handler(app: OpsGuiApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = app.handle({}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            form = parse_qs(payload)
            body = app.handle(form).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler



def main() -> None:
    parser = argparse.ArgumentParser(description="Local GUI for SemOp operations copilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--memory-store")
    parser.add_argument("--memory-source")
    parser.add_argument("--review-queue", default="data/ops_review_queue.db")
    parser.add_argument("--feedback-rules")
    parser.add_argument("--baseline-config")
    args = parser.parse_args()

    app = OpsGuiApp(
        mode=args.mode,
        model_id=args.model_id,
        memory_store=args.memory_store,
        memory_source=args.memory_source,
        review_queue=args.review_queue,
        feedback_rules=args.feedback_rules,
        baseline_config=args.baseline_config,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"SemOp Ops GUI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
