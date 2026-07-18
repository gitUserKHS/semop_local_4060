from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from .beginner import (
    BeginnerInputError,
    BeginnerReasoner,
    vision_presets_for_ui,
)


MAX_REQUEST_BYTES = 64 * 1024


class BeginnerHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: BeginnerReasoner | None = None,
    ) -> None:
        super().__init__(server_address, BeginnerRequestHandler)
        self.service = service or BeginnerReasoner()


class BeginnerRequestHandler(BaseHTTPRequestHandler):
    server: BeginnerHttpServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html(render_home_page())
            return
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "semop-beginner"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "페이지를 찾지 못했어."})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api/solve":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "요청 주소가 달라."})
            return

        try:
            payload = self._read_json()
            domain = payload.get("domain", "")
            values = payload.get("values", {})
            if not isinstance(values, Mapping):
                raise BeginnerInputError("입력 형식이 올바르지 않아. 화면을 새로고침해 줘.")
            result = self.server.service.solve(str(domain), values)
        except BeginnerInputError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "요청 내용을 읽지 못했어. 화면을 새로고침해 줘."},
            )
            return
        except Exception as exc:
            self.log_error("solve failed: %s: %s", type(exc).__name__, exc)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "풀이 중 오류가 났어. 터미널의 오류 기록을 확인해 줘."},
            )
            return

        self._send_json(HTTPStatus.OK, {"ok": True, "result": result.to_dict()})

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise BeginnerInputError("요청 크기를 확인할 수 없어.") from exc
        if length <= 0:
            raise BeginnerInputError("입력 내용이 비어 있어.")
        if length > MAX_REQUEST_BYTES:
            raise BeginnerInputError("입력이 너무 길어. 64KB보다 작게 줄여 줘.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise BeginnerInputError("입력 형식이 올바르지 않아.")
        return payload

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._common_headers("text/html; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=None,
            default=str,
        ).encode("utf-8")
        self.send_response(status)
        self._common_headers("application/json; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _common_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'",
        )


def render_home_page() -> str:
    presets = json.dumps(vision_presets_for_ui(), ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__VISION_PRESETS__", presets)


def create_server(
    preferred_port: int = 8765,
    *,
    service: BeginnerReasoner | None = None,
    attempts: int = 10,
) -> BeginnerHttpServer:
    if not 0 <= preferred_port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    ports = (
        (0,)
        if preferred_port == 0
        else range(preferred_port, min(65536, preferred_port + attempts))
    )
    last_error: OSError | None = None
    for port in ports:
        try:
            return BeginnerHttpServer(("127.0.0.1", port), service)
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SemOp 초보자용 로컬 브라우저 화면을 실행합니다."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="시작 포트입니다. 사용 중이면 다음 포트를 자동으로 찾습니다. (기본: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="브라우저를 자동으로 열지 않습니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        server = create_server(args.port)
    except (OSError, ValueError) as exc:
        print(f"SemOp을 시작하지 못했어: {exc}")
        return 1

    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/"
    print("SemOp 쉬운 시작이 준비됐어.")
    print(f"브라우저 주소: {url}")
    print("끝낼 때는 이 창에서 Ctrl+C를 눌러 줘.")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nSemOp을 종료했어.")
    finally:
        server.server_close()
    return 0


_PAGE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SemOp 쉬운 시작</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #22233a;
      --muted: #67677d;
      --paper: #fffdf9;
      --panel: #ffffff;
      --line: #e9e3dc;
      --accent: #8758c7;
      --accent-dark: #6841a1;
      --accent-soft: #f1eafd;
      --ok: #177754;
      --ok-soft: #e8f7f0;
      --warn: #9a5b12;
      --warn-soft: #fff4df;
      --fail: #a63b49;
      --fail-soft: #fff0f1;
      font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #f8ebff 0, transparent 31rem),
        radial-gradient(circle at 90% 10%, #fff0df 0, transparent 27rem),
        var(--paper);
    }
    button, input, textarea, select { font: inherit; }
    .shell { width: min(980px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 64px; }
    .hero { text-align: center; margin-bottom: 24px; }
    .eyebrow {
      display: inline-block; padding: 7px 11px; border-radius: 999px;
      color: var(--accent-dark); background: var(--accent-soft); font-size: 13px; font-weight: 800;
    }
    h1 { margin: 14px 0 8px; font-size: clamp(32px, 6vw, 50px); letter-spacing: -0.04em; }
    .hero p { margin: 0 auto; max-width: 650px; color: var(--muted); line-height: 1.7; }
    .scope {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
      margin: 24px 0; padding: 12px; background: rgba(255,255,255,.72);
      border: 1px solid var(--line); border-radius: 20px;
    }
    .scope div { padding: 11px; text-align: center; font-size: 14px; color: var(--muted); }
    .scope strong { display: block; margin-bottom: 3px; color: var(--ink); }
    .workspace {
      overflow: hidden; background: rgba(255,255,255,.93); border: 1px solid var(--line);
      border-radius: 26px; box-shadow: 0 22px 70px rgba(57, 39, 78, .10);
    }
    .tabs { display: grid; grid-template-columns: repeat(3, 1fr); padding: 10px; gap: 8px; border-bottom: 1px solid var(--line); }
    .tab {
      padding: 13px 8px; border: 0; border-radius: 14px; color: var(--muted);
      background: transparent; cursor: pointer; font-weight: 800;
    }
    .tab[aria-selected="true"] { color: var(--accent-dark); background: var(--accent-soft); }
    .pane { display: none; padding: clamp(20px, 5vw, 38px); }
    .pane.active { display: block; }
    .pane h2 { margin: 0 0 7px; font-size: 24px; letter-spacing: -0.025em; }
    .lead { margin: 0 0 22px; color: var(--muted); line-height: 1.65; }
    .field { margin-bottom: 18px; }
    label { display: block; margin-bottom: 7px; font-weight: 800; }
    .hint { display: block; margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    input, textarea, select {
      width: 100%; padding: 13px 14px; border: 1px solid #d9d2ca; border-radius: 12px;
      color: var(--ink); background: white; outline: none;
    }
    input:focus, textarea:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    textarea { min-height: 78px; resize: vertical; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 3px 0 22px; }
    .example {
      padding: 8px 11px; border: 1px solid var(--line); border-radius: 999px;
      color: var(--accent-dark); background: white; cursor: pointer; font-size: 13px; font-weight: 700;
    }
    .example:hover { background: var(--accent-soft); }
    .solve {
      width: 100%; padding: 14px 18px; border: 0; border-radius: 13px;
      color: white; background: var(--accent); cursor: pointer; font-weight: 900;
      box-shadow: 0 9px 24px rgba(104, 65, 161, .22);
    }
    .solve:hover { background: var(--accent-dark); }
    .solve:disabled { opacity: .58; cursor: wait; }
    .vision-layout { display: grid; grid-template-columns: minmax(220px, .9fr) minmax(250px, 1.1fr); gap: 22px; align-items: center; }
    .preview-wrap { padding: 20px; border: 1px solid var(--line); border-radius: 18px; background: #faf8f4; }
    .pixel-grid { display: grid; gap: 3px; width: min(100%, 330px); margin: auto; }
    .pixel { aspect-ratio: 1; border-radius: 3px; border: 1px solid rgba(0,0,0,.035); }
    .pixel.dot { background: #fff; }
    .pixel.R { background: #f44336; }
    .pixel.G { background: #19b765; }
    .pixel.B { background: #2962ff; }
    .question { margin: 13px 0 0; text-align: center; font-weight: 800; line-height: 1.5; }
    .result { margin-top: 22px; padding: clamp(20px, 4vw, 30px); border-radius: 22px; border: 1px solid var(--line); background: var(--panel); }
    .result.ok { border-color: #b9e3d1; background: var(--ok-soft); }
    .result.warn { border-color: #f1d29e; background: var(--warn-soft); }
    .result.fail { border-color: #efc2c8; background: var(--fail-soft); }
    .result h2 { margin: 0 0 8px; font-size: 22px; }
    .result > p { margin: 0; line-height: 1.65; }
    .status-line { display: flex; align-items: center; gap: 9px; margin-bottom: 15px; color: var(--muted); font-size: 13px; font-weight: 800; }
    .dot-status { width: 10px; height: 10px; border-radius: 50%; background: currentColor; }
    .ok .status-line { color: var(--ok); }
    .warn .status-line { color: var(--warn); }
    .fail .status-line { color: var(--fail); }
    .interpreted { margin: 20px 0 0; padding: 16px 18px; border-radius: 14px; background: rgba(255,255,255,.72); }
    .interpreted strong { display: block; margin-bottom: 8px; }
    .interpreted ul { margin: 0; padding-left: 20px; line-height: 1.7; }
    .trust { margin-top: 16px; padding: 13px 15px; border-left: 4px solid currentColor; border-radius: 8px; background: rgba(255,255,255,.62); line-height: 1.65; font-size: 14px; }
    details { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,.10); }
    summary { cursor: pointer; font-weight: 800; }
    pre { overflow: auto; white-space: pre-wrap; padding: 14px; border-radius: 12px; background: #242334; color: #f7f4ff; line-height: 1.62; font-size: 13px; }
    .foot { margin-top: 18px; text-align: center; color: var(--muted); font-size: 13px; line-height: 1.6; }
    @media (max-width: 680px) {
      .shell { width: min(100% - 20px, 980px); padding-top: 24px; }
      .scope { grid-template-columns: 1fr; }
      .scope div { text-align: left; }
      .vision-layout { grid-template-columns: 1fr; }
      .tab { font-size: 14px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <span class="eyebrow">내 PC에서 돌아가는 검증형 추론</span>
      <h1>SemOp 쉬운 시작</h1>
      <p>전문 명령어 없이 예제를 고르거나 빈칸을 채워 봐. SemOp이 작은 typed 연산자를 조합해 결론을 찾고, 찾은 과정은 처음부터 다시 실행해 확인해.</p>
    </header>

    <section class="scope" aria-label="현재 지원 범위">
      <div><strong>언어</strong>목표와 필요 조건 확인</div>
      <div><strong>수학</strong>정확한 계산식과 일차방정식</div>
      <div><strong>비전</strong>작은 색상 격자 측정</div>
    </section>

    <section class="workspace">
      <nav class="tabs" aria-label="문제 종류">
        <button class="tab" type="button" data-domain="language" aria-selected="true">1. 언어 조건</button>
        <button class="tab" type="button" data-domain="math" aria-selected="false">2. 수학식</button>
        <button class="tab" type="button" data-domain="vision" aria-selected="false">3. 색상 비전</button>
      </nav>

      <form class="pane active" id="pane-language" data-domain="language">
        <h2>조건이 모두 갖춰졌는지 확인하기</h2>
        <p class="lead">목표 하나와 그 목표에 필요한 조건을 적어 줘. 쉼표로 여러 개를 나눌 수 있어.</p>
        <div class="examples">
          <button class="example" type="button" data-language-example="ready">예제: 배포 준비 완료</button>
          <button class="example" type="button" data-language-example="blocked">예제: 승인에서 막힘</button>
          <button class="example" type="button" data-language-example="missing">예제: 확인이 덜 됨</button>
        </div>
        <div class="field">
          <label for="lang-goal">무엇을 하려는 거야?</label>
          <input id="lang-goal" value="배포" maxlength="80" required>
        </div>
        <div class="field">
          <label for="lang-required">꼭 필요한 조건</label>
          <textarea id="lang-required" required>테스트 통과, 관리자 승인</textarea>
          <span class="hint">예: 테스트 통과, 관리자 승인</span>
        </div>
        <div class="field">
          <label for="lang-satisfied">이미 충족한 조건</label>
          <textarea id="lang-satisfied">테스트 통과, 관리자 승인</textarea>
        </div>
        <div class="field">
          <label for="lang-blocked">막혔거나 실패한 조건</label>
          <textarea id="lang-blocked" placeholder="없으면 비워 둬"></textarea>
        </div>
        <button class="solve" type="submit">조건 검증하기</button>
      </form>

      <form class="pane" id="pane-math" data-domain="math">
        <h2>수학식이 맞는지 계산하고 검증하기</h2>
        <p class="lead">지원 범위 안에서는 답만 내지 않고, 리터럴과 연산을 한 단계씩 실행한 뒤 다시 확인해.</p>
        <div class="examples">
          <button class="example" type="button" data-math-example="(2 + 3) * 4 == 20">예제: 사칙연산</button>
          <button class="example" type="button" data-math-example="3*x + 2 = 11">예제: 일차방정식</button>
          <button class="example" type="button" data-math-example="7 &lt; 10">예제: 크기 비교</button>
        </div>
        <div class="field">
          <label for="math-expression">검증할 식</label>
          <input id="math-expression" value="(2 + 3) * 4 == 20" maxlength="240" required>
          <span class="hint">예: (2 + 3) * 4 == 20 또는 3*x + 2 = 11</span>
        </div>
        <button class="solve" type="submit">수학식 검증하기</button>
      </form>

      <form class="pane" id="pane-vision" data-domain="vision">
        <h2>작은 색상 장면을 연산자로 살펴보기</h2>
        <p class="lead">아직 일반 사진은 아니야. 먼저 정확히 검증할 수 있는 색상 픽셀 장면 네 개로 작동 방식을 체험해 봐.</p>
        <div class="vision-layout">
          <div class="field">
            <label for="vision-preset">장면과 질문</label>
            <select id="vision-preset"></select>
            <span class="hint" id="vision-description"></span>
          </div>
          <div class="preview-wrap">
            <div class="pixel-grid" id="pixel-grid" aria-label="색상 장면 미리보기"></div>
            <p class="question" id="vision-question"></p>
          </div>
        </div>
        <button class="solve" type="submit">색상 장면 검증하기</button>
      </form>
    </section>

    <section class="result" id="result" hidden aria-live="polite">
      <div class="status-line"><span class="dot-status"></span><span id="result-status"></span></div>
      <h2 id="result-title"></h2>
      <p id="result-summary"></p>
      <div class="interpreted">
        <strong>SemOp이 이렇게 이해했어</strong>
        <ul id="result-input"></ul>
      </div>
      <div class="trust" id="result-trust"></div>
      <details>
        <summary>검증 과정 보기</summary>
        <pre id="result-proof"></pre>
      </details>
      <details>
        <summary>연구용 기술 정보 보기</summary>
        <pre id="result-technical"></pre>
      </details>
    </section>

    <p class="foot">모든 처리는 이 PC의 로컬 서버에서 이뤄져. 이 화면은 범용 챗봇이 아니라 현재 검증 가능한 MVP 범위만 보여 줘.</p>
  </main>

  <script>
    const visionPresets = __VISION_PRESETS__;
    const panes = [...document.querySelectorAll('.pane')];
    const tabs = [...document.querySelectorAll('.tab')];
    const resultBox = document.getElementById('result');
    const visionSelect = document.getElementById('vision-preset');

    function selectDomain(domain) {
      tabs.forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.domain === domain)));
      panes.forEach(pane => pane.classList.toggle('active', pane.dataset.domain === domain));
      resultBox.hidden = true;
    }

    tabs.forEach(tab => tab.addEventListener('click', () => selectDomain(tab.dataset.domain)));

    const languageExamples = {
      ready: ['배포', '테스트 통과, 관리자 승인', '테스트 통과, 관리자 승인', ''],
      blocked: ['배포', '테스트 통과, 관리자 승인', '테스트 통과', '관리자 승인'],
      missing: ['배포', '테스트 통과, 관리자 승인', '테스트 통과', '']
    };
    document.querySelectorAll('[data-language-example]').forEach(button => {
      button.addEventListener('click', () => {
        const values = languageExamples[button.dataset.languageExample];
        ['lang-goal', 'lang-required', 'lang-satisfied', 'lang-blocked'].forEach((id, index) => {
          document.getElementById(id).value = values[index];
        });
      });
    });
    document.querySelectorAll('[data-math-example]').forEach(button => {
      button.addEventListener('click', () => {
        document.getElementById('math-expression').value = button.dataset.mathExample;
      });
    });

    visionPresets.forEach(preset => {
      const option = document.createElement('option');
      option.value = preset.key;
      option.textContent = `${preset.title}: ${preset.question}`;
      visionSelect.appendChild(option);
    });

    function renderVision() {
      const preset = visionPresets.find(item => item.key === visionSelect.value) || visionPresets[0];
      const grid = document.getElementById('pixel-grid');
      grid.replaceChildren();
      grid.style.gridTemplateColumns = `repeat(${preset.pattern[0].length}, 1fr)`;
      preset.pattern.forEach(row => [...row].forEach(color => {
        const pixel = document.createElement('span');
        pixel.className = `pixel ${color === '.' ? 'dot' : color}`;
        grid.appendChild(pixel);
      }));
      document.getElementById('vision-question').textContent = preset.question;
      document.getElementById('vision-description').textContent = preset.description;
    }
    visionSelect.addEventListener('change', renderVision);
    renderVision();

    function payloadFor(domain) {
      if (domain === 'language') {
        return {
          goal: document.getElementById('lang-goal').value,
          required: document.getElementById('lang-required').value,
          satisfied: document.getElementById('lang-satisfied').value,
          blocked: document.getElementById('lang-blocked').value
        };
      }
      if (domain === 'math') {
        return {expression: document.getElementById('math-expression').value};
      }
      return {preset: visionSelect.value};
    }

    function showError(message) {
      resultBox.hidden = false;
      resultBox.className = 'result fail';
      document.getElementById('result-status').textContent = '입력 확인 필요';
      document.getElementById('result-title').textContent = '입력을 다시 확인해 줘';
      document.getElementById('result-summary').textContent = message;
      document.getElementById('result-input').replaceChildren();
      document.getElementById('result-trust').textContent = '입력을 고치면 같은 자리에서 다시 검증할 수 있어.';
      document.getElementById('result-proof').textContent = '아직 실행된 증명이 없어.';
      document.getElementById('result-technical').textContent = '';
      resultBox.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function showResult(result) {
      const isVerified = result.success && result.verified;
      const kind = result.conclusion === 'not_ready' ? 'warn' : (isVerified ? 'ok' : 'fail');
      resultBox.hidden = false;
      resultBox.className = `result ${kind}`;
      document.getElementById('result-status').textContent = result.conclusion === 'not_ready'
        ? '검증됨: 진행 조건 미충족'
        : (isVerified ? '검증된 결론' : '미증명');
      document.getElementById('result-title').textContent = result.title;
      document.getElementById('result-summary').textContent = result.summary;
      const list = document.getElementById('result-input');
      list.replaceChildren();
      result.interpreted_input.forEach(line => {
        const item = document.createElement('li');
        item.textContent = line;
        list.appendChild(item);
      });
      document.getElementById('result-trust').textContent = result.trust_notice;
      document.getElementById('result-proof').textContent = result.proof;
      document.getElementById('result-technical').textContent = JSON.stringify(result.technical, null, 2);
      resultBox.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    panes.forEach(form => form.addEventListener('submit', async event => {
      event.preventDefault();
      const domain = form.dataset.domain;
      const button = form.querySelector('.solve');
      const oldText = button.textContent;
      button.disabled = true;
      button.textContent = '연산자를 조합하는 중...';
      try {
        const response = await fetch('/api/solve', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({domain, values: payloadFor(domain)})
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || '요청을 처리하지 못했어.');
        showResult(data.result);
      } catch (error) {
        showError(error.message || '로컬 서버와 연결하지 못했어.');
      } finally {
        button.disabled = false;
        button.textContent = oldText;
      }
    }));
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
