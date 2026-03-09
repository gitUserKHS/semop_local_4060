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

from semop import CompetitiveProgrammingReasoner, CpIncidentCase, CpIncidentIngestor, CpEpisodeStore

EXAMPLES_PATH = Path("examples/cp_statement_seeds.jsonl")
INCIDENTS_PATH = Path("examples/cp_incident_cases.jsonl")
CATEGORY_LABELS = {
    "prefix_sum_range_query": "Prefix sum / static range sum",
    "fenwick_tree": "Fenwick tree / online range sum",
    "segment_tree": "Segment tree",
    "lazy_segment_tree": "Lazy segment tree",
    "dijkstra_shortest_path": "Dijkstra shortest path",
    "dsu_connectivity": "Union-Find connectivity",
    "grid_bfs": "Grid BFS",
    "binary_search_answer": "Binary search on answer",
    "knapsack_dp": "Knapsack DP",
    "generic_contest_analysis": "Generic analysis",
}
FAILURE_LABELS = {
    "": "Not specified",
    "compile_error": "Compile error",
    "runtime_error": "Runtime error",
    "time_limit": "Time limit",
    "output_mismatch": "Wrong answer",
}


def _first(form: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form.get(key)
    return values[0] if values else default


def load_seed_examples() -> list[dict]:
    if not EXAMPLES_PATH.exists():
        return []
    records: list[dict] = []
    with EXAMPLES_PATH.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            statement = payload.get("statement", "").strip()
            if statement:
                records.append({"id": f"seed_{index}", "statement": statement})
    return records


def load_incident_examples() -> list[dict]:
    if not INCIDENTS_PATH.exists():
        return []
    records: list[dict] = []
    with INCIDENTS_PATH.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["id"] = payload.get("problem_id") or f"incident_{index}"
            records.append(payload)
    return records


class CpGuiApp:
    def __init__(self, episode_store: str | None, compiler: str) -> None:
        self.episode_store = episode_store
        self.reasoner = CompetitiveProgrammingReasoner(episode_store_path=episode_store, compiler=compiler)
        self.ingestor = CpIncidentIngestor(episode_store or "data/cp_episodes.db", compiler=compiler)
        self.examples = load_seed_examples()
        self.incidents = load_incident_examples()

    def handle(self, form: dict[str, list[str]]) -> str:
        action = _first(form, "action", "run")
        flash = ""
        flash_tone = "neutral"
        statement = _first(form, "statement", "")
        selected_example = _first(form, "example_id", "")
        selected_incident = _first(form, "incident_id", "")
        incident_problem_id = _first(form, "incident_problem_id", "")
        incident_editorial = _first(form, "incident_editorial", "")
        incident_outcome = _first(form, "incident_outcome", "")
        incident_failure = _first(form, "incident_failure", "")
        incident_code = _first(form, "incident_code", "")
        solve_missing = _first(form, "solve_missing", "1") == "1"

        if selected_example:
            matched = next((item for item in self.examples if item["id"] == selected_example), None)
            if matched is not None:
                statement = matched["statement"]
                if action == "load_example":
                    flash = "Loaded a seed problem. You can run it as-is or edit the statement first."

        if selected_incident:
            matched = next((item for item in self.incidents if item.get("id") == selected_incident), None)
            if matched is not None:
                statement = matched.get("statement", statement)
                incident_problem_id = matched.get("problem_id", incident_problem_id)
                incident_editorial = matched.get("editorial_summary", incident_editorial)
                incident_outcome = matched.get("outcome", incident_outcome)
                incident_failure = matched.get("failure_kind", incident_failure)
                incident_code = matched.get("code", incident_code)
                if action == "load_incident":
                    flash = "Loaded an incident example. Store it to grow episodic memory."

        result = None
        if action == "run" and statement.strip():
            result = self.reasoner.solve(statement)
            flash = "CP analysis finished. Inspect the chosen algorithm, validator result, and similar episodes below."
            flash_tone = "success"

        if action == "ingest_incident" and statement.strip():
            case = CpIncidentCase(
                statement=statement,
                problem_id=incident_problem_id,
                editorial_summary=incident_editorial,
                outcome_label=incident_outcome,
                failure_kind=incident_failure,
                code=incident_code,
                source_kind="gui_incident",
            )
            record_id = self.ingestor.ingest_case(case, solve_missing=solve_missing)
            result = self.reasoner.solve(statement)
            flash = f"Stored incident episode #{record_id} in the episode DB."
            flash_tone = "success"

        recent = []
        count = 0
        if self.episode_store:
            store = CpEpisodeStore(self.episode_store)
            recent = store.fetch_recent(limit=8)
            count = store.count()
        return render_page(
            statement=statement,
            selected_example=selected_example,
            selected_incident=selected_incident,
            result=result,
            examples=self.examples,
            incidents=self.incidents,
            recent=recent,
            count=count,
            episode_store=self.episode_store,
            flash=flash,
            flash_tone=flash_tone,
            incident_problem_id=incident_problem_id,
            incident_editorial=incident_editorial,
            incident_outcome=incident_outcome,
            incident_failure=incident_failure,
            incident_code=incident_code,
            solve_missing=solve_missing,
        )


def _select_option(value: str, current: str) -> str:
    selected = " selected" if value == current else ""
    return f'<option value="{html.escape(value)}"{selected}>{html.escape(value)}</option>'


def render_page(*, statement: str, selected_example: str, selected_incident: str, result, examples: list[dict], incidents: list[dict], recent: list, count: int, episode_store: str | None, flash: str, flash_tone: str, incident_problem_id: str, incident_editorial: str, incident_outcome: str, incident_failure: str, incident_code: str, solve_missing: bool) -> str:
    flash_html = f'<div class="flash {html.escape(flash_tone)}">{html.escape(flash)}</div>' if flash else ""
    example_cards = "".join(
        f'<button class="pick{(" active" if item["id"] == selected_example else "")}" type="submit" name="example_id" value="{html.escape(item["id"])}"><input type="hidden" name="action" value="load_example"><strong>Seed</strong><div>{html.escape(item["statement"])}</div></button>'
        for item in examples
    ) or '<div class="row">No seed examples found.</div>'
    incident_cards = "".join(
        f'<button class="pick{(" active" if item.get("id") == selected_incident else "")}" type="submit" name="incident_id" value="{html.escape(item.get("id", ""))}"><input type="hidden" name="action" value="load_incident"><strong>{html.escape(item.get("outcome", "INCIDENT"))} / {html.escape(item.get("failure_kind", "incident"))}</strong><div>{html.escape(item.get("statement", ""))}</div></button>'
        for item in incidents
    ) or '<div class="row">No incident examples found.</div>'
    recent_rows = "".join(
        f'<div class="row"><strong>{html.escape(CATEGORY_LABELS.get(item.category, item.category))}</strong><div>{html.escape(item.statement)}</div><small>outcome={html.escape(item.outcome_label)} | failure={html.escape(item.failure_kind)}</small></div>'
        for item in recent
    ) or '<div class="row">No episodes stored yet.</div>'
    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SemOp CP Copilot</title>
<style>
:root {{ --bg:#f6f3ee; --ink:#17212b; --muted:#5b6773; --card:#ffffff; --line:#d6dde6; --accent:#0f658f; --ok:#1d6f48; --shadow:0 16px 36px rgba(20,32,44,.1); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#f8f5ef 0%,#eef3f7 100%); color:var(--ink); font-family:"Segoe UI","Malgun Gothic",sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:24px 20px 40px; }}
.hero,.layout,.grid2,.metrics {{ display:grid; gap:18px; }}
.hero {{ grid-template-columns:1.5fr 1fr; margin-bottom:20px; }}
.layout {{ grid-template-columns:1.2fr .95fr; }}
.grid2,.metrics {{ grid-template-columns:1fr 1fr; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); padding:22px; }}
h1 {{ margin:0 0 10px; font-size:34px; letter-spacing:-.03em; }} h2 {{ margin:0 0 12px; font-size:20px; }} h3 {{ margin:0 0 8px; font-size:16px; }}
p,li,small {{ line-height:1.6; color:var(--muted); }}
textarea,input,select {{ width:100%; border-radius:16px; border:1px solid var(--line); padding:12px 14px; font:inherit; }} textarea {{ min-height:136px; resize:vertical; }}
button.primary,button.secondary,.pick {{ border:0; border-radius:18px; padding:12px 14px; font:inherit; cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; font-weight:600; }} button.secondary {{ background:#edf4f8; color:var(--ink); font-weight:600; }}
.pick {{ width:100%; background:#fff; border:1px solid var(--line); text-align:left; }} .pick.active {{ border-color:var(--accent); box-shadow:inset 0 0 0 1px var(--accent); }}
.flash {{ margin-bottom:14px; padding:12px 14px; border-radius:16px; background:#eef5f8; color:var(--accent); }} .flash.success {{ background:#ecf8f0; color:var(--ok); }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; }} .chip {{ background:#eef5f8; color:var(--accent); border-radius:999px; padding:6px 10px; font-size:12px; }}
.metric,.row {{ border:1px solid var(--line); border-radius:16px; padding:12px; background:#fff; }}
pre {{ background:#14202a; color:#f0f6fb; border-radius:18px; padding:16px; white-space:pre-wrap; overflow-x:auto; }}
.inline {{ display:flex; align-items:center; gap:8px; }}
@media (max-width:1060px) {{ .hero,.layout,.grid2,.metrics {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
<section class="hero">
  <div class="card">
    <h1>SemOp CP Copilot</h1>
    <p>A beginner-friendly local GUI for contest analysis. It reads a statement, extracts hidden structure, chooses an algorithm family, validates generated C++17 code, and learns from WA/TLE incidents stored in the episode DB.</p>
    <div class="chips"><div class="chip">statement -> DSL</div><div class="chip">algorithm ranking</div><div class="chip">validator</div><div class="chip">counterexample capture</div><div class="chip">repair loop</div><div class="chip">episode memory</div></div>
  </div>
  <div class="card">
    <h2>Quick Workflow</h2>
    <ol>
      <li>Load a seed statement or paste your own.</li>
      <li>Run analysis and inspect category, complexity, and validation.</li>
      <li>Store real WA/TLE/editorial cases in the episode DB.</li>
      <li>Later build a training bundle and run LoRA offline.</li>
    </ol>
    <small>Episode store: {html.escape(episode_store or 'disabled')}</small>
  </div>
</section>
{flash_html}
<section class="layout">
  <div class="card">
    <h2>1. Solve A Contest Problem</h2>
    <form method="post">
      <div class="grid2">{example_cards}</div>
      <div style="height:12px"></div>
      <textarea name="statement" placeholder="Paste a contest statement or a concise summary...">{html.escape(statement)}</textarea>
      <div style="height:12px"></div>
      <div class="grid2">
        <button class="primary" type="submit" name="action" value="run">Run CP Analysis</button>
        <button class="secondary" type="submit" name="action" value="load_example">Reload Selected Example</button>
      </div>
    </form>
    <div style="height:18px"></div>
    {render_result(result)}
  </div>
  <div>
    <div class="card" style="margin-bottom:18px;">
      <h2>2. Ingest WA/TLE Or Editorial Incidents</h2>
      <form method="post">
        <div class="grid2">{incident_cards}</div>
        <div style="height:12px"></div>
        <textarea name="statement" placeholder="Problem statement for the incident...">{html.escape(statement)}</textarea>
        <div style="height:10px"></div>
        <input name="incident_problem_id" placeholder="problem id or slug" value="{html.escape(incident_problem_id)}">
        <div style="height:10px"></div>
        <textarea name="incident_editorial" placeholder="editorial summary or intended solution">{html.escape(incident_editorial)}</textarea>
        <div style="height:10px"></div>
        <div class="grid2">
          <select name="incident_outcome"><option value="">verdict</option>{_select_option('AC', incident_outcome)}{_select_option('WA', incident_outcome)}{_select_option('TLE', incident_outcome)}{_select_option('RE', incident_outcome)}</select>
          <select name="incident_failure"><option value="">failure kind</option>{_select_option('output_mismatch', incident_failure)}{_select_option('time_limit', incident_failure)}{_select_option('runtime_error', incident_failure)}{_select_option('compile_error', incident_failure)}</select>
        </div>
        <div style="height:10px"></div>
        <textarea name="incident_code" placeholder="optional C++ submission">{html.escape(incident_code)}</textarea>
        <div style="height:10px"></div>
        <label class="inline"><input type="checkbox" name="solve_missing" value="1" {'checked' if solve_missing else ''}> solve with SemOp first when no code is provided</label>
        <div style="height:12px"></div>
        <div class="grid2">
          <button class="primary" type="submit" name="action" value="ingest_incident">Store Incident Episode</button>
          <button class="secondary" type="submit" name="action" value="load_incident">Reload Selected Incident</button>
        </div>
      </form>
    </div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Episode Memory</h2>
      <div class="metrics"><div class="metric"><strong>Total Episodes</strong><div>{count}</div></div><div class="metric"><strong>DB Path</strong><div>{html.escape(episode_store or 'disabled')}</div></div></div>
      <div style="height:12px"></div>
      {recent_rows}
    </div>
    <div class="card">
      <h2>Train Later</h2>
      <p>Use the GUI to accumulate statements, editorials, and failure cases first. Heavy training stays offline.</p>
      <ol>
        <li>Ingest incidents into <code>data\\cp_episodes.db</code>.</li>
        <li>Build a merged bundle with <code>build_cp_training_bundle.py</code>.</li>
        <li>Run <code>train_cp_parser.py --dry-run</code>.</li>
        <li>Then run actual LoRA training on your RTX 4060 8GB setup.</li>
      </ol>
    </div>
  </div>
</section>
</main>
</body>
</html>
"""


def render_result(result) -> str:
    if result is None:
        return '<div class="row"><h3>Waiting For Input</h3><p>Run a statement to see extracted frames, hidden concepts, validator status, and generated C++17 code.</p></div>'
    validation = result.validation_report or {}
    similar = result.memory_projection.get("episodic", {}).get("similar_episodes", []) if result.memory_projection else []
    similar_rows = "".join(
        f'<div class="row"><strong>{html.escape(item.get("category", "unknown"))}</strong><div>{html.escape(item.get("statement", ""))}</div><small>similarity={html.escape(str(item.get("similarity", 0.0)))} | outcome={html.escape(item.get("outcome_label", ""))} | failure={html.escape(item.get("failure_kind", ""))}</small></div>'
        for item in similar[:5]
    ) or '<div class="row">No similar episodes were found yet.</div>'
    repair_rows = "".join(
        f'<div class="row"><strong>{html.escape(item.get("rule_id", "repair"))}</strong><div>{html.escape(str(item.get("reason", "")))}</div><small>applied={html.escape(str(item.get("applied", False)))}</small></div>'
        for item in result.repair_attempts
    ) or '<div class="row">No repair attempts were recorded.</div>'
    structure_chips = "".join(f'<div class="chip">{html.escape(item)}</div>' for item in (result.hidden_concepts + result.logical_frames + result.dsl_operators)[:18]) or '<div class="row">No structure extracted.</div>'
    reasoning_rows = "".join(f'<div class="row">{html.escape(step)}</div>' for step in result.reasoning_steps) or '<div class="row">No reasoning steps recorded.</div>'
    return f'''
<div class="metrics">
  <div class="metric"><strong>Category</strong><div>{html.escape(CATEGORY_LABELS.get(result.category, result.category))}</div></div>
  <div class="metric"><strong>Confidence</strong><div>{html.escape(f"{result.confidence:.2f}")}</div></div>
  <div class="metric"><strong>Time Complexity</strong><div>{html.escape(result.time_complexity)}</div></div>
  <div class="metric"><strong>Memory Complexity</strong><div>{html.escape(result.memory_complexity)}</div></div>
  <div class="metric"><strong>Compile</strong><div>{'ok' if result.compile_ok else 'failed'}</div></div>
  <div class="metric"><strong>Validation</strong><div>{'ok' if validation.get('overall_ok') else 'failed' if validation else 'not available'}</div></div>
</div>
<div style="height:16px"></div>
<div class="grid2"><div><h3>Reasoning Steps</h3>{reasoning_rows}</div><div><h3>Recovered Structure</h3><div class="chips">{structure_chips}</div></div></div>
<div style="height:16px"></div>
<div class="grid2">
  <div>
    <h3>Validation Details</h3>
    <div class="row"><strong>checker kind</strong><div>{html.escape(str(validation.get('checker_kind', '')))}</div></div>
    <div class="row"><strong>failure type</strong><div>{html.escape(FAILURE_LABELS.get(str(validation.get('failure_type', '')), str(validation.get('failure_type', ''))))}</div></div>
    <div class="row"><strong>sample / random / brute force</strong><div>{html.escape(str(validation.get('sample_cases_run', 0)))} / {html.escape(str(validation.get('random_cases_run', 0)))} / {html.escape(str(validation.get('brute_force_cases_run', 0)))}</div></div>
    <div class="row"><strong>counterexample input</strong><div>{html.escape(str(validation.get('counterexample_input', ''))[:500]) or 'none'}</div></div>
    <div class="row"><strong>expected vs actual</strong><div>{html.escape(str(validation.get('expected_output', ''))[:120])} | {html.escape(str(validation.get('actual_output', ''))[:120])}</div></div>
  </div>
  <div>
    <h3>Repair Loop</h3>
    {repair_rows}
  </div>
</div>
<div style="height:16px"></div>
<h3>Similar Episodes</h3>
{similar_rows}
<div style="height:16px"></div>
<h3>Approach</h3>
<div class="row">{html.escape(result.approach)}</div>
<div style="height:16px"></div>
<h3>C++17 Code</h3>
<pre>{html.escape(result.cpp_code)}</pre>
'''


class CpGuiHandler(BaseHTTPRequestHandler):
    app: CpGuiApp

    def do_GET(self) -> None:
        self._write(self.app.handle({}))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        self._write(self.app.handle(parse_qs(body, keep_blank_values=True)))

    def log_message(self, format: str, *args) -> None:
        return

    def _write(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Beginner-friendly CP GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--episode-store", default="data/cp_episodes.db")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    CpGuiHandler.app = CpGuiApp(episode_store=args.episode_store, compiler=args.compiler)
    server = ThreadingHTTPServer((args.host, args.port), CpGuiHandler)
    print(f"SemOp CP GUI is running at http://{args.host}:{args.port}")
    print(f"Episode store: {args.episode_store}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
