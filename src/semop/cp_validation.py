from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import subprocess
import uuid
from typing import List


@dataclass
class CpSampleCase:
    name: str
    input_data: str
    expected_output: str


@dataclass
class CppBuildResult:
    ok: bool
    command: str
    returncode: int
    stderr: str
    exe_path: str = ""
    source_path: str = ""


@dataclass
class CppRunResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass
class CpValidationReport:
    checked: bool
    build_ok: bool
    sample_ok: bool | None
    random_ok: bool | None
    overall_ok: bool
    sample_cases_run: int = 0
    random_cases_run: int = 0
    brute_force_cases_run: int = 0
    checker_kind: str = ""
    failure_type: str = ""
    counterexample_input: str = ""
    expected_output: str = ""
    actual_output: str = ""
    notes: List[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "checked": self.checked,
            "build_ok": self.build_ok,
            "sample_ok": self.sample_ok,
            "random_ok": self.random_ok,
            "overall_ok": self.overall_ok,
            "sample_cases_run": self.sample_cases_run,
            "random_cases_run": self.random_cases_run,
            "brute_force_cases_run": self.brute_force_cases_run,
            "checker_kind": self.checker_kind,
            "failure_type": self.failure_type,
            "counterexample_input": self.counterexample_input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "notes": self.notes,
        }


class CppProgramRunner:
    def __init__(self, compiler: str = "g++", standard: str = "c++17", workspace_tmp: str | Path = "data/tmp/cpp_validation") -> None:
        self.compiler = compiler
        self.standard = standard
        self.workspace_tmp = Path(workspace_tmp)

    def build(self, code: str) -> CppBuildResult:
        self.workspace_tmp.mkdir(parents=True, exist_ok=True)
        stem = f"cp_build_{uuid.uuid4().hex}"
        source = self.workspace_tmp / f"{stem}.cpp"
        exe = self.workspace_tmp / f"{stem}.exe"
        source.write_text(code, encoding="utf-8")
        command = [self.compiler, f"-std={self.standard}", "-O2", str(source), "-o", str(exe)]
        process = subprocess.run(command, capture_output=True, text=True)
        return CppBuildResult(
            ok=process.returncode == 0 and exe.exists(),
            command=" ".join(command),
            returncode=process.returncode,
            stderr=(process.stderr or "").strip(),
            exe_path=str(exe),
            source_path=str(source),
        )

    @staticmethod
    def run(exe_path: str, input_data: str, timeout_sec: float = 2.0) -> CppRunResult:
        try:
            process = subprocess.run(
                [exe_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return CppRunResult(
                ok=process.returncode == 0,
                stdout=process.stdout,
                stderr=process.stderr,
                returncode=process.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CppRunResult(
                ok=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                returncode=-1,
                timed_out=True,
            )

    @staticmethod
    def cleanup(build_result: CppBuildResult) -> None:
        if build_result.source_path:
            Path(build_result.source_path).unlink(missing_ok=True)
        if build_result.exe_path:
            Path(build_result.exe_path).unlink(missing_ok=True)


class CpSolutionValidator:
    SUPPORTED_CATEGORIES = {
        "prefix_sum_range_query",
        "fenwick_tree",
        "segment_tree",
        "lazy_segment_tree",
        "dijkstra_shortest_path",
        "dsu_connectivity",
        "grid_bfs",
        "binary_search_answer",
        "knapsack_dp",
    }

    def __init__(self, compiler: str = "g++", standard: str = "c++17") -> None:
        self.runner = CppProgramRunner(compiler=compiler, standard=standard)

    def validate(self, code: str, category: str) -> CpValidationReport:
        sample_cases = self._sample_cases_for(category)
        checker_kind = self._checker_kind(category)
        if not sample_cases and category not in self.SUPPORTED_CATEGORIES:
            return CpValidationReport(
                checked=False,
                build_ok=False,
                sample_ok=None,
                random_ok=None,
                overall_ok=False,
                checker_kind=checker_kind,
                notes=["No validator is registered for this algorithm family yet."],
            )

        build = self.runner.build(code)
        notes: List[str] = []
        failure_type = ""
        counterexample_input = ""
        expected_output = ""
        actual_output = ""
        try:
            if not build.ok:
                notes.append(build.stderr or "Build failed.")
                return CpValidationReport(
                    checked=True,
                    build_ok=False,
                    sample_ok=False if sample_cases else None,
                    random_ok=False if category in self.SUPPORTED_CATEGORIES else None,
                    overall_ok=False,
                    sample_cases_run=0,
                    random_cases_run=0,
                    checker_kind=checker_kind,
                    failure_type="compile_error",
                    notes=notes,
                )

            sample_ok = True
            for case in sample_cases:
                run = self.runner.run(build.exe_path, case.input_data)
                if not run.ok:
                    sample_ok = False
                    failure_type = "time_limit" if run.timed_out else "runtime_error"
                    counterexample_input = case.input_data
                    expected_output = case.expected_output
                    actual_output = run.stdout or run.stderr
                    notes.append(f"Sample `{case.name}` runtime failure.")
                    break
                if not self._compare_outputs(category, run.stdout, case.expected_output):
                    sample_ok = False
                    failure_type = "output_mismatch"
                    counterexample_input = case.input_data
                    expected_output = case.expected_output
                    actual_output = run.stdout
                    notes.append(f"Sample `{case.name}` output mismatch.")
                    break

            random_ok: bool | None = None
            random_cases_run = 0
            if sample_ok:
                validators = {
                    "prefix_sum_range_query": self._validate_prefix_sum_random,
                    "fenwick_tree": self._validate_fenwick_random,
                    "segment_tree": self._validate_segment_tree_random,
                    "lazy_segment_tree": self._validate_lazy_segment_tree_random,
                    "dijkstra_shortest_path": self._validate_dijkstra_random,
                    "dsu_connectivity": self._validate_dsu_random,
                    "grid_bfs": self._validate_grid_bfs_random,
                    "binary_search_answer": self._validate_binary_search_random,
                    "knapsack_dp": self._validate_knapsack_random,
                }
                validator = validators.get(category)
                if validator is not None:
                    random_ok, random_cases_run, random_notes = validator(build.exe_path)
                    notes.extend(random_notes)
                    for note in random_notes:
                        if note.startswith('COUNTEREXAMPLE:') and not failure_type:
                            payload = json.loads(note.split(':', 1)[1])
                            failure_type = payload.get('failure_type', 'output_mismatch')
                            counterexample_input = payload.get('input_data', '')
                            expected_output = payload.get('expected_output', '')
                            actual_output = payload.get('actual_output', '')

            overall_ok = build.ok and sample_ok and (random_ok is not False)
            return CpValidationReport(
                checked=True,
                build_ok=build.ok,
                sample_ok=sample_ok,
                random_ok=random_ok,
                overall_ok=overall_ok,
                sample_cases_run=len(sample_cases),
                random_cases_run=random_cases_run,
                brute_force_cases_run=random_cases_run,
                checker_kind=checker_kind,
                failure_type=failure_type,
                counterexample_input=counterexample_input,
                expected_output=expected_output,
                actual_output=actual_output,
                notes=notes,
            )
        finally:
            self.runner.cleanup(build)

    @staticmethod
    def _normalize(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()

    @staticmethod
    def _checker_kind(category: str) -> str:
        mapping = {
            "prefix_sum_range_query": "linewise_ints",
            "fenwick_tree": "linewise_ints",
            "segment_tree": "linewise_ints",
            "lazy_segment_tree": "linewise_ints",
            "dijkstra_shortest_path": "linewise_ints",
            "dsu_connectivity": "boolean_lines",
            "grid_bfs": "single_integer",
            "binary_search_answer": "single_integer",
            "knapsack_dp": "single_integer",
        }
        return mapping.get(category, "normalized_text")

    def _compare_outputs(self, category: str, actual: str, expected: str) -> bool:
        checker_kind = self._checker_kind(category)
        actual_norm = self._normalize(actual)
        expected_norm = self._normalize(expected)
        if checker_kind in {"single_integer", "linewise_ints"}:
            try:
                actual_lines = [int(item) for item in actual_norm.splitlines() if item.strip()]
                expected_lines = [int(item) for item in expected_norm.splitlines() if item.strip()]
            except ValueError:
                return False
            return actual_lines == expected_lines
        if checker_kind == "boolean_lines":
            normalize_bool = lambda lines: [item.strip().upper() for item in lines.splitlines() if item.strip()]
            return normalize_bool(actual_norm) == normalize_bool(expected_norm)
        return actual_norm == expected_norm

    @staticmethod
    def _counterexample_note(failure_type: str, input_data: str, expected_output: str, actual_output: str) -> str:
        return "COUNTEREXAMPLE:" + json.dumps(
            {
                "failure_type": failure_type,
                "input_data": input_data,
                "expected_output": expected_output,
                "actual_output": actual_output,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _sample_cases_for(category: str) -> List[CpSampleCase]:
        if category == "prefix_sum_range_query":
            return [
                CpSampleCase(
                    name="basic_prefix_sum",
                    input_data="5 3\n1 2 3 4 5\n1 3\n2 5\n4 4\n",
                    expected_output="6\n14\n4\n",
                )
            ]
        if category == "fenwick_tree":
            return [
                CpSampleCase(
                    name="basic_fenwick",
                    input_data="5 5\n1 2 3 4 5\n2 1 3\n1 2 5\n2 2 5\n1 5 -2\n2 4 5\n",
                    expected_output="6\n19\n7\n",
                )
            ]
        if category == "segment_tree":
            return [
                CpSampleCase(
                    name="basic_segment_tree",
                    input_data="5 5\n5 2 7 1 4\n2 1 5\n1 4 6\n2 3 5\n1 2 8\n2 1 3\n",
                    expected_output="1\n4\n5\n",
                )
            ]
        if category == "lazy_segment_tree":
            return [
                CpSampleCase(
                    name="basic_lazy_segment_tree",
                    input_data="5 5\n1 2 3 4 5\n2 1 5\n1 2 4 3\n2 2 5\n1 1 5 -1\n2 1 3\n",
                    expected_output="15\n23\n9\n",
                )
            ]
        if category == "dijkstra_shortest_path":
            return [
                CpSampleCase(
                    name="basic_dijkstra",
                    input_data="4 4 1\n1 2 5\n1 3 2\n3 2 1\n2 4 3\n",
                    expected_output="0\n3\n2\n6\n",
                )
            ]
        if category == "dsu_connectivity":
            return [
                CpSampleCase(
                    name="basic_dsu",
                    input_data="5 6\n2 1 2\n1 1 2\n2 1 2\n1 2 3\n2 1 3\n2 4 5\n",
                    expected_output="NO\nYES\nYES\nNO\n",
                )
            ]
        if category == "grid_bfs":
            return [
                CpSampleCase(
                    name="basic_grid_bfs",
                    input_data="3 4\n....\n.##.\n....\n1 1 3 4\n",
                    expected_output="5\n",
                )
            ]
        if category == "binary_search_answer":
            return [
                CpSampleCase(
                    name="basic_binary_search_answer",
                    input_data="5 2\n7 2 5 10 8\n",
                    expected_output="18\n",
                )
            ]
        if category == "knapsack_dp":
            return [
                CpSampleCase(
                    name="basic_knapsack",
                    input_data="4 7\n6 13\n4 8\n3 6\n5 12\n",
                    expected_output="14\n",
                )
            ]
        return []

    def _validate_prefix_sum_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(200 + seed)
            n = rng.randint(1, 6)
            q = rng.randint(1, 6)
            arr = [rng.randint(-5, 9) for _ in range(n)]
            queries = []
            expected = []
            for _ in range(q):
                l = rng.randint(1, n)
                r = rng.randint(l, n)
                queries.append((l, r))
                expected.append(str(sum(arr[l - 1:r])))
            input_lines = [f"{n} {q}", " ".join(map(str, arr))] + [f"{l} {r}" for l, r in queries]
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random prefix-sum case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", "\n".join(expected) + "\n", run.stdout or run.stderr))
                return False, seed + 1, notes
            if self._normalize(run.stdout) != self._normalize("\n".join(expected) + "\n"):
                notes.append(f"Random prefix-sum case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", "\n".join(expected) + "\n", run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_fenwick_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(700 + seed)
            n = rng.randint(1, 7)
            q = rng.randint(2, 8)
            arr = [rng.randint(-5, 8) for _ in range(n)]
            initial = list(arr)
            expected_lines: List[str] = []
            query_lines: List[str] = []
            for _ in range(q):
                if rng.random() < 0.45:
                    idx = rng.randint(1, n)
                    delta = rng.randint(-4, 6)
                    arr[idx - 1] += delta
                    query_lines.append(f"1 {idx} {delta}")
                else:
                    l = rng.randint(1, n)
                    r = rng.randint(l, n)
                    expected_lines.append(str(sum(arr[l - 1:r])))
                    query_lines.append(f"2 {l} {r}")
            input_lines = [f"{n} {q}", " ".join(map(str, initial))] + query_lines
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random fenwick case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", expected_output, run.stdout or run.stderr))
                return False, seed + 1, notes
            expected_output = "\n".join(expected_lines)
            if expected_output:
                expected_output += "\n"
            if self._normalize(run.stdout) != self._normalize(expected_output):
                notes.append(f"Random fenwick case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", expected_output, run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_segment_tree_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(800 + seed)
            n = rng.randint(1, 7)
            q = rng.randint(2, 8)
            arr = [rng.randint(-5, 10) for _ in range(n)]
            initial = list(arr)
            expected_lines: List[str] = []
            query_lines: List[str] = []
            for _ in range(q):
                if rng.random() < 0.45:
                    idx = rng.randint(1, n)
                    value = rng.randint(-5, 10)
                    arr[idx - 1] = value
                    query_lines.append(f"1 {idx} {value}")
                else:
                    l = rng.randint(1, n)
                    r = rng.randint(l, n)
                    expected_lines.append(str(min(arr[l - 1:r])))
                    query_lines.append(f"2 {l} {r}")
            input_lines = [f"{n} {q}", " ".join(map(str, initial))] + query_lines
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random segment-tree case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", expected_output, run.stdout or run.stderr))
                return False, seed + 1, notes
            expected_output = "\n".join(expected_lines)
            if expected_output:
                expected_output += "\n"
            if self._normalize(run.stdout) != self._normalize(expected_output):
                notes.append(f"Random segment-tree case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", expected_output, run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_lazy_segment_tree_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(1000 + seed)
            n = rng.randint(1, 7)
            q = rng.randint(2, 8)
            arr = [rng.randint(-5, 10) for _ in range(n)]
            initial = list(arr)
            expected_lines: List[str] = []
            query_lines: List[str] = []
            for _ in range(q):
                if rng.random() < 0.5:
                    l = rng.randint(1, n)
                    r = rng.randint(l, n)
                    delta = rng.randint(-3, 6)
                    for idx in range(l - 1, r):
                        arr[idx] += delta
                    query_lines.append(f"1 {l} {r} {delta}")
                else:
                    l = rng.randint(1, n)
                    r = rng.randint(l, n)
                    expected_lines.append(str(sum(arr[l - 1:r])))
                    query_lines.append(f"2 {l} {r}")
            input_lines = [f"{n} {q}", " ".join(map(str, initial))] + query_lines
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random lazy-segtree case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", expected_output, run.stdout or run.stderr))
                return False, seed + 1, notes
            expected_output = "\n".join(expected_lines)
            if expected_output:
                expected_output += "\n"
            if self._normalize(run.stdout) != self._normalize(expected_output):
                notes.append(f"Random lazy-segtree case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", expected_output, run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_dijkstra_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(500 + seed)
            n = rng.randint(2, 6)
            s = 1
            edges = []
            for u in range(1, n + 1):
                for v in range(u + 1, n + 1):
                    if rng.random() < 0.45:
                        edges.append((u, v, rng.randint(1, 9)))
            if not edges:
                edges.append((1, n, rng.randint(1, 9)))
            m = len(edges)
            expected = self._python_dijkstra(n, edges, s)
            input_lines = [f"{n} {m} {s}"] + [f"{u} {v} {w}" for u, v, w in edges]
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random dijkstra case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", "\n".join(map(str, expected)) + "\n", run.stdout or run.stderr))
                return False, seed + 1, notes
            if self._normalize(run.stdout) != self._normalize("\n".join(map(str, expected)) + "\n"):
                notes.append(f"Random dijkstra case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", "\n".join(map(str, expected)) + "\n", run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_dsu_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(900 + seed)
            n = rng.randint(2, 7)
            q = rng.randint(3, 9)
            parent = list(range(n + 1))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def unite(a: int, b: int) -> None:
                ra = find(a)
                rb = find(b)
                if ra != rb:
                    parent[rb] = ra

            query_lines: List[str] = []
            expected_lines: List[str] = []
            for _ in range(q):
                a = rng.randint(1, n)
                b = rng.randint(1, n)
                if rng.random() < 0.55:
                    unite(a, b)
                    query_lines.append(f"1 {a} {b}")
                else:
                    expected_lines.append("YES" if find(a) == find(b) else "NO")
                    query_lines.append(f"2 {a} {b}")
            input_data = f"{n} {q}\n" + "\n".join(query_lines) + "\n"
            run = self.runner.run(exe_path, input_data)
            if not run.ok:
                notes.append(f"Random dsu case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", input_data, expected_output, run.stdout or run.stderr))
                return False, seed + 1, notes
            expected_output = "\n".join(expected_lines)
            if expected_output:
                expected_output += "\n"
            if self._normalize(run.stdout) != self._normalize(expected_output):
                notes.append(f"Random dsu case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", input_data, expected_output, run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_grid_bfs_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(1100 + seed)
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            grid = []
            for _ in range(h):
                row = ["." if rng.random() < 0.75 else "#" for _ in range(w)]
                grid.append(row)
            sx, sy = 0, 0
            tx, ty = h - 1, w - 1
            grid[sx][sy] = "."
            grid[tx][ty] = "."
            expected = self._python_grid_bfs(["".join(row) for row in grid], (sx, sy), (tx, ty))
            input_lines = [f"{h} {w}"] + ["".join(row) for row in grid] + [f"{sx + 1} {sy + 1} {tx + 1} {ty + 1}"]
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random grid bfs case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout or run.stderr))
                return False, seed + 1, notes
            if self._normalize(run.stdout) != self._normalize(f"{expected}\n"):
                notes.append(f"Random grid bfs case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_binary_search_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(1200 + seed)
            n = rng.randint(2, 7)
            k = rng.randint(1, n)
            arr = [rng.randint(1, 12) for _ in range(n)]
            expected = self._python_split_array_min_largest_sum(arr, k)
            input_lines = [f"{n} {k}", " ".join(map(str, arr))]
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random binary-search case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout or run.stderr))
                return False, seed + 1, notes
            if self._normalize(run.stdout) != self._normalize(f"{expected}\n"):
                notes.append(f"Random binary-search case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    def _validate_knapsack_random(self, exe_path: str) -> tuple[bool, int, List[str]]:
        notes: List[str] = []
        cases = 5
        for seed in range(cases):
            rng = random.Random(1300 + seed)
            n = rng.randint(1, 7)
            capacity = rng.randint(3, 14)
            items = [(rng.randint(1, 8), rng.randint(1, 15)) for _ in range(n)]
            expected = self._python_knapsack(items, capacity)
            input_lines = [f"{n} {capacity}"] + [f"{weight} {value}" for weight, value in items]
            run = self.runner.run(exe_path, "\n".join(input_lines) + "\n")
            if not run.ok:
                notes.append(f"Random knapsack case {seed} runtime failure.")
                notes.append(self._counterexample_note("runtime_error", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout or run.stderr))
                return False, seed + 1, notes
            if self._normalize(run.stdout) != self._normalize(f"{expected}\n"):
                notes.append(f"Random knapsack case {seed} output mismatch.")
                notes.append(self._counterexample_note("output_mismatch", "\n".join(input_lines) + "\n", f"{expected}\n", run.stdout))
                return False, seed + 1, notes
        return True, cases, notes

    @staticmethod
    def _python_dijkstra(n: int, edges: List[tuple[int, int, int]], source: int) -> List[int]:
        inf = 10**15
        dist = [inf] * (n + 1)
        dist[source] = 0
        changed = True
        for _ in range(n - 1):
            if not changed:
                break
            changed = False
            for u, v, w in edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    changed = True
                if dist[v] + w < dist[u]:
                    dist[u] = dist[v] + w
                    changed = True
        return [0 if i == source and dist[i] == 0 else (-1 if dist[i] >= inf else int(dist[i])) for i in range(1, n + 1)]

    @staticmethod
    def _python_grid_bfs(grid: List[str], source: tuple[int, int], target: tuple[int, int]) -> int:
        h = len(grid)
        w = len(grid[0]) if h else 0
        dist = [[-1] * w for _ in range(h)]
        queue: List[tuple[int, int]] = [source]
        dist[source[0]][source[1]] = 0
        head = 0
        while head < len(queue):
            x, y = queue[head]
            head += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= h or ny < 0 or ny >= w:
                    continue
                if grid[nx][ny] == "#" or dist[nx][ny] != -1:
                    continue
                dist[nx][ny] = dist[x][y] + 1
                queue.append((nx, ny))
        return dist[target[0]][target[1]]

    @staticmethod
    def _python_split_array_min_largest_sum(arr: List[int], k: int) -> int:
        lo = max(arr)
        hi = sum(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            groups = 1
            current = 0
            for value in arr:
                if current + value > mid:
                    groups += 1
                    current = value
                else:
                    current += value
            if groups <= k:
                hi = mid
            else:
                lo = mid + 1
        return lo

    @staticmethod
    def _python_knapsack(items: List[tuple[int, int]], capacity: int) -> int:
        best = 0
        n = len(items)
        for mask in range(1 << n):
            total_weight = 0
            total_value = 0
            for index in range(n):
                if mask & (1 << index):
                    weight, value = items[index]
                    total_weight += weight
                    total_value += value
            if total_weight <= capacity:
                best = max(best, total_value)
        return best

