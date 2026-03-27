from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hardware_profiles import detect_local_hardware, detect_local_ml_stack


@dataclass
class RuntimeSurfaceSpec:
    name: str
    script_path: str
    command: list[str]
    url: str = ''
    ready: bool = False
    required: bool = True
    notes: list[str] = field(default_factory=list)
    log_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeDoctorReport:
    workspace: str
    python_executable: str
    hardware_profile: dict[str, Any]
    dependency_status: dict[str, Any]
    surfaces: list[RuntimeSurfaceSpec] = field(default_factory=list)
    install_command: str = ''
    recommended_actions: list[str] = field(default_factory=list)
    beginner_launch_command: str = ''
    one_click_command: str = ''
    double_click_launcher: str = ''
    operator_algebra_mode: str = 'symbolic_cpu_first'

    def model_dump(self) -> dict[str, Any]:
        return {
            'workspace': self.workspace,
            'python_executable': self.python_executable,
            'hardware_profile': dict(self.hardware_profile),
            'dependency_status': dict(self.dependency_status),
            'surfaces': [item.model_dump() for item in self.surfaces],
            'install_command': self.install_command,
            'recommended_actions': list(self.recommended_actions),
            'beginner_launch_command': self.beginner_launch_command,
            'one_click_command': self.one_click_command,
            'double_click_launcher': self.double_click_launcher,
            'operator_algebra_mode': self.operator_algebra_mode,
        }


@dataclass
class RuntimeLaunchEntry:
    name: str
    pid: int | None
    command: list[str]
    url: str = ''
    log_path: str = ''
    started: bool = False
    dry_run: bool = False

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeLaunchSummary:
    manifest_path: str
    runtime_dir: str
    dry_run: bool
    entries: list[RuntimeLaunchEntry] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'manifest_path': self.manifest_path,
            'runtime_dir': self.runtime_dir,
            'dry_run': self.dry_run,
            'entries': [item.model_dump() for item in self.entries],
        }


@dataclass
class BeginnerOneClickSummary:
    workspace: str
    runtime_dir: str
    manifest_path: str
    dry_run: bool
    preferred_url: str = ''
    opened_urls: list[str] = field(default_factory=list)
    browser_opened: bool = False
    doctor_report: dict[str, Any] = field(default_factory=dict)
    launch_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    summary_path: str = ''

    def model_dump(self) -> dict[str, Any]:
        return {
            'workspace': self.workspace,
            'runtime_dir': self.runtime_dir,
            'manifest_path': self.manifest_path,
            'dry_run': self.dry_run,
            'preferred_url': self.preferred_url,
            'opened_urls': list(self.opened_urls),
            'browser_opened': self.browser_opened,
            'doctor_report': dict(self.doctor_report),
            'launch_summary': dict(self.launch_summary),
            'notes': list(self.notes),
            'summary_path': self.summary_path,
        }


def _python_executable(python_executable: str | None = None) -> str:
    if python_executable:
        return python_executable
    venv_python = Path('.venv312') / 'Scripts' / 'python.exe'
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _launch_windows_surface(command: list[str], workspace_path: Path, log_path: str) -> int:
    if not command:
        raise RuntimeError('missing command')
    args = ', '.join(_ps_quote(item) for item in command[1:])
    script = (
        "$p = Start-Process "
        + "-FilePath " + _ps_quote(command[0]) + " "
        + "-ArgumentList @(" + args + ") "
        + "-WorkingDirectory " + _ps_quote(str(workspace_path)) + " "
        + "-WindowStyle Hidden "
        + "-PassThru; $p.Id"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-Command', script],
        cwd=str(workspace_path),
        capture_output=True,
        text=True,
        check=True,
    )
    output = str(result.stdout or '').strip().splitlines()
    pid_text = output[-1].strip() if output else ''
    return int(pid_text)


def _gui_python_executable(python_executable: str) -> str:
    candidate = Path(python_executable)
    if os.name == 'nt' and candidate.name.lower() == 'python.exe':
        gui_candidate = candidate.with_name('pythonw.exe')
        if gui_candidate.exists():
            return str(gui_candidate)
    return python_executable


def _surface_specs(workspace: str, python_executable: str, host: str, include_math_service: bool = True) -> list[RuntimeSurfaceSpec]:
    root = Path(workspace)
    runtime_dir = root / 'data' / 'runtime_ops'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    hardware = detect_local_hardware()
    deps = detect_local_ml_stack(hardware)
    gui_python = _gui_python_executable(python_executable)
    studio = RuntimeSurfaceSpec(
        name='semop_studio',
        script_path=str(root / 'semop_studio_gui.py'),
        command=[gui_python, 'semop_studio_gui.py', '--host', host, '--port', '8780'],
        url=f'http://{host}:8780',
        ready=Path(root / 'semop_studio_gui.py').exists(),
        required=True,
        notes=['Main beginner GUI for operator algebra, unified training, and benchmark gating.', 'Preferred one-click landing page for beginners.'],
        log_path=str(runtime_dir / 'semop_studio.log'),
    )
    math_gui_notes = ['Math and visual world-model GUI for proof solving and image-to-3D workflows.']
    if not deps.llm_ready:
        math_gui_notes.append('Core ML packages are missing, so local LLM-assisted modes will be limited.')
    math_gui = RuntimeSurfaceSpec(
        name='math_world_model_gui',
        script_path=str(root / 'math_world_model_gui.py'),
        command=[gui_python, 'math_world_model_gui.py', '--host', host, '--port', '8791'],
        url=f'http://{host}:8791',
        ready=Path(root / 'math_world_model_gui.py').exists(),
        required=False,
        notes=math_gui_notes,
        log_path=str(runtime_dir / 'math_world_model_gui.log'),
    )
    surfaces = [studio, math_gui]
    if include_math_service:
        service_notes = ['Production-gated math API with /healthz, /readyz, /metrics, /self_test, and /solve.']
        if hardware.low_vram and not deps.qlora_ready:
            service_notes.append('On RTX 4060-class low-VRAM setups, QLoRA is optional; symbolic/operator-algebra-first fallback remains available.')
        surfaces.append(
            RuntimeSurfaceSpec(
                name='math_world_model_service',
                script_path=str(root / 'math_world_model_service.py'),
                command=[python_executable, 'math_world_model_service.py', '--host', host, '--port', '8792'],
                url=f'http://{host}:8792',
                ready=Path(root / 'math_world_model_service.py').exists() and deps.llm_ready,
                required=False,
                notes=service_notes,
                log_path=str(runtime_dir / 'math_world_model_service.log'),
            )
        )
    return surfaces


def build_runtime_doctor_report(workspace: str = '.', python_executable: str | None = None, host: str = '127.0.0.1') -> RuntimeDoctorReport:
    workspace_path = str(Path(workspace).resolve())
    python_exec = _python_executable(python_executable)
    hardware = detect_local_hardware()
    deps = detect_local_ml_stack(hardware)
    surfaces = _surface_specs(workspace_path, python_exec, host)
    install_command = f'"{python_exec}" -m pip install -r requirements.txt'
    launcher_path = str(Path(workspace_path) / 'launch_semop_studio.bat')
    recommended_actions: list[str] = []
    if deps.missing_core:
        recommended_actions.append('Install the core ML packages first: ' + ', '.join(deps.missing_core))
        recommended_actions.append('Run this from the repo root: ' + install_command)
    if not deps.training_ready:
        recommended_actions.append('Install `peft` if you want LoRA fine-tuning from this machine.')
    if hardware.low_vram and not deps.qlora_ready:
        if os.name == 'nt':
            recommended_actions.append('QLoRA is optional on Windows; if bitsandbytes is unavailable, keep using the symbolic-first fp16 fallback on RTX 4060 8GB.')
        else:
            recommended_actions.append('Install `bitsandbytes` and `accelerate` if you want 4-bit QLoRA on the low-VRAM CUDA profile.')
    recommended_actions.append('Fastest one-click path: double-click launch_semop_studio.bat from the repo root.')
    recommended_actions.append('Terminal one-click path: ' + f'"{python_exec}" run_beginner_local_stack.py')
    recommended_actions.append('Start with `semop_studio_gui.py`, then open `math_world_model_gui.py` only when you need proof solving or image-to-3D.')
    beginner_launch_command = f'"{python_exec}" run_beginner_local_stack.py launch'
    one_click_command = f'"{python_exec}" run_beginner_local_stack.py'
    return RuntimeDoctorReport(
        workspace=workspace_path,
        python_executable=python_exec,
        hardware_profile=hardware.model_dump(),
        dependency_status=deps.model_dump(),
        surfaces=surfaces,
        install_command=install_command,
        recommended_actions=list(dict.fromkeys(recommended_actions)),
        beginner_launch_command=beginner_launch_command,
        one_click_command=one_click_command,
        double_click_launcher=launcher_path,
        operator_algebra_mode=hardware.operator_algebra_mode,
    )


def launch_runtime_stack(
    workspace: str = '.',
    python_executable: str | None = None,
    host: str = '127.0.0.1',
    include_math_service: bool = True,
    dry_run: bool = False,
) -> RuntimeLaunchSummary:
    workspace_path = Path(workspace).resolve()
    python_exec = _python_executable(python_executable)
    runtime_dir = workspace_path / 'data' / 'runtime_ops'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    surfaces = _surface_specs(str(workspace_path), python_exec, host, include_math_service=include_math_service)
    entries: list[RuntimeLaunchEntry] = []
    for surface in surfaces:
        entry = RuntimeLaunchEntry(
            name=surface.name,
            pid=None,
            command=list(surface.command),
            url=surface.url,
            log_path=surface.log_path,
            started=False,
            dry_run=dry_run,
        )
        if surface.ready and not dry_run:
            if os.name == 'nt':
                entry.pid = _launch_windows_surface(surface.command, workspace_path, surface.log_path)
                entry.started = True
            else:
                popen_kwargs: dict[str, Any] = {
                    'cwd': str(workspace_path),
                    'stdin': subprocess.DEVNULL,
                    'start_new_session': True,
                }
                with open(surface.log_path, 'a', encoding='utf-8') as log_handle:
                    process = subprocess.Popen(
                        surface.command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        **popen_kwargs,
                    )
                entry.pid = int(process.pid)
                entry.started = True
        entries.append(entry)
    manifest_path = runtime_dir / 'launch_manifest.json'
    payload = {
        'timestamp_ms': int(time.time() * 1000),
        'workspace': str(workspace_path),
        'python_executable': python_exec,
        'dry_run': dry_run,
        'entries': [item.model_dump() for item in entries],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return RuntimeLaunchSummary(
        manifest_path=str(manifest_path),
        runtime_dir=str(runtime_dir),
        dry_run=dry_run,
        entries=entries,
    )


def launch_beginner_one_click(
    workspace: str = '.',
    python_executable: str | None = None,
    host: str = '127.0.0.1',
    include_math_service: bool = True,
    dry_run: bool = False,
    open_browser: bool = True,
    open_all: bool = False,
    wait_timeout_seconds: float = 15.0,
) -> BeginnerOneClickSummary:
    workspace_path = Path(workspace).resolve()
    doctor = build_runtime_doctor_report(workspace=str(workspace_path), python_executable=python_executable, host=host)
    launch = launch_runtime_stack(
        workspace=str(workspace_path),
        python_executable=python_executable,
        host=host,
        include_math_service=include_math_service,
        dry_run=dry_run,
    )
    started_entries = [item for item in launch.entries if item.url and (item.started or dry_run)]
    preferred_entry = next((item for item in started_entries if item.name == 'semop_studio'), started_entries[0] if started_entries else None)
    entries_to_open = started_entries if open_all else ([preferred_entry] if preferred_entry is not None else [])
    opened_urls: list[str] = []
    browser_opened = False
    if not dry_run:
        for index, entry in enumerate(entries_to_open):
            url = str(entry.url or '').strip()
            if not url:
                continue
            _wait_for_http_ready(url, timeout_seconds=wait_timeout_seconds if index == 0 else min(wait_timeout_seconds, 6.0))
            if open_browser:
                try:
                    webbrowser.open(url, new=2, autoraise=True)
                    browser_opened = True
                except Exception:
                    pass
            opened_urls.append(url)
    notes = list(dict.fromkeys(list(doctor.recommended_actions) + [
        ('Browser opening was skipped by request.' if not open_browser else 'The browser is opened automatically for the main studio URL.'),
        ('Dry-run mode wrote the manifest without starting servers.' if dry_run else 'Launch manifest and one-click summary were written under data/runtime_ops.'),
    ]))
    summary = BeginnerOneClickSummary(
        workspace=str(workspace_path),
        runtime_dir=launch.runtime_dir,
        manifest_path=launch.manifest_path,
        dry_run=dry_run,
        preferred_url=str(preferred_entry.url if preferred_entry is not None else ''),
        opened_urls=opened_urls,
        browser_opened=browser_opened,
        doctor_report=doctor.model_dump(),
        launch_summary=launch.model_dump(),
        notes=notes,
        summary_path=str(Path(launch.runtime_dir) / 'one_click_summary.json'),
    )
    Path(summary.summary_path).write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def _wait_for_http_ready(url: str, timeout_seconds: float = 15.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                status = int(getattr(response, 'status', 200) or 200)
                if 200 <= status < 500:
                    return True
        except Exception:
            time.sleep(0.25)
    return False
