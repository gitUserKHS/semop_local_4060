from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import uuid


@dataclass
class CppCompileResult:
    ok: bool
    command: str
    returncode: int
    stderr: str


class CppSyntaxChecker:
    def __init__(self, compiler: str = "g++", standard: str = "c++17", workspace_tmp: str | Path = "data/tmp/cpp_syntax_checks") -> None:
        self.compiler = compiler
        self.standard = standard
        self.workspace_tmp = Path(workspace_tmp)

    def check(self, code: str) -> CppCompileResult:
        self.workspace_tmp.mkdir(parents=True, exist_ok=True)
        source = self.workspace_tmp / f"main_{uuid.uuid4().hex}.cpp"
        try:
            source.write_text(code, encoding="utf-8")
            command = [self.compiler, f"-std={self.standard}", "-O2", "-Wall", "-Wextra", "-fsyntax-only", str(source)]
            process = subprocess.run(command, capture_output=True, text=True)
            return CppCompileResult(
                ok=process.returncode == 0,
                command=" ".join(command),
                returncode=process.returncode,
                stderr=(process.stderr or "").strip(),
            )
        finally:
            source.unlink(missing_ok=True)
