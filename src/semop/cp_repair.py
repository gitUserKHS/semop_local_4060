from dataclasses import dataclass
import re
from typing import List, Sequence


@dataclass
class CpRepairAttempt:
    rule_id: str
    reason: str
    applied: bool

    def model_dump(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "reason": self.reason,
            "applied": self.applied,
        }


class CppRepairEngine:
    def repair(
        self,
        category: str,
        code: str,
        compile_stderr: str = "",
        validation_notes: Sequence[str] | None = None,
        failure_type: str = "",
        counterexample_input: str = "",
        fallback_code: str | None = None,
    ) -> tuple[str, List[CpRepairAttempt]]:
        notes = list(validation_notes or [])
        attempts: List[CpRepairAttempt] = []
        updated = code

        repaired_newlines = updated.replace("'\n'", "'\\n'")
        repaired_newlines = repaired_newlines.replace("<< '\n';", "<< '\\n';")
        repaired_newlines = repaired_newlines.replace("<< '\n'", "<< '\\n'")
        if repaired_newlines == updated and "missing terminating" in compile_stderr.lower():
            repaired_newlines = repaired_newlines.replace("'\\n'", '"\\n"')
            repaired_newlines = repaired_newlines.replace("'\n'", '"\\n"')
        applied = repaired_newlines != updated
        attempts.append(
            CpRepairAttempt(
                rule_id="escape_newline_literal",
                reason="Normalize generated newline literals so MinGW C++17 parses them reliably.",
                applied=applied,
            )
        )
        updated = repaired_newlines

        if failure_type == "time_limit":
            faster = self._ensure_fast_io(updated)
            attempts.append(
                CpRepairAttempt(
                    rule_id="ensure_fast_io",
                    reason="Add or restore fast I/O setup when the validator reports a likely time-limit issue.",
                    applied=faster != updated,
                )
            )
            updated = faster

        should_reset = False
        reset_reason = ""
        if failure_type in {"output_mismatch", "runtime_error", "time_limit"} and fallback_code:
            should_reset = True
            reset_reason = f"Use the canonical {category} template when validation reports {failure_type}."
        elif "missing terminating" in compile_stderr.lower() and fallback_code:
            should_reset = True
            reset_reason = f"Use the canonical {category} template after a compile error in generated code."

        attempts.append(
            CpRepairAttempt(
                rule_id="canonical_template_reset",
                reason=reset_reason or "Reset to canonical template if failure signals indicate template corruption.",
                applied=should_reset and fallback_code is not None and fallback_code != updated,
            )
        )
        if should_reset and fallback_code:
            updated = fallback_code

        if counterexample_input and fallback_code and failure_type == "output_mismatch":
            attempts.append(
                CpRepairAttempt(
                    rule_id="counterexample_guided_rewrite",
                    reason="Replace the implementation with the canonical template after a concrete counterexample was found.",
                    applied=fallback_code != updated,
                )
            )
            updated = fallback_code
        else:
            attempts.append(
                CpRepairAttempt(
                    rule_id="counterexample_guided_rewrite",
                    reason="Replace the implementation with the canonical template after a concrete counterexample was found.",
                    applied=False,
                )
            )

        if category == "binary_search_answer" and (failure_type in {"output_mismatch", "time_limit"} or "terminating" in compile_stderr.lower()):
            rewritten = updated.replace("cout << lo << '\n';", "cout << lo << '\\n';")
            attempts.append(
                CpRepairAttempt(
                    rule_id="binary_search_output_fix",
                    reason="Repair the final binary-search output line and preserve the canonical answer template.",
                    applied=rewritten != updated,
                )
            )
            updated = rewritten

        if category == "lazy_segment_tree" and any("output mismatch" in note.lower() for note in notes) and fallback_code:
            attempts.append(
                CpRepairAttempt(
                    rule_id="lazy_segment_tree_reset",
                    reason="Reset to the canonical lazy propagation template after a wrong-answer signal.",
                    applied=fallback_code != updated,
                )
            )
            updated = fallback_code
        else:
            attempts.append(
                CpRepairAttempt(
                    rule_id="lazy_segment_tree_reset",
                    reason="Reset to the canonical lazy propagation template after a wrong-answer signal.",
                    applied=False,
                )
            )

        return updated, attempts

    @staticmethod
    def _ensure_fast_io(code: str) -> str:
        if "ios::sync_with_stdio(false);" in code and "cin.tie(nullptr);" in code:
            return code
        if "int main() {" in code:
            return code.replace(
                "int main() {",
                "int main() {\n    ios::sync_with_stdio(false);\n    cin.tie(nullptr);",
                1,
            )
        return code
