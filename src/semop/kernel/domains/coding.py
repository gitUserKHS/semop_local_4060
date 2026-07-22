from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from ...contest_programmer import CompetitiveProgrammingReasoner
from ..contracts import DomainInstance
from ..grounding import (
    GroundingAuthority,
    GroundingDisposition,
    GroundingTrace,
    grounding_payload_digest,
    make_grounding_record,
)
from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Goal,
    OperatorFamily,
    Rule,
    SolveResult,
    WorldState,
)
from ..registry import KernelRegistry

if TYPE_CHECKING:
    from ...contest_programmer import ContestSolution


@dataclass(frozen=True)
class CodingProblem:
    """A small, explicit boundary for code-generation requests."""

    statement: str
    language: str = "cpp"

    def __post_init__(self) -> None:
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("coding problem statement cannot be empty")
        language = str(self.language).strip().lower()
        if language not in {"cpp", "c++"}:
            raise ValueError("coding v1 currently supports only C++")
        object.__setattr__(self, "statement", self.statement.strip())
        object.__setattr__(self, "language", "cpp")


class CodingInputAdapter:
    """Ground generated C++ and external compiler/test evidence into typed facts.

    The legacy generator remains outside the trusted kernel. Only its concrete
    artifact and independently measured compiler/category-test outcomes enter the
    world state. A successful typed goal therefore means "compiled and passed the
    registered category validator", not acceptance by an unseen online judge.
    """

    def __init__(
        self,
        reasoner: CompetitiveProgrammingReasoner | None = None,
    ) -> None:
        self._reasoner = reasoner

    @property
    def reasoner(self) -> CompetitiveProgrammingReasoner:
        if self._reasoner is None:
            self._reasoner = CompetitiveProgrammingReasoner()
        return self._reasoner

    def adapt(self, value: str | CodingProblem) -> DomainInstance:
        problem = value if isinstance(value, CodingProblem) else CodingProblem(value)
        solution = self.reasoner.solve(problem.statement)
        if solution is None:
            raise ValueError("coding reasoner did not produce a candidate")

        registry = _create_coding_registry()
        input_digest = grounding_payload_digest(
            f"{problem.language}\0{problem.statement}"
        )
        code_digest = sha256(solution.cpp_code.encode("utf-8")).hexdigest()
        problem_symbol = registry.symbol(f"problem_{input_digest[:16]}", "CodingProblem")
        program_symbol = registry.symbol(f"program_{code_digest[:16]}", "Program")
        algorithm_symbol = registry.symbol(solution.category, "Algorithm")

        request = registry.atom("CODING_REQUEST", problem_symbol)
        candidate = registry.atom("CANDIDATE_PROGRAM", problem_symbol, program_symbol)
        implements = registry.atom("IMPLEMENTS", program_symbol, algorithm_symbol)
        compiles = registry.atom("COMPILES", program_symbol)
        tests_passed = registry.atom("TESTS_PASSED", program_symbol)
        verified = registry.atom("VERIFIED_SOLUTION", problem_symbol, program_symbol)

        records = [
            _adapter_record(
                request,
                statement=problem.statement,
                input_digest=input_digest,
                assertion_status=AssertionStatus.EXPLICIT,
                rationale="the deterministic adapter parsed an explicit coding request",
            ),
            _adapter_record(
                candidate,
                statement=f"generated C++ artifact {code_digest}",
                input_digest=input_digest,
                assertion_status=AssertionStatus.GENERATED,
                rationale="the code generator emitted the recorded immutable artifact",
                evidence=(f"sha256:{code_digest}",),
            ),
            _adapter_record(
                implements,
                statement=f"template family {solution.category}",
                input_digest=input_digest,
                assertion_status=AssertionStatus.GENERATED,
                rationale="the selected template has a registered algorithm-family identity",
                evidence=(f"category:{solution.category}",),
            ),
            _external_record(
                compiles,
                accepted=bool(solution.compile_ok),
                statement="C++ compiler syntax/build check",
                input_digest=input_digest,
                producer_id="cpp_compiler",
                rationale=(
                    "the configured C++ compiler accepted the generated artifact"
                    if solution.compile_ok
                    else "the configured C++ compiler rejected the generated artifact"
                ),
                evidence=(solution.compile_command or "compiler command unavailable",),
            ),
        ]

        validation = solution.validation_report
        checked = validation.get("checked") is True
        overall_ok = checked and validation.get("overall_ok") is True
        records.append(
            _external_record(
                tests_passed,
                accepted=overall_ok,
                statement="registered category sample and randomized tests",
                input_digest=input_digest,
                producer_id="cp_category_validator",
                rationale=(
                    "the registered category validator passed all executed checks"
                    if overall_ok
                    else (
                        "the registered category validator found a failing check"
                        if checked
                        else "no executable validator is registered for this category"
                    )
                ),
                evidence=_validation_evidence(validation),
                authority=(
                    GroundingAuthority.EXTERNAL_VERIFIER
                    if checked
                    else GroundingAuthority.DETERMINISTIC_ADAPTER
                ),
            )
        )

        registry.register_guard(
            "verify_recorded_compiler_and_tests",
            _recorded_verification_guard(
                solution,
                expected_code_digest=code_digest,
            ),
        )
        registry.register_operator(
            Rule(
                name="accept_compiled_tested_program",
                parameters=(),
                preconditions=(
                    request,
                    candidate,
                    implements,
                    compiles,
                    tests_passed,
                ),
                effects=(verified,),
                guards=("verify_recorded_compiler_and_tests",),
                description_ko=(
                    "생성한 C++ 프로그램이 컴파일되고 등록된 알고리즘 계열 테스트를 "
                    "모두 통과했음을 확인했다."
                ),
            ),
            family=OperatorFamily.VERIFY.value,
            tags=("coding", "compiler", "execution", solution.category),
        )

        trace = GroundingTrace(tuple(records))
        facts = tuple(record.fact for record in records if record.fact is not None)
        return DomainInstance(
            registry=registry,
            state=WorldState(facts),
            goals=(
                Goal(
                    verified,
                    label=(
                        f"{solution.category} C++ candidate passes compiler and "
                        "registered category tests"
                    ),
                ),
            ),
            domain="coding",
            metadata={
                "statement": problem.statement,
                "language": problem.language,
                "category": solution.category,
                "approach": solution.approach,
                "code": solution.cpp_code,
                "code_sha256": code_digest,
                "time_complexity": solution.time_complexity,
                "memory_complexity": solution.memory_complexity,
                "compile_ok": bool(solution.compile_ok),
                "compile_command": solution.compile_command,
                "validation": dict(validation),
                "search_trace": list(solution.search_trace),
                "selection_strategy": solution.selection_strategy,
                "verification_scope": "compiler_and_registered_category_tests",
                "online_judge_acceptance_claimed": False,
                "operator_depth": 1,
                "reviewed_examples": 0,
                "grounding": trace.to_dict(include_records=False),
            },
            grounding_trace=trace,
        )

    @staticmethod
    def project(_value: str | CodingProblem, _result: SolveResult) -> bool:
        return False


def _create_coding_registry() -> KernelRegistry:
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    problem = registry.types.register("CodingProblem", entity)
    program = registry.types.register("Program", entity)
    algorithm = registry.types.register("Algorithm", entity)
    registry.register_predicate("CODING_REQUEST", (problem,))
    registry.register_predicate("CANDIDATE_PROGRAM", (problem, program))
    registry.register_predicate("IMPLEMENTS", (program, algorithm))
    registry.register_predicate("COMPILES", (program,))
    registry.register_predicate("TESTS_PASSED", (program,))
    registry.register_predicate("VERIFIED_SOLUTION", (problem, program))
    return registry


def _adapter_record(
    atom,
    *,
    statement: str,
    input_digest: str,
    assertion_status: AssertionStatus,
    rationale: str,
    evidence: tuple[str, ...] = (),
):
    return make_grounding_record(
        domain="coding",
        statement=statement,
        atom=atom,
        producer_id="competitive_programming_reasoner",
        source="coding_adapter",
        disposition=GroundingDisposition.OBSERVED,
        authority=GroundingAuthority.DETERMINISTIC_ADAPTER,
        assertion_status=assertion_status,
        evidence_status=EvidenceStatus.ADAPTER_VERIFIED,
        rationale=rationale,
        input_digest=input_digest,
        evidence=evidence,
        sensor_features=(("artifact.atom_arity", len(atom.arguments) / 4.0),),
    )


def _external_record(
    atom,
    *,
    accepted: bool,
    statement: str,
    input_digest: str,
    producer_id: str,
    rationale: str,
    evidence: tuple[str, ...],
    authority: GroundingAuthority = GroundingAuthority.EXTERNAL_VERIFIER,
):
    return make_grounding_record(
        domain="coding",
        statement=statement,
        atom=atom,
        producer_id=producer_id,
        source="coding_verifier",
        disposition=(
            GroundingDisposition.OBSERVED
            if accepted
            else GroundingDisposition.REJECTED
        ),
        authority=authority,
        assertion_status=AssertionStatus.MEASURED,
        evidence_status=(
            EvidenceStatus.EXTERNAL_VERIFIED
            if accepted and authority is GroundingAuthority.EXTERNAL_VERIFIER
            else (
                EvidenceStatus.ADAPTER_VERIFIED
                if accepted
                else EvidenceStatus.UNVERIFIED
            )
        ),
        rationale=rationale,
        input_digest=input_digest,
        evidence=evidence,
        evidence_refs=evidence,
        sensor_features=(("artifact.atom_arity", len(atom.arguments) / 4.0),),
    )


def _validation_evidence(validation: dict[str, object]) -> tuple[str, ...]:
    return (
        f"checker:{validation.get('checker_kind', '')}",
        f"samples:{validation.get('sample_cases_run', 0)}",
        f"random:{validation.get('random_cases_run', 0)}",
        f"failure:{validation.get('failure_type', '')}",
    )


def _recorded_verification_guard(
    solution: ContestSolution,
    *,
    expected_code_digest: str,
):
    def guard(_bindings, _state) -> bool:
        validation = solution.validation_report
        actual_digest = sha256(solution.cpp_code.encode("utf-8")).hexdigest()
        return bool(
            actual_digest == expected_code_digest
            and solution.compile_ok
            and validation.get("checked") is True
            and validation.get("overall_ok") is True
        )

    return guard


__all__ = ["CodingInputAdapter", "CodingProblem"]
