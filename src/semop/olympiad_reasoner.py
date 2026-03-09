from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, List

from .structures import SymbolicResult


@dataclass
class ProofStep:
    operator: str
    statement: str
    rationale: str


@dataclass
class PatternMemoryEntry:
    problem_type: str
    trigger_terms: List[str]
    hint_facts: List[str]
    preferred_operators: List[str]
    explanation: str


@dataclass
class ProofState:
    problem_type: str
    normalized_query: str
    goal: str
    facts: List[str] = field(default_factory=list)
    steps: List[ProofStep] = field(default_factory=list)
    score: float = 0.0
    solved: bool = False
    memory_notes: List[str] = field(default_factory=list)
    preferred_operators: List[str] = field(default_factory=list)

    def has_fact(self, needle: str) -> bool:
        lowered = needle.lower()
        return any(lowered == fact.lower() for fact in self.facts)

    def clone(self) -> "ProofState":
        return ProofState(
            problem_type=self.problem_type,
            normalized_query=self.normalized_query,
            goal=self.goal,
            facts=list(self.facts),
            steps=list(self.steps),
            score=self.score,
            solved=self.solved,
            memory_notes=list(self.memory_notes),
            preferred_operators=list(self.preferred_operators),
        )


@dataclass
class OlympiadProblem:
    problem_type: str
    goal: str
    query: str
    triggers: List[str] = field(default_factory=list)
    parameters: dict[str, int | str] = field(default_factory=dict)


@dataclass
class OlympiadSearchResult:
    solved: bool
    final_statement: str
    proof_outline: List[str]
    operator_trace: List[str]
    explored_states: int
    confidence: float
    problem_type: str

    def to_symbolic_result(self) -> SymbolicResult:
        return SymbolicResult(
            domain="olympiad_proof",
            answer=self.final_statement,
            evidence=self.proof_outline,
            equations=self.operator_trace,
            confidence=self.confidence,
            source="olympiad_reasoner",
        )


PATTERN_MEMORY: List[PatternMemoryEntry] = [
    PatternMemoryEntry(
        problem_type="odd_sum_even",
        trigger_terms=["odd", "sum", "even"],
        hint_facts=["rewrite each odd integer as 2k+1", "look for a factor of 2 after expansion"],
        preferred_operators=["PARITY_REPRESENTATION", "ALGEBRAIC_EXPANSION", "DIVISIBILITY_CONCLUSION"],
        explanation="Parity problems are often solved by normal-form rewriting into 2k or 2k+1.",
    ),
    PatternMemoryEntry(
        problem_type="odd_square_odd",
        trigger_terms=["odd", "square"],
        hint_facts=["rewrite the odd integer as 2k+1", "expand and keep the +1 parity tail"],
        preferred_operators=["PARITY_REPRESENTATION", "ALGEBRAIC_EXPANSION", "PARITY_CONCLUSION"],
        explanation="Odd-square proofs usually preserve the 2t+1 shape after expansion.",
    ),
    PatternMemoryEntry(
        problem_type="consecutive_product_even",
        trigger_terms=["consecutive", "product", "even"],
        hint_facts=["parameterize the pair as n and n+1", "one of two consecutive integers is even"],
        preferred_operators=["CONSECUTIVE_REWRITE", "PARITY_CASE_SPLIT", "PRODUCT_EVENNESS"],
        explanation="Consecutive-integer proofs often reduce to an unavoidable parity split.",
    ),
    PatternMemoryEntry(
        problem_type="three_consecutive_sum_mod3",
        trigger_terms=["three consecutive", "divisible by 3"],
        hint_facts=["parameterize the triple as n, n+1, n+2", "factor a 3 out of the sum"],
        preferred_operators=["CONSECUTIVE_REWRITE", "MODULAR_REWRITE", "MODULAR_DIVISIBILITY"],
        explanation="Small modular divisibility problems often collapse after rewriting the sum into a multiple of the modulus.",
    ),
    PatternMemoryEntry(
        problem_type="pigeonhole_shared_bucket",
        trigger_terms=["at least two", "same month", "same bucket"],
        hint_facts=["count objects and categories first", "compare objects with available buckets"],
        preferred_operators=["COUNT_OBJECTS_AND_BUCKETS", "PIGEONHOLE_PRINCIPLE", "PIGEONHOLE_CONCLUSION"],
        explanation="Pigeonhole arguments are count-first proofs: identify objects, boxes, and the overflow point.",
    ),
    PatternMemoryEntry(
        problem_type="infinitely_many_primes",
        trigger_terms=["infinitely many primes", "prime"],
        hint_facts=["try contradiction", "construct product of all listed primes plus one"],
        preferred_operators=["CONTRADICTION_SETUP", "AUXILIARY_CONSTRUCTION", "DIVISIBILITY_CONTRADICTION", "CONTRADICTION_CONCLUSION"],
        explanation="Classic existence proofs often succeed by contradiction plus an auxiliary construction.",
    ),
    PatternMemoryEntry(
        problem_type="mutilated_chessboard",
        trigger_terms=["domino", "chessboard", "opposite corners"],
        hint_facts=["color the board", "check what one domino preserves", "compare the two color counts"],
        preferred_operators=["COLORING_INVARIANT", "CORNER_COLOR_ANALYSIS", "INVARIANT_COUNT_MISMATCH", "INVARIANT_CONCLUSION"],
        explanation="Coloring is a reusable invariant when each move preserves a color balance.",
    ),
    PatternMemoryEntry(
        problem_type="induction_sum_formula",
        trigger_terms=["first n", "sum", "n(n+1)/2"],
        hint_facts=["verify the base case first", "assume the formula for k", "add k+1 and simplify"],
        preferred_operators=["INDUCTION_BASE", "INDUCTION_HYPOTHESIS", "INDUCTION_STEP", "INDUCTION_CONCLUSION"],
        explanation="Formula proofs over n often reduce to the base case plus one symbolic induction step.",
    ),
    PatternMemoryEntry(
        problem_type="extremal_average_bound",
        trigger_terms=["finite set", "average", "at most the average"],
        hint_facts=["pick an extremal element", "compare every element with the average", "use the sum contradiction if needed"],
        preferred_operators=["EXTREMAL_CHOICE", "AVERAGE_BOUND", "SUM_CONTRADICTION", "EXTREMAL_CONCLUSION"],
        explanation="Extremal arguments often start by choosing a maximal element and comparing it against a global average bound.",
    ),
]


class OlympiadReasoner:
    def solve(self, query: str) -> SymbolicResult | None:
        problem = self._parse_problem(query)
        if problem is None:
            return None
        result = self._search(problem)
        return result.to_symbolic_result()

    def _parse_problem(self, query: str) -> OlympiadProblem | None:
        normalized = self._normalize(query)
        if not self._looks_like_proof_problem(normalized):
            return None

        if self._is_odd_sum_problem(normalized):
            return OlympiadProblem(
                problem_type="odd_sum_even",
                goal="the sum of two odd integers is even",
                query=query,
                triggers=["odd", "even", "sum"],
            )
        if self._is_odd_square_problem(normalized):
            return OlympiadProblem(
                problem_type="odd_square_odd",
                goal="the square of an odd integer is odd",
                query=query,
                triggers=["odd", "square"],
            )
        if self._is_consecutive_product_problem(normalized):
            return OlympiadProblem(
                problem_type="consecutive_product_even",
                goal="the product of two consecutive integers is even",
                query=query,
                triggers=["consecutive", "product", "even"],
            )
        if self._is_three_consecutive_sum_problem(normalized):
            return OlympiadProblem(
                problem_type="three_consecutive_sum_mod3",
                goal="the sum of three consecutive integers is divisible by 3",
                query=query,
                triggers=["consecutive", "divisible by 3", "sum"],
            )
        if self._is_infinite_primes_problem(normalized):
            return OlympiadProblem(
                problem_type="infinitely_many_primes",
                goal="there are infinitely many primes",
                query=query,
                triggers=["infinitely many primes", "prime"],
            )
        if self._is_induction_sum_problem(normalized):
            return OlympiadProblem(
                problem_type="induction_sum_formula",
                goal="1+2+...+n = n(n+1)/2",
                query=query,
                triggers=["first n", "sum", "induction"],
            )
        if self._is_extremal_average_problem(normalized):
            return OlympiadProblem(
                problem_type="extremal_average_bound",
                goal="some element is at most the average",
                query=query,
                triggers=["finite set", "average", "extremal"],
            )
        if self._is_domino_coloring_problem(normalized):
            return OlympiadProblem(
                problem_type="mutilated_chessboard",
                goal="the mutilated chessboard cannot be tiled by dominoes",
                query=query,
                triggers=["chessboard", "domino", "opposite corners"],
                parameters={"white_squares": 30, "black_squares": 32},
            )

        pigeonhole = self._parse_pigeonhole_problem(normalized)
        if pigeonhole is not None:
            return pigeonhole
        return self._parse_generic_problem(query, normalized)

    def _search(self, problem: OlympiadProblem) -> OlympiadSearchResult:
        memory_entries = self._memory_entries(problem)
        initial = ProofState(
            problem_type=problem.problem_type,
            normalized_query=self._normalize(problem.query),
            goal=problem.goal,
            facts=[f"goal: {problem.goal}"],
            score=0.1,
            memory_notes=[entry.explanation for entry in memory_entries],
            preferred_operators=self._preferred_operators(memory_entries),
        )
        frontier: List[ProofState] = [initial]
        explored = 0
        visited: set[tuple[str, tuple[str, ...], int]] = set()
        best = initial

        for _depth in range(6):
            next_frontier: List[ProofState] = []
            for state in frontier:
                explored += 1
                if state.score > best.score or len(state.steps) > len(best.steps):
                    best = state
                key = (state.problem_type, tuple(state.facts), len(state.steps))
                if key in visited:
                    continue
                visited.add(key)
                if state.solved:
                    return self._result_from_state(state, explored)
                for candidate in self._apply_operators(state, problem):
                    candidate.score = self._state_priority(candidate, problem, memory_entries)
                    if candidate.score > best.score or len(candidate.steps) > len(best.steps):
                        best = candidate
                    if candidate.solved:
                        return self._result_from_state(candidate, explored)
                    next_frontier.append(candidate)
            next_frontier.sort(key=lambda item: (-item.score, len(item.steps), -self._operator_diversity(item)))
            frontier = next_frontier[:10]
            if not frontier:
                break

        return self._result_from_state(best, explored, solved=False)

    def _memory_entries(self, problem: OlympiadProblem) -> List[PatternMemoryEntry]:
        normalized = self._normalize(problem.query)
        matches: List[PatternMemoryEntry] = []
        for entry in PATTERN_MEMORY:
            if entry.problem_type != problem.problem_type:
                continue
            if all(token in normalized for token in entry.trigger_terms[: min(2, len(entry.trigger_terms))]):
                matches.append(entry)
        return matches

    @staticmethod
    def _preferred_operators(entries: List[PatternMemoryEntry]) -> List[str]:
        ordered: List[str] = []
        for entry in entries:
            for operator in entry.preferred_operators:
                if operator not in ordered:
                    ordered.append(operator)
        return ordered

    def _state_priority(self, state: ProofState, problem: OlympiadProblem, entries: List[PatternMemoryEntry]) -> float:
        score = state.score
        goal_tokens = set(re.findall(r"[a-z0-9]+", problem.goal.lower()))
        fact_text = " ".join(state.facts).lower()
        step_text = " ".join(step.statement + " " + step.rationale for step in state.steps).lower()
        overlap = len({token for token in goal_tokens if token and token in fact_text})
        score += min(0.18, 0.03 * overlap)
        memory_hits = 0
        for entry in entries:
            for hint in entry.hint_facts:
                keywords = [token for token in re.findall(r"[a-z0-9]+", hint.lower()) if len(token) > 2]
                if keywords and any(token in step_text for token in keywords):
                    memory_hits += 1
        score += min(0.16, 0.04 * memory_hits)
        preferred_hits = sum(1 for step in state.steps if step.operator in state.preferred_operators)
        score += min(0.14, 0.03 * preferred_hits)
        score += min(0.08, 0.02 * self._operator_diversity(state))
        score -= 0.01 * max(0, len(state.steps) - 4)
        return round(min(0.99, score), 2)

    @staticmethod
    def _operator_diversity(state: ProofState) -> int:
        return len({step.operator for step in state.steps})

    def _apply_operators(self, state: ProofState, problem: OlympiadProblem) -> List[ProofState]:
        operators: List[Callable[[ProofState, OlympiadProblem], ProofState | None]] = []
        if problem.problem_type == "odd_sum_even":
            operators = [self._setup_odd_representations, self._expand_odd_sum, self._conclude_even_form]
        elif problem.problem_type == "odd_square_odd":
            operators = [self._setup_single_odd_representation, self._expand_odd_square, self._conclude_odd_square]
        elif problem.problem_type == "consecutive_product_even":
            operators = [self._setup_consecutive_pair, self._split_parity_cases, self._conclude_consecutive_product_even]
        elif problem.problem_type == "three_consecutive_sum_mod3":
            operators = [self._setup_three_consecutive_terms, self._expand_three_term_sum, self._conclude_mod_three]
        elif problem.problem_type == "pigeonhole_shared_bucket":
            operators = [self._setup_pigeonhole_counts, self._apply_pigeonhole_bound, self._conclude_pigeonhole]
        elif problem.problem_type == "infinitely_many_primes":
            operators = [self._assume_finite_primes, self._build_euclid_number, self._derive_prime_contradiction, self._conclude_infinitely_many_primes]
        elif problem.problem_type == "induction_sum_formula":
            operators = [self._setup_induction_base, self._state_induction_hypothesis, self._perform_induction_step, self._conclude_induction_formula]
        elif problem.problem_type == "extremal_average_bound":
            operators = [self._choose_extremal_element, self._compare_extremal_with_average, self._derive_average_contradiction, self._conclude_extremal_average_bound]
        elif problem.problem_type == "generic_number_theory":
            operators = [self._restate_goal, self._scan_number_theory_meta_operators, self._propose_subgoal_split, self._record_strategy_checkpoint]
        elif problem.problem_type == "generic_combinatorics":
            operators = [self._restate_goal, self._scan_combinatorics_meta_operators, self._propose_subgoal_split, self._record_strategy_checkpoint]
        elif problem.problem_type == "generic_geometry":
            operators = [self._restate_goal, self._scan_geometry_meta_operators, self._propose_subgoal_split, self._record_strategy_checkpoint]
        elif problem.problem_type == "generic_algebra":
            operators = [self._restate_goal, self._scan_algebra_meta_operators, self._propose_subgoal_split, self._record_strategy_checkpoint]
        elif problem.problem_type == "mutilated_chessboard":
            operators = [self._setup_chessboard_coloring, self._remove_same_color_corners, self._apply_domino_color_invariant, self._conclude_mutilated_chessboard]

        results: List[ProofState] = []
        for operator in operators:
            candidate = operator(state, problem)
            if candidate is not None:
                results.append(candidate)
        return results

    def _setup_odd_representations(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "let a = 2m + 1 and b = 2n + 1"
        if state.has_fact(fact):
            return None
        return self._step(state, "PARITY_REPRESENTATION", fact, "Any odd integer can be written as 2k+1.", 0.25)

    def _expand_odd_sum(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let a = 2m + 1 and b = 2n + 1") or state.has_fact("a + b = 2(m + n + 1)"):
            return None
        return self._step(state, "ALGEBRAIC_EXPANSION", "a + b = 2(m + n + 1)", "Substitute the odd representations and factor out 2.", 0.35)

    def _conclude_even_form(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("a + b = 2(m + n + 1)"):
            return None
        candidate = self._step(state, "DIVISIBILITY_CONCLUSION", "therefore the sum of two odd integers is even", "A number of the form 2t is even.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_single_odd_representation(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "let n = 2k + 1"
        if state.has_fact(fact):
            return None
        return self._step(state, "PARITY_REPRESENTATION", fact, "Represent an odd integer as 2k+1.", 0.25)

    def _expand_odd_square(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let n = 2k + 1") or state.has_fact("n^2 = 2(2k^2 + 2k) + 1"):
            return None
        return self._step(state, "ALGEBRAIC_EXPANSION", "n^2 = 2(2k^2 + 2k) + 1", "Expand (2k+1)^2 and rewrite it as twice an integer plus 1.", 0.35)

    def _conclude_odd_square(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("n^2 = 2(2k^2 + 2k) + 1"):
            return None
        candidate = self._step(state, "PARITY_CONCLUSION", "therefore the square of an odd integer is odd", "A number of the form 2t+1 is odd.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_consecutive_pair(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "let the integers be n and n+1"
        if state.has_fact(fact):
            return None
        return self._step(state, "CONSECUTIVE_REWRITE", fact, "Consecutive integers differ by exactly 1.", 0.25)

    def _split_parity_cases(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let the integers be n and n+1") or state.has_fact("one of n or n+1 is even"):
            return None
        return self._step(state, "PARITY_CASE_SPLIT", "one of n or n+1 is even", "Among consecutive integers, parity alternates, so exactly one is even.", 0.35)

    def _conclude_consecutive_product_even(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("one of n or n+1 is even"):
            return None
        candidate = self._step(state, "PRODUCT_EVENNESS", "therefore the product of two consecutive integers is even", "A product containing an even factor is even.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_three_consecutive_terms(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "let the integers be n, n+1, and n+2"
        if state.has_fact(fact):
            return None
        return self._step(state, "CONSECUTIVE_REWRITE", fact, "Three consecutive integers can be parameterized by n, n+1, and n+2.", 0.22)

    def _expand_three_term_sum(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let the integers be n, n+1, and n+2") or state.has_fact("n + (n+1) + (n+2) = 3(n+1)"):
            return None
        return self._step(state, "MODULAR_REWRITE", "n + (n+1) + (n+2) = 3(n+1)", "Group the three terms and factor out 3.", 0.36)

    def _conclude_mod_three(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("n + (n+1) + (n+2) = 3(n+1)"):
            return None
        candidate = self._step(state, "MODULAR_DIVISIBILITY", "therefore the sum of three consecutive integers is divisible by 3", "Any expression equal to 3 times an integer is divisible by 3.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_pigeonhole_counts(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        items = int(problem.parameters.get("items", 0))
        buckets = int(problem.parameters.get("buckets", 0))
        fact = f"there are {items} items and {buckets} buckets"
        if not items or not buckets or state.has_fact(fact):
            return None
        return self._step(state, "COUNT_OBJECTS_AND_BUCKETS", fact, "Pigeonhole arguments begin by counting objects and categories.", 0.25)

    def _apply_pigeonhole_bound(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        items = int(problem.parameters.get("items", 0))
        buckets = int(problem.parameters.get("buckets", 0))
        bound = f"{items} > {buckets}"
        if not state.has_fact(f"there are {items} items and {buckets} buckets") or state.has_fact(bound):
            return None
        return self._step(state, "PIGEONHOLE_PRINCIPLE", bound, "If there are more objects than buckets, some bucket contains at least two objects.", 0.35)

    def _conclude_pigeonhole(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        items = int(problem.parameters.get("items", 0))
        if not state.has_fact(f"{items} > {int(problem.parameters.get('buckets', 0))}"):
            return None
        bucket_label = str(problem.parameters.get("bucket_label", "category"))
        candidate = self._step(state, "PIGEONHOLE_CONCLUSION", f"therefore at least two objects share the same {bucket_label}", "This is the direct conclusion of the pigeonhole principle.", 0.4)
        candidate.solved = True
        return candidate

    def _assume_finite_primes(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "assume there are finitely many primes p1, p2, ..., pk"
        if state.has_fact(fact):
            return None
        return self._step(state, "CONTRADICTION_SETUP", fact, "Start a contradiction proof by assuming the opposite of the claim.", 0.22)

    def _build_euclid_number(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("assume there are finitely many primes p1, p2, ..., pk") or state.has_fact("let N = p1*p2*...*pk + 1"):
            return None
        return self._step(state, "AUXILIARY_CONSTRUCTION", "let N = p1*p2*...*pk + 1", "Construct a number one larger than the product of all listed primes.", 0.32)

    def _derive_prime_contradiction(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let N = p1*p2*...*pk + 1") or state.has_fact("no listed prime divides N"):
            return None
        return self._step(state, "DIVISIBILITY_CONTRADICTION", "no listed prime divides N", "Dividing N by any listed prime leaves remainder 1.", 0.34)

    def _conclude_infinitely_many_primes(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("no listed prime divides N"):
            return None
        candidate = self._step(state, "CONTRADICTION_CONCLUSION", "therefore there are infinitely many primes", "N itself is prime or has a prime factor outside the assumed list, contradicting finiteness.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_induction_base(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "base case n=1 gives 1 = 1(1+1)/2"
        if state.has_fact(fact):
            return None
        return self._step(state, "INDUCTION_BASE", fact, "Check the claimed formula at the smallest valid value.", 0.22)

    def _state_induction_hypothesis(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("base case n=1 gives 1 = 1(1+1)/2"):
            return None
        fact = "assume 1+2+...+k = k(k+1)/2"
        if state.has_fact(fact):
            return None
        return self._step(state, "INDUCTION_HYPOTHESIS", fact, "Assume the formula holds for an arbitrary positive integer k.", 0.28)

    def _perform_induction_step(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("assume 1+2+...+k = k(k+1)/2"):
            return None
        fact = "1+2+...+k+(k+1) = (k+1)(k+2)/2"
        if state.has_fact(fact):
            return None
        return self._step(state, "INDUCTION_STEP", fact, "Add k+1 to both sides of the induction hypothesis and simplify the algebra.", 0.34)

    def _conclude_induction_formula(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("1+2+...+k+(k+1) = (k+1)(k+2)/2"):
            return None
        candidate = self._step(state, "INDUCTION_CONCLUSION", "therefore 1+2+...+n = n(n+1)/2 for all positive integers n", "The base case and induction step together prove the formula for every positive integer n.", 0.4)
        candidate.solved = True
        return candidate

    def _choose_extremal_element(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "let m be a largest element of the finite set"
        if state.has_fact(fact):
            return None
        return self._step(state, "EXTREMAL_CHOICE", fact, "Finite sets admit a largest element, which is a natural extremal choice.", 0.22)

    def _compare_extremal_with_average(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("let m be a largest element of the finite set"):
            return None
        fact = "if every element were greater than the average, then their sum would exceed average times the number of elements"
        if state.has_fact(fact):
            return None
        return self._step(state, "AVERAGE_BOUND", fact, "Assume the contrary and compare each element with the common average threshold.", 0.3)

    def _derive_average_contradiction(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("if every element were greater than the average, then their sum would exceed average times the number of elements"):
            return None
        fact = "but average times the number of elements equals the sum, a contradiction"
        if state.has_fact(fact):
            return None
        return self._step(state, "SUM_CONTRADICTION", fact, "The definition of average makes the assumed strict inequality impossible.", 0.34)

    def _conclude_extremal_average_bound(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("but average times the number of elements equals the sum, a contradiction"):
            return None
        candidate = self._step(state, "EXTREMAL_CONCLUSION", "therefore some element is at most the average", "The contradiction shows not every element can lie strictly above the average.", 0.4)
        candidate.solved = True
        return candidate

    def _setup_chessboard_coloring(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "a domino placed on a chessboard always covers one white square and one black square"
        if state.has_fact(fact):
            return None
        return self._step(state, "COLORING_INVARIANT", fact, "Use checkerboard coloring to track an invariant under domino placement.", 0.24)

    def _remove_same_color_corners(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("a domino placed on a chessboard always covers one white square and one black square") or state.has_fact("removing two opposite corners removes two squares of the same color"):
            return None
        return self._step(state, "CORNER_COLOR_ANALYSIS", "removing two opposite corners removes two squares of the same color", "Opposite corners of a chessboard share the same color.", 0.34)

    def _apply_domino_color_invariant(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("removing two opposite corners removes two squares of the same color") or state.has_fact("the mutilated board has unequal numbers of white and black squares"):
            return None
        return self._step(state, "INVARIANT_COUNT_MISMATCH", "the mutilated board has unequal numbers of white and black squares", "After removing two same-color corners, the white/black counts are no longer balanced.", 0.34)

    def _conclude_mutilated_chessboard(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        if not state.has_fact("the mutilated board has unequal numbers of white and black squares"):
            return None
        candidate = self._step(state, "INVARIANT_CONCLUSION", "therefore the mutilated chessboard cannot be tiled by dominoes", "Every domino preserves equal white/black coverage, so a color imbalance makes tiling impossible.", 0.4)
        candidate.solved = True
        return candidate

    def _restate_goal(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = f"restated goal: {problem.goal}"
        if state.has_fact(fact):
            return None
        return self._step(state, "GOAL_RESTATEMENT", fact, "Rewrite the claim in a form that exposes the target relation or invariant.", 0.18)

    def _scan_number_theory_meta_operators(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "candidate tools: modular arithmetic, parity, divisibility, contradiction"
        if state.has_fact(fact):
            return None
        return self._step(state, "META_OPERATOR_SCAN", fact, "Number-theory olympiad problems often open with modular, divisibility, or contradiction tactics.", 0.2)

    def _scan_combinatorics_meta_operators(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "candidate tools: pigeonhole, invariant, extremal counting, constructive bijection"
        if state.has_fact(fact):
            return None
        return self._step(state, "META_OPERATOR_SCAN", fact, "Combinatorics proofs often depend on a counting bottleneck or a preserved invariant.", 0.2)

    def _scan_geometry_meta_operators(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "candidate tools: auxiliary lines, similar triangles, angle chase, area ratios"
        if state.has_fact(fact):
            return None
        return self._step(state, "META_OPERATOR_SCAN", fact, "Geometry problems usually need an auxiliary construction before the key relation becomes visible.", 0.2)

    def _scan_algebra_meta_operators(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "candidate tools: substitution, factorization, symmetry, inequality normalization"
        if state.has_fact(fact):
            return None
        return self._step(state, "META_OPERATOR_SCAN", fact, "Algebra olympiad problems often simplify after symmetry detection or a well-chosen substitution.", 0.2)

    def _propose_subgoal_split(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "subgoals: identify invariant quantities, normalize notation, and test a small case"
        if state.has_fact(fact):
            return None
        return self._step(state, "SUBGOAL_DECOMPOSITION", fact, "Before committing to a proof, isolate reusable lemmas and sanity-check the claim on a small example.", 0.18)

    def _record_strategy_checkpoint(self, state: ProofState, problem: OlympiadProblem) -> ProofState | None:
        fact = "strategy checkpoint: continue from the strongest meta-operator and search for an auxiliary lemma"
        if state.has_fact(fact):
            return None
        return self._step(state, "STRATEGY_CHECKPOINT", fact, "The exact proof is not complete yet, but the search state now records a structured next move.", 0.14)

    def _step(self, state: ProofState, operator: str, statement: str, rationale: str, score_gain: float) -> ProofState:
        candidate = state.clone()
        candidate.facts.append(statement)
        candidate.steps.append(ProofStep(operator=operator, statement=statement, rationale=rationale))
        candidate.score = round(min(0.99, candidate.score + score_gain), 2)
        return candidate

    def _result_from_state(self, state: ProofState, explored: int, solved: bool | None = None) -> OlympiadSearchResult:
        solved = state.solved if solved is None else solved
        if solved:
            answer = f"Olympiad proof sketch: {state.steps[-1].statement}."
            confidence = max(0.72, state.score)
        else:
            answer = "Olympiad proof sketch is incomplete; the current search state identifies a plausible direction but not a full proof."
            confidence = min(0.65, state.score)
        memory_lines = [f"memory hint: {note}" for note in state.memory_notes]
        proof_outline = memory_lines + [f"{step.statement} [{step.rationale}]" for step in state.steps]
        operator_trace = (["MEMORY_HINT"] * len(memory_lines)) + [step.operator for step in state.steps]
        return OlympiadSearchResult(
            solved=solved,
            final_statement=answer,
            proof_outline=proof_outline,
            operator_trace=operator_trace,
            explored_states=explored,
            confidence=round(confidence, 2),
            problem_type=state.problem_type,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().replace("\n", " ").split())

    @staticmethod
    def _looks_like_proof_problem(normalized: str) -> bool:
        triggers = ["prove", "show that", "demonstrate that", "at least two", "odd", "consecutive", "integer", "prime", "domino", "chessboard", "first n", "1+2", "average", "finite set"]
        return any(token in normalized for token in triggers)

    @staticmethod
    def _is_odd_sum_problem(normalized: str) -> bool:
        return all(token in normalized for token in ["odd", "sum", "even"]) and any(token in normalized for token in ["two", "2"])

    @staticmethod
    def _is_odd_square_problem(normalized: str) -> bool:
        return "odd" in normalized and "square" in normalized and any(token in normalized for token in ["prove", "show that"])

    @staticmethod
    def _is_consecutive_product_problem(normalized: str) -> bool:
        if "consecutive" not in normalized or "even" not in normalized:
            return False
        return any(token in normalized for token in ["product", "multiply", "n(n+1)", "n(n + 1)"])

    @staticmethod
    def _is_three_consecutive_sum_problem(normalized: str) -> bool:
        return "three consecutive" in normalized and "sum" in normalized and any(token in normalized for token in ["divisible by 3", "multiple of 3"])

    @staticmethod
    def _is_infinite_primes_problem(normalized: str) -> bool:
        return "prime" in normalized and any(token in normalized for token in ["infinitely many", "infinite many", "infinitely", "infinite number"])

    @staticmethod
    def _is_induction_sum_problem(normalized: str) -> bool:
        has_formula = any(token in normalized for token in ["n(n+1)/2", "n(n + 1)/2", "n(n+1) / 2"])
        has_sum = any(token in normalized for token in ["1+2+...+n", "1 + 2 + ... + n", "first n positive integers", "sum of the first n"])
        return has_formula and has_sum

    @staticmethod
    def _is_extremal_average_problem(normalized: str) -> bool:
        if "average" not in normalized:
            return False
        if not any(token in normalized for token in ["finite set", "numbers", "elements", "integers"]):
            return False
        return any(token in normalized for token in ["at most the average", "not greater than the average", "less than or equal to the average"])

    @staticmethod
    def _is_domino_coloring_problem(normalized: str) -> bool:
        return all(token in normalized for token in ["domino", "chessboard"]) and any(token in normalized for token in ["opposite corners", "two opposite corners", "removed corners"])

    def _parse_pigeonhole_problem(self, normalized: str) -> OlympiadProblem | None:
        if "at least two" not in normalized:
            return None
        items_match = re.search(r"(\d+)\s+(?:people|students|items|objects|integers|persons)", normalized)
        bucket_match = re.search(r"(\d+)\s+(?:months|boxes|drawers|buckets|classes)", normalized)
        if items_match and bucket_match:
            bucket_label = self._bucket_label(bucket_match.group(0))
            return OlympiadProblem(
                problem_type="pigeonhole_shared_bucket",
                goal=f"at least two objects share the same {bucket_label}",
                query=normalized,
                triggers=["at least two", "pigeonhole"],
                parameters={"items": int(items_match.group(1)), "buckets": int(bucket_match.group(1)), "bucket_label": bucket_label},
            )
        if "birth month" in normalized or "same month" in normalized:
            items_match = re.search(r"(\d+)\s+(?:people|students|persons)", normalized)
            if items_match:
                return OlympiadProblem(
                    problem_type="pigeonhole_shared_bucket",
                    goal="at least two people share the same birth month",
                    query=normalized,
                    triggers=["at least two", "birth month", "pigeonhole"],
                    parameters={"items": int(items_match.group(1)), "buckets": 12, "bucket_label": "birth month"},
                )
        return None

    def _parse_generic_problem(self, query: str, normalized: str) -> OlympiadProblem | None:
        if any(token in normalized for token in ["triangle", "circle", "perpendicular", "parallel", "median", "angle"]):
            return OlympiadProblem(
                problem_type="generic_geometry",
                goal="find a geometry proof strategy",
                query=query,
                triggers=["geometry", "auxiliary construction"],
            )
        if any(token in normalized for token in ["mod", "divisible", "prime", "integer", "remainder"]):
            return OlympiadProblem(
                problem_type="generic_number_theory",
                goal="find a number-theory proof strategy",
                query=query,
                triggers=["modular arithmetic", "divisibility"],
            )
        if any(token in normalized for token in ["arrangement", "coloring", "count", "subset", "set of"]):
            return OlympiadProblem(
                problem_type="generic_combinatorics",
                goal="find a combinatorial proof strategy",
                query=query,
                triggers=["counting", "invariant"],
            )
        if any(token in normalized for token in ["inequality", "polynomial", "symmetric", "factor"]):
            return OlympiadProblem(
                problem_type="generic_algebra",
                goal="find an algebraic proof strategy",
                query=query,
                triggers=["substitution", "normalization"],
            )
        return None

    @staticmethod
    def _bucket_label(text: str) -> str:
        lowered = text.lower()
        if "month" in lowered:
            return "month"
        if "box" in lowered or "drawer" in lowered or "bucket" in lowered:
            return "box"
        if "class" in lowered:
            return "class"
        return "category"
