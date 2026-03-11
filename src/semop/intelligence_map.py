from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntelligenceSubsystem:
    name: str
    goal: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class IntelligenceAxis:
    name: str
    description: str
    subsystems: tuple[IntelligenceSubsystem, ...]


@dataclass(frozen=True)
class OperatorIntelligenceMap:
    mission: str
    axes: tuple[IntelligenceAxis, ...]


def build_operator_intelligence_map() -> OperatorIntelligenceMap:
    return OperatorIntelligenceMap(
        mission=(
            "Convert language, vision, operations, and problem-solving inputs into shared "
            "operator-structured world models, then reason with memory and verification "
            "instead of relying on direct text generation alone."
        ),
        axes=(
            IntelligenceAxis(
                name="operator_learning",
                description=(
                    "Learn reusable operator families, logical frames, and relation patterns "
                    "that generalize across domains with small labeled or pseudo-labeled data."
                ),
                subsystems=(
                    IntelligenceSubsystem(
                        name="logical_grammar",
                        goal="Bind logical words such as has, if, before, and can to reusable frames.",
                        modules=(
                            "src/semop/logical_grammar.py",
                            "src/semop/emergent_operators.py",
                            "src/semop/operator_hierarchy.py",
                            "src/semop/operator_registry.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="visual_operator_learning",
                        goal="Compress few-shot visual relations into reusable operator prototypes.",
                        modules=(
                            "src/semop/vlso/operator_learning.py",
                            "src/semop/vlso/self_training.py",
                            "src/semop/vlso/concept_learning.py",
                        ),
                    ),
                ),
            ),
            IntelligenceAxis(
                name="world_model",
                description="Build explicit, typed, inspectable models of the current task or scene.",
                subsystems=(
                    IntelligenceSubsystem(
                        name="structured_meaning_graph",
                        goal="Turn text into typed reasoning graphs with constraints and scripts.",
                        modules=(
                            "src/semop/pipeline.py",
                            "src/semop/heuristic_extractors.py",
                            "src/semop/structures.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="vlso_world_model",
                        goal="Align language and visual evidence inside a shared operator space.",
                        modules=(
                            "src/semop/vlso/language_parser.py",
                            "src/semop/vlso/visual_parser.py",
                            "src/semop/vlso/aligner.py",
                            "src/semop/vlso/types.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="cp_problem_model",
                        goal="Lift contest statements into DSL, hidden structure, and algorithm families.",
                        modules=(
                            "src/semop/contest_programmer.py",
                            "src/semop/cp_knowledge.py",
                        ),
                    ),
                ),
            ),
            IntelligenceAxis(
                name="memory",
                description=(
                    "Retain reusable local exemplars, compressed global prototypes, episodes, and "
                    "embeddings that can be recalled during reasoning."
                ),
                subsystems=(
                    IntelligenceSubsystem(
                        name="corpus_and_query_memory",
                        goal="Store graphs, reusable operator families, and query priors.",
                        modules=(
                            "src/semop/corpus_store.py",
                            "src/semop/corpus_learning.py",
                            "src/semop/memory_retrieval.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="cp_episode_memory",
                        goal="Reuse solved, failed, and repaired contest episodes.",
                        modules=(
                            "src/semop/cp_episode_store.py",
                            "src/semop/cp_episode_ingest.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="vlso_hybrid_memory",
                        goal="Fuse local visual exemplars with compressed operator prototypes.",
                        modules=(
                            "src/semop/vlso/concept_memory.py",
                            "src/semop/vlso/embedding_store.py",
                            "src/semop/vlso/hybrid_memory.py",
                        ),
                    ),
                ),
            ),
            IntelligenceAxis(
                name="verifier",
                description=(
                    "Check executable validity, counterexamples, grounded evidence, and repair paths "
                    "so the system can reject brittle reasoning."
                ),
                subsystems=(
                    IntelligenceSubsystem(
                        name="operations_verifier",
                        goal="Measure executability, unsafe advice, and audit usefulness.",
                        modules=(
                            "src/semop/ops_kpi.py",
                            "src/semop/review_queue.py",
                            "src/semop/feedback_rules.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="cp_verifier",
                        goal="Compile, run, counterexample-check, and repair generated contest code.",
                        modules=(
                            "src/semop/cp_validation.py",
                            "src/semop/cp_repair.py",
                            "src/semop/cpp_compiler.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="hard_problem_verifier",
                        goal="Verify symbolic reasoning and update pattern weights from outcomes.",
                        modules=(
                            "src/semop/hard_problem_engine.py",
                            "src/semop/symbolic_reasoners.py",
                            "src/semop/olympiad_reasoner.py",
                        ),
                    ),
                    IntelligenceSubsystem(
                        name="grounded_visual_qa",
                        goal="Answer visual questions from grounded operator evidence rather than raw captioning.",
                        modules=(
                            "src/semop/vlso/qa.py",
                            "src/semop/vlso/reasoner.py",
                            "src/semop/vlso/object_reasoner.py",
                        ),
                    ),
                ),
            ),
        ),
    )
