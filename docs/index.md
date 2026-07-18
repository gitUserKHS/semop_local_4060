# Docs Index

This folder contains the active documentation for SemOp Local 4060.

## Start Here

- `../README.md`
  - top-level overview, quick start, and validation status
- `usage_manual.md`
  - practical guide, including the easiest starter GUI
- `project_structure.md`
  - generated code layout; refresh with `tools/maintenance/update_code_structure_docs.py`
- `architecture_and_features.md`
  - current capability map and module-level status

## Product And Evaluation

- `gui_and_eval_workflow.md`
  - operations copilot GUI and KPI evaluation flow
- `feedback_loop_workflow.md`
  - review queue to feedback-rule workflow
- `customer_eval_schema.md`
  - labeled customer-eval JSONL schema
- `productization_review.md`
  - product framing and domain strategy notes

## Reasoning Core And Research

- `operator_intelligence_system.md`
  - high-level definition of the logic-operator intelligence architecture
- `low_resource_operator_intelligence.md`
  - mandatory sample-efficiency and ordinary-PC resource doctrine and evaluation gates
- `compositional_operator_intelligence.md`
  - 작은 공유 뇌가 언어·수학·비전의 typed operator program을 조합하는 장기 구조와 코드 정리 원칙
- `typed_operator_core.md`
  - immutable typed IR, verifier-first search, three domain adapters, and migration modes
- `trust_provenance_and_metrics.md`
  - assertion/evidence/logical 신뢰 축, 가정 의존 proof, replay integrity와 semantic correctness 분리, CI 계약
- `lmv_semantic_benchmark.md`
  - digest-bound 사람 리뷰, 언어·수학·비전 near-miss seed, 권한별 의미 정확도 평가 절차
- `language_math_vision_typed_runtime.md`
  - direct language/math/vision adapters, trust boundaries, unified runtime, and benchmark workflow
- `language_text_adapter.md`
  - 명시적 한국어·영어 문장을 typed 전제로 바꾸고 애매한 후보를 격리하는 계약
- `typed_compositional_extensions.md`
  - 언어 Horn 추론, exact 일차방정식, raster 도형·개수·면적, controller v5 점수 계약
- `composed_operator_runtime.md`
  - 공통 registry 조합, goal-independent 관측, operator frontier, 비전→수학→언어 proof program
- `typed_dataflow.md`
  - verified 숫자 측정을 조건과 결론 operator로 컴파일하고 실제 의미 흐름을 held-out 평가하는 계약
- `frontier_llm_judge.md`
  - 프론티어 LLM teacher/judge의 proposed 경계, replay 검증, 안전한 controller 학습 유입
- `verifier_gated_self_learning.md`
  - 실제 adapter capability를 구조적으로 분리하고 능동 선택한 뒤 작은 정책을 승격·rollback하는 닫힌 루프
- `raw_grounded_self_learning.md`
  - raw 언어·수학·pixel 입력을 typed 경험으로 감사하고 verifier trace만 학습하는 공개 자가 학습 경계
- `hierarchical_operator_brain.md`
  - 언어-only controller 학습, 수학·비전 zero-shot 전이, 검증된 macro memory 결합, sparse·29K·5.84M 비교
- `active_macro_learning.md`
  - primitive-expanded procedural memory의 독립 validation, schema pinning, replay, 승격·rollback 계약
- `self_discovered_curriculum.md`
  - 검증된 seed에서 새 다중도메인 조합과 counterfactual을 발견하고 계보·중복·깊이를 통제하는 curriculum
- `verified_rule_discovery.md`
  - flat Horn 규칙 귀납, 개별 반증, final joint library 승격, promotion certificate와 atomic rollback 경계
- `raster_vision.md`
  - dependency-free RGB component detection, verified pixel geometry, and neural detector boundary
- `tiny_controller.md`
  - 5.84M relation-aware policy, NumPy inference, PyTorch training, trace data, and MDL macros
- `low_resource_transfer_evaluation.md`
  - three-domain structural splits, A/B metrics, resource limits, and promotion gates
- `lodo_controller_experiment.md`
  - leakage-controlled leave-one-domain-out controller training and current evidence
- `operator_intelligence_roadmap.md`
  - roadmap grouped by operator learning, world model, memory, and verifier quality
- `operator_intelligence_execution_steps.md`
  - staged execution plan from geometry primitives to cross-modal reasoning
- `premise_first_refactor_status.md`
  - current premise-first architecture changes and remaining limits
- `premise_engine_next_steps.md`
  - current hidden-premise metrics and recommended next implementation order
- `jepa_relevance_and_integration.md`
  - why JEPA helps and how it is integrated as latent structural prediction
- `research_strengthening_plan_2026_03_10.md`
  - 3, 6, and 12 month roadmap along operator, world-model, memory, and verifier axes
- `logical_grammar_goal_and_implementation.md`
  - logical word to concept-frame binding goal and current implementation
- `hard_problem_training_and_verification.md`
  - hard-problem verification and weight-learning loop
- `corpus_scale_hierarchy_plan.md`
  - operator hierarchy roadmap and status
- `pdf_alignment_notes.md`
  - alignment notes against the original research PDF
- `vlso_structural_first_rearchitecture.md`
  - why VLSO now uses geometry/topology-first operator induction before semantic labels
- `operator_algebra_and_functors.md`
  - hidden-premise alignment, operator decomposition, and functor-hypothesis notes
- `operator_intelligence_to_100_plan.md`
  - staged path from current prototype toward stronger operator-centered intelligence
- `operator_learning_plan.md`
  - teacher-trace curriculum, QLoRA workflow, and generic operator-student training
- `analogical_memory_plan.md`
  - multi-analogy retrieval, structural similarity explanation, and rollout plan

## Competitive Programming

- `cp_focus_research_and_implementation.md`
  - CP DSL, episodic memory, validator, and repair loop status
- `cp_training_manual.md`
  - detailed CP parser training guide for RTX 4060 8GB
- `cp_gui_manual.md`
  - CP GUI usage and incident-ingest workflow
- `../examples/cp_geometry_parser_eval.jsonl`
  - generated geometry-only CP parser evaluation target

## VLSO And Visual Learning

- `vlso_implementation_plan.md`
  - VLSO architecture and implementation scope
- `vlso_data_collection_guide.md`
  - small-data visual concept collection and labeling workflow
- `data_collection_api_research.md`
  - researched public API and dataset options for automated collection
- `../examples/vlso_geometry_eval.jsonl`
  - geometry-grounded VLSO QA starter set

## Maintenance And Generated Docs

- `data_layout.md`
  - where inputs, generated outputs, and SQLite stores live
- `project_structure.md`
  - generated structure snapshot of entrypoints, tools, and source directories

## Starter Eval Assets

- `../examples/vlso_eval.jsonl`
  - starter grounded image QA evaluation set
- `../examples/vlso_geometry_eval.jsonl`
  - starter geometry-grounded image QA evaluation set
- `../examples/cp_parser_eval.jsonl`
  - starter held-out CP parser evaluation set
- `../examples/cp_geometry_parser_eval.jsonl`
  - geometry-only CP parser evaluation set target

- `operator_intelligence_to_100_plan.md`: research-backed staged plan toward a full operator-intelligence stack

- `qlora_distillation_roadmap.md`
  - QLoRA and distillation roadmap for small local models
- `multi_agent_operator_doctrine.md`
  - mandatory shared rule for all agents: basis operators first, verifier-retained operator algebra second


