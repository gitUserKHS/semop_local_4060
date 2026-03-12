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


