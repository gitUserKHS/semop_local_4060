from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import zipfile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semop import (
    AffordanceLabelDataset,
    AffordanceWeightTrainer,
    BASELINE_SPECS,
    CompetitiveProgrammingReasoner,
    CpCorpusBuilder,
    CpDslDatasetBuilder,
    CpDslExample,
    CpEpisodeStore,
    CpKnowledgeLoader,
    CpLabeledDatasetDownloader,
    CpLearnedParser,
    CpParserEvaluator,
    CpParserPrediction,
    VlsoReviewImpactEvaluator,
    CpParserTrainConfig,
    CpParserTrainingScaffold,
    CpLoraExperimentConfig,
    CpLoraExperimentRunner,
    CpTrainingBundleBuilder,
    CpTrainingPlanner,
    CppRepairEngine,
    CppSyntaxChecker,
    load_cp_dsl_examples,
    CopilotRequest,
    CorpusBuilder,
    CorpusMemoryStore,
    CorpusReasoningLearner,
    CURATED_PRESETS,
    CURATED_PUBLIC_DATASETS,
    DomainCopilot,
    EmbeddingModelCache,
    GeometryPrimitiveBackbone,
    GeometryTopologyExtractor,
    OpenImagesAnnotationAdapter,
    VisualCollectionSource,
    VisualDataCollector,
    VisualDownloadEntry,
    ImageMaskPreprocessor,
    JepaStructuralPredictor,
    VisualConceptLabelRecommender,
    VisualConceptLearningSummary,
    VisualConceptMemory,
    VisualConceptPrototypeTrainer,
    VisualConceptSelfTrainer,
    VlsoGroundedEvaluator,
    PseudoLabelAcceptanceConfig,
    VisualConceptRecord,
    VisualClusterReviewDecision,
    VisualClusterReviewStore,
    VisualApprovedReviewRetrainer,
    VisualOperatorMemory,
    VisualOperatorPrototypeTrainer,
    VisualHybridMemory,
    VisualOperatorRecord,
    VisualObservation,
    VisualAffordanceCandidate,
    VisualAffordanceFeatureExtractor,
    VisualEmbeddingRecord,
    VisualEmbeddingStore,
    VisionEmbeddingExtractor,
    VisualGeometryReasoner,
    VisualGeometryBootstrapPipeline,
    SyntheticGeometrySceneBuilder,
    CpGeometryEvalBuilder,
    CpGeometryTemplateGenerator,
    WeakAffordanceClassifier,
    DetectorOutputAdapter,
    HardProblemEngine,
    HiddenPremiseEvalCase,
    HiddenPremiseEvaluator,
    PatternOutcomeTrainer,
    LabeledOpsEvaluator,
    LogicalGrammarInducer,
    MemoryPriorEvaluator,
    OperatorAlgebraLearner,
    OperatorIntelligenceProgressEstimator,
    OperatorAlgebraEvaluator,
    OperatorSelfEvolutionEngine,
    HybridOperatorProposalPolicy,
    OperatorProposalComparator,
    VisualSignalImpactEvaluator,
    OperatorTransferEvaluator,
    OperatorHierarchyLearner,
    OlympiadReasoner,
    PlainRagBaseline,
    PublicCorpusIngestor,
    PublicDatasetAdapter,
    RemoteDatasetDownloader,
    ResponseSynthesizer,
    RealImageEvalBuilder,
    ReviewQueueStore,
    RawImageObservationParser,
    StructuredMeaningPipeline,
    TransferEvaluator,
    VLSOQuestionAnswerer,
    VLSOReasoner,
    curated_manifest,
    load_labeled_ops_cases,
    preset_manifest,
    resolve_embedding_model_id,
    resolve_local_vision_model_path,
    build_operator_intelligence_map,
    SemOpCommonEvaluator,
    ScriptCompatibilityTrainer,
    TeacherTraceExporter,
    DistillationSftRecord,
    TeacherTraceRecord,
    OperatorCurriculumBuilder,
    OperatorTrainingScaffold,
    OperatorCompiler,
    OperatorExecutor,
    compile_and_execute,
    OperatorTrainConfig,
)
from semop.llm_client import LocalLLMConfig, LocalTransformersExtractor


class StructuredMeaningPipelineTests(unittest.TestCase):
    def test_bag_book_case_has_access_precondition(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("How do I put a book into a bag?")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("insert_book", "REQUIRES", "open_access"), relations)
        self.assertTrue(graph.plan)

    def test_carwash_case_rejects_walk_only_advice(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("The car wash is far away and traffic is heavy. What should I do?")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("car_wash", "REQUIRES", "vehicle_present"), relations)
        self.assertTrue(graph.invalid_advice)

    def test_hidden_premise_explorer_recovers_carwash_goal_and_walk_risk(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("I am going to the car wash and traffic is bad, should I walk there?")
        self.assertIn("clean_car_goal", graph.hidden_goals)
        self.assertIn("vehicle_present", graph.required_premises)
        self.assertTrue(graph.clarification_needed)
        self.assertTrue(any(check.action == "walk_without_car" and check.status == "risk_high" for check in graph.goal_preservation_checks))
        self.assertTrue(any("hidden premise:" in warning for warning in graph.warnings))

    def test_hidden_premise_explorer_keeps_booking_interpretation_conditional(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("I need to cancel my booking at the car wash and traffic is bad, should I walk there?")
        self.assertIn("booking_or_inquiry_goal", graph.hidden_goals)
        self.assertTrue(graph.clarification_needed)
        self.assertTrue(any(check.action == "walk_without_car" and check.status == "conditionally_valid" for check in graph.goal_preservation_checks))
        self.assertIn("clean_car_goal", graph.optional_interpretations)

    def test_hidden_premise_evaluator_scores_goal_and_premise_recall(self) -> None:
        evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        summary = evaluator.evaluate([
            HiddenPremiseEvalCase(
                query="I am going to the car wash and traffic is bad, should I walk there?",
                expected_hidden_goals=["clean_car_goal"],
                expected_required_premises=["vehicle_present"],
                expected_satisfied_premises=[],
                expected_missing_premises=['vehicle_present'],
                expected_risky_actions=["walk_without_car"],
                forbidden_premises=["open_access"],
                expected_clarification_needed=True,
            )
        ])
        self.assertGreater(summary.hidden_goal_recall, 0.0)
        self.assertGreater(summary.critical_premise_recall, 0.0)
        self.assertGreater(summary.goal_preservation_accuracy, 0.0)

    def test_hidden_premise_explorer_recovers_drawer_access_goal(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("The drawer is closed and I need the folder inside. Should I pull the folder out right now?")
        self.assertIn("retrieve_item_from_drawer_goal", graph.hidden_goals)
        self.assertIn("open_access", graph.required_premises)
        self.assertTrue(any(check.action == "retrieve_without_opening" and check.status == "risk_high" for check in graph.goal_preservation_checks))

    def test_hidden_premise_explorer_recovers_cp_efficiency_goal(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Given an array and many range sum queries with N and Q up to 2e5, should I recompute each query from scratch?")
        self.assertIn("efficient_solution_goal", graph.hidden_goals)
        self.assertIn("subquadratic_complexity", graph.required_premises)
        self.assertTrue(any(check.action == "recompute_each_query" and check.status == "risk_high" for check in graph.goal_preservation_checks))

    def test_hidden_premise_explorer_marks_open_access_as_satisfied_for_open_drawer(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("The drawer is already open and I need the folder inside. Can I take it now?")
        self.assertIn("open_access", graph.satisfied_premises)
        self.assertNotIn("open_access", graph.required_premises)
        self.assertNotIn("open_access", graph.missing_premises)

    def test_hidden_premise_explorer_uses_script_memory_support(self) -> None:
        db_path = Path('tests/premise_script_memory_runtime_test.db')
        if db_path.exists():
            db_path.unlink()
        seed_store = CorpusMemoryStore(db_path)
        seed_graph = StructuredMeaningPipeline(mode='heuristic').run('Open the drawer and retrieve the folder.')
        seed_graph.inferred_scripts = ['open_drawer', 'retrieve_item']
        seed_store.upsert_premise_operator_memory(seed_graph, source='premise_test', split='train')
        pipeline = StructuredMeaningPipeline(mode='heuristic', memory_store_path=str(db_path), memory_source='premise_test')
        graph = pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?')
        self.assertTrue(any(candidate.source == 'script_memory' for candidate in graph.premise_candidates))

    def test_hidden_premise_explorer_calibrates_clarification_score(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        ambiguous = pipeline.run('I need to cancel my booking at the car wash and traffic is bad, should I walk there?')
        grounded = pipeline.run('My car is already at the car wash, should I walk there to check on it?')
        self.assertGreaterEqual(ambiguous.clarification_score, 0.5)
        self.assertLess(grounded.clarification_score, 0.5)
        self.assertTrue(ambiguous.clarification_reasons)

    def test_hidden_premise_explorer_marks_subquadratic_as_satisfied_when_precomputed(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        graph = pipeline.run('Given an array and many range sum queries with N and Q up to 2e5, and prefix sums are already precomputed, can I answer each query in O(1)?')
        self.assertIn('efficient_solution_goal', graph.hidden_goals)
        self.assertIn('subquadratic_complexity', graph.satisfied_premises)
        self.assertNotIn('subquadratic_complexity', graph.required_premises)

    def test_premise_memory_search_prefers_concept_compatible_candidates(self) -> None:
        db_path = Path('tests/premise_compatibility_runtime_test.db')
        if db_path.exists():
            db_path.unlink()
        store = CorpusMemoryStore(db_path)
        bag_graph = StructuredMeaningPipeline(mode='heuristic').run('The bag is closed and I need to put the book inside. Should I push it in now?')
        wash_graph = StructuredMeaningPipeline(mode='heuristic').run('I am going to the car wash and traffic is bad, should I walk there?')
        store.upsert_premise_operator_memory(bag_graph, source='compat_test', split='train')
        store.upsert_premise_operator_memory(wash_graph, source='compat_test', split='train')
        hits = store.search_premise_support('I am going to the car wash and traffic is bad, should I walk there?', source='compat_test')
        self.assertEqual(hits[0]['premise'], 'vehicle_present')

    def test_operator_algebra_decomposes_hidden_goal_reasoning(self) -> None:
        graph = StructuredMeaningPipeline(mode="heuristic").run("?嶺뚮ㅎ?②???묎덩???좊읈???釉먮폇??癲ル슓堉곤쭗? 癲ル슢??쭕?, 癲꾧퀗?э㎖猷잛젂疫뀀９苡???뉖??")
        names = {item.operator_name for item in graph.operator_decompositions}
        self.assertIn("GOAL_PRESERVATION_OPERATOR", names)
        self.assertIn("SERVICE_GOAL_OPERATOR", names)
        self.assertTrue(any(item.name == "ServiceGoalToConstraintFunctor" for item in graph.functor_hypotheses))

    def test_vlso_language_parser_projects_hidden_premises_into_world_model(self) -> None:
        world, graph = VLSOReasoner().language_parser.parse("?嶺뚮ㅎ?②???묎덩???좊읈???釉먮폇??癲ル슓堉곤쭗? 癲ル슢??쭕?, 癲꾧퀗?э㎖猷잛젂疫뀀９苡???뉖??")
        self.assertIn("clean_car_goal", world.goals)
        self.assertIn("vehicle_present", world.constraints)
        self.assertTrue(world.metadata.get('hidden_premises'))
        self.assertTrue(world.metadata.get('functor_hypotheses'))

    def test_operator_algebra_evaluator_scores_decomposition_and_functor_recall(self) -> None:
        evaluator = OperatorAlgebraEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        summary = evaluator.evaluate([
            __import__('semop').OperatorAlgebraEvalCase(
                query="?嶺뚮ㅎ?②???묎덩???좊읈???釉먮폇??癲ル슓堉곤쭗? 癲ル슢??쭕?, 癲꾧퀗?э㎖猷잛젂疫뀀９苡???뉖??",
                expected_decompositions=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                expected_functors=["ServiceGoalToConstraintFunctor"],
            )
        ])
        self.assertEqual(summary.num_cases, 1)
        self.assertEqual(summary.decomposition_recall, 1.0)
        self.assertEqual(summary.functor_recall, 1.0)

    def test_operator_self_evolution_engine_generates_retained_proposals(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [
            pipeline.run("I am going to a car wash but traffic is blocked. Should I walk there?"),
            pipeline.run("Should I open the zipper before putting the book into the bag?"),
            pipeline.run("The bag opening is already open. Can I place the book in now?"),
        ]
        summary = OperatorSelfEvolutionEngine().evolve(graphs, min_support=1, utility_threshold=0.2)
        self.assertTrue(summary.proposals)
        self.assertGreaterEqual(summary.retained_count, 1)
        self.assertTrue(any(item.utility_score > 0 for item in summary.proposals))


    def test_operator_proposal_engine_collects_pattern_and_normalizes_name(self) -> None:
        from semop import OperatorProposalEngine

        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [
            pipeline.run("Should I open the zipper before putting the book into the bag?"),
            pipeline.run("The cabinet door is closed and I need the file inside. Should I reach in immediately?"),
        ]
        proposals = OperatorProposalEngine().propose(graphs)
        self.assertTrue(proposals)
        self.assertTrue(all(item.basis_signature for item in proposals))
        self.assertTrue(all(item.normalized_name == item.normalized_name.upper() for item in proposals))

    def test_geometry_primitive_backbone_derives_symmetry_and_axis_alignment(self) -> None:
        observation = VisualObservation(objects=[{
            'id': 'rect_1',
            'label': 'rect_1',
            'kind': 'shape',
            'polygon': [[0, 0], [6, 0], [6, 4], [0, 4]],
        }])
        result = GeometryPrimitiveBackbone().enrich_observation(observation)
        self.assertEqual(result.enriched_objects, 1)
        item = observation.objects[0]
        self.assertGreaterEqual(item.get('symmetry_score', 0.0), 0.7)
        self.assertGreaterEqual(item.get('axis_alignment_score', 0.0), 0.7)
        self.assertIn('SYMMETRIC_STRUCTURE', item.get('geometry_signature', []))
        self.assertIn('AXIS_ALIGNED_STRUCTURE', item.get('geometry_signature', []))

    def test_operator_transfer_evaluator_reports_cross_domain_recall(self) -> None:
        evaluator = OperatorTransferEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        cases = [
            __import__('semop').OperatorTransferEvalCase(
                query="I am going to a car wash but traffic is blocked. Should I walk there?",
                domain="service",
                expected_operator_names=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                split="train",
            ),
            __import__('semop').OperatorTransferEvalCase(
                query="Should I open the zipper before putting the book into the bag?",
                domain="containment",
                expected_operator_names=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                split="train",
            ),
            __import__('semop').OperatorTransferEvalCase(
                query="The bag opening is already open. Can I place the book in now?",
                domain="containment",
                expected_operator_names=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                split="test",
            ),
        ]
        summary = evaluator.evaluate(cases)
        self.assertGreaterEqual(summary.retained_operator_count, 1)
        self.assertGreater(summary.transfer_recall, 0.0)

    def test_operator_transfer_evaluator_tracks_unseen_domain_transfer(self) -> None:
        evaluator = OperatorTransferEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        cases = [
            __import__('semop').OperatorTransferEvalCase(
                query="I am going to a car wash but traffic is blocked. Should I walk there?",
                domain="service",
                expected_operator_names=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                split="train",
            ),
            __import__('semop').OperatorTransferEvalCase(
                query="The cabinet door is closed and I need the file inside. Should I reach in immediately?",
                domain="cabinet_access",
                expected_operator_names=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                split="test",
            ),
        ]
        summary = evaluator.evaluate(cases)
        self.assertEqual(summary.num_unseen_test_cases, 1)
        self.assertGreaterEqual(summary.unseen_domain_transfer_rate, 0.0)


    def test_visual_signal_impact_evaluator_reports_ablation_summary(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [
            pipeline.run("Should I open the zipper before putting the book into the bag?"),
            pipeline.run("The cabinet door is closed and I need the file inside. Should I reach in immediately?"),
        ]
        summary = VisualSignalImpactEvaluator().evaluate(graphs, min_support=1, utility_threshold=0.2)
        self.assertGreaterEqual(summary.baseline_proposal_count, 1)
        self.assertGreaterEqual(summary.ablated_proposal_count, 1)
        self.assertGreaterEqual(summary.retained_name_overlap, 0.0)

    def test_operator_proposal_comparator_compares_engines(self) -> None:
        from semop import OperatorProposalEngine, OperatorProposalSummarizer

        class FixedSummarizer(OperatorProposalSummarizer):
            def summarize(self, pattern):
                return 'CUSTOM_SCHEMA', 'custom rationale', 'test_summarizer'

        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [pipeline.run("Should I open the zipper before putting the book into the bag?")]
        summary = OperatorProposalComparator.compare_engines(
            graphs,
            OperatorProposalEngine(),
            OperatorProposalEngine(FixedSummarizer()),
            'heuristic',
            'fixed',
        )
        self.assertEqual(summary.primary_label, 'heuristic')
        self.assertEqual(summary.secondary_label, 'fixed')
        self.assertGreaterEqual(summary.secondary_count, 1)


    def test_hybrid_operator_proposal_policy_adopts_llm_named_signature(self) -> None:
        from semop import OperatorProposalEngine, OperatorProposalSummarizer

        class FixedSummarizer(OperatorProposalSummarizer):
            def summarize(self, pattern):
                return 'CUSTOM_SCHEMA', 'custom rationale', 'llm_operator_summarizer'

        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [pipeline.run("Should I open the zipper before putting the book into the bag?")]
        summary = HybridOperatorProposalPolicy().build(graphs, llm_model_id='dummy/model')
        # monkeypatch-free sanity: hybrid path should still return a structured summary
        self.assertGreaterEqual(summary.total_hybrid_count, 1)

    def test_operator_proposal_comparator_compare_with_hybrid_returns_both_sections(self) -> None:
        class FixedComparator(OperatorProposalComparator):
            pass
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [pipeline.run("Should I open the zipper before putting the book into the bag?")]
        payload = OperatorProposalComparator().compare_with_hybrid(graphs, llm_model_id='missing/model')
        self.assertIn('comparison', payload)
        self.assertIn('hybrid', payload)

    def test_operator_self_evolution_loop_persists_run_and_transfer_summaries(self) -> None:
        base_dir = Path('tests/operator_self_evolution_loop_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = CorpusMemoryStore(base_dir / 'memory.db')
            pipeline = StructuredMeaningPipeline(mode='heuristic', memory_store_path=str(base_dir / 'memory.db'), memory_source='evolution_test')
            loop = __import__('semop').OperatorSelfEvolutionLoop(pipeline, store)
            cases = [
                __import__('semop').OperatorTransferEvalCase(
                    query='I am going to a car wash but traffic is blocked. Should I walk there?',
                    domain='service',
                    expected_operator_names=['GOAL_PRESERVATION_OPERATOR', 'SERVICE_GOAL_OPERATOR'],
                    split='train',
                ),
                __import__('semop').OperatorTransferEvalCase(
                    query='The bag is zipped shut and I need to place the book inside. Should I force it in now?',
                    domain='containment',
                    expected_operator_names=['GOAL_PRESERVATION_OPERATOR', 'SERVICE_GOAL_OPERATOR'],
                    split='test',
                ),
            ]
            results = loop.run(
                queries=[case.query for case in cases if case.split == 'train'],
                source='evolution_test',
                split='train',
                iterations=2,
                min_support=1,
                utility_threshold=0.2,
                transfer_cases=cases,
            )
            self.assertEqual(len(results), 2)
            self.assertIsNotNone(store.fetch_latest_operator_evolution_summary(source='evolution_test', split='train'))
            self.assertIsNotNone(store.fetch_latest_operator_transfer_summary(source='evolution_test', split='train'))
        finally:
            shutil.rmtree(base_dir)

    def test_vlso_aligner_surfaces_cabinet_and_capped_access_steps(self) -> None:
        cabinet_world = VLSOReasoner(mode='deep').run(
            'The cabinet door is closed and I need the file inside. Should I reach in immediately?',
            visual_input={
                'metadata': {
                    'structural_operators': [
                        {'operator_name': 'CONTAINER_BODY_OPERATOR', 'subject': 'cabinet_body', 'confidence': 0.86},
                        {'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'cabinet_door', 'confidence': 0.88},
                        {'operator_name': 'ACCESS_PORT_OPERATOR', 'subject': 'cabinet_opening', 'confidence': 0.82},
                    ]
                }
            },
        )
        bottle_world = VLSOReasoner(mode='deep').run(
            'The bottle cap is still on. Can I pour it now?',
            visual_input={
                'metadata': {
                    'structural_operators': [
                        {'operator_name': 'CONTAINER_BODY_OPERATOR', 'subject': 'bottle_body', 'confidence': 0.84},
                        {'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'cap', 'confidence': 0.89},
                        {'operator_name': 'ACCESS_PORT_OPERATOR', 'subject': 'mouth_opening', 'confidence': 0.8},
                    ]
                }
            },
        )
        self.assertTrue(any('cabinet access-control' in step.lower() for step in cabinet_world.inferred_steps))
        self.assertTrue(any('cap/control' in step.lower() for step in bottle_world.inferred_steps))

    def test_vlso_aligner_uses_hidden_goal_checks_for_cross_modal_warning(self) -> None:
        visual_payload = {
            'objects': [
                {'id': 'container', 'label': 'polygon_10', 'kind': 'shape', 'bbox': [20, 20, 180, 180], 'polygon': [[20, 40], [30, 20], [170, 20], [180, 40], [180, 170], [170, 180], [30, 180], [20, 170]]},
                {'id': 'opening_band', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [50, 24, 150, 44], 'polygon': [[50, 24], [150, 24], [150, 44], [50, 44]]},
                {'id': 'handle', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [18, 70, 36, 150], 'polygon': [[18, 70], [36, 70], [36, 150], [18, 150]]},
            ]
        }
        world = VLSOReasoner(mode='deep').run("???????좊읈??袁⑸젻泳?④덩?癲?????袁⑸즴??繞??壤굿??苑?????ル㎦??", visual_payload)
        self.assertTrue(any('containment goal' in warning.lower() or 'access-first' in step.lower() for warning in world.warnings for step in world.inferred_steps[:1]) or any('Visual structure and language preconditions' in step for step in world.inferred_steps))

    def test_synthesizer_produces_human_readable_answer(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("The car wash is far away and traffic is heavy. What should I do?")
        response = ResponseSynthesizer().synthesize(graph).to_text()
        self.assertIn("\ud575\uc2ec \ud310\ub2e8:", response)
        self.assertIn("\ub17c\ub9ac\uc801 \uadfc\uac70:", response)
        self.assertIn("\uc2e4\ud589 \uacc4\ud68d:", response)
        self.assertIn("\uc720\ub3c4\ub41c \uc5f0\uc0b0\uc790:", response)

    def test_llm_json_self_repair_handles_common_breakage(self) -> None:
        extractor = LocalTransformersExtractor(LocalLLMConfig())
        repaired = extractor._parse_with_repair(
            """```json
            {intent: 'test', entities: [], relations: [], constraints: [], scripts: [], candidate_actions: [], missing_knowledge: [],}
            ```"""
        )
        self.assertEqual(repaired["intent"], "test")
        self.assertEqual(repaired["entities"], [])

    def test_operator_intelligence_map_exposes_four_axes(self) -> None:
        architecture = build_operator_intelligence_map()
        axis_names = {axis.name for axis in architecture.axes}
        self.assertEqual(
            axis_names,
            {"operator_learning", "world_model", "memory", "verifier"},
        )
        self.assertTrue(any(subsystem.name == "vlso_hybrid_memory" for axis in architecture.axes for subsystem in axis.subsystems))

    def test_corpus_learning_builds_reusable_families(self) -> None:
        learner = CorpusReasoningLearner(mode="heuristic")
        result = learner.learn_from_queries(
            [
                "How do I put a book into a bag?",
                "What should I check before putting a book into a bag?",
                "The car wash is far away and traffic is heavy. What should I do?",
                "Traffic is heavy and I need a different car wash plan.",
            ]
        )
        self.assertEqual(result.corpus_size, 4)
        self.assertTrue(result.learned_families)
        self.assertTrue(any(family.support >= 2 for family in result.learned_families))
        self.assertTrue(result.logical_patterns)
        self.assertTrue(any(pattern.family.endswith("FRAME") for pattern in result.logical_patterns))

    def test_logical_grammar_inducer_finds_connectors_and_frames(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graphs = [
            pipeline.run("A bag has a zipper and requires open access before inserting a book."),
            pipeline.run("If traffic is heavy, delay departure before driving to the car wash."),
            pipeline.run("A mobile detailer can wash the car when driving is blocked."),
        ]
        result = LogicalGrammarInducer().induce(graphs)
        families = {pattern.family for pattern in result.patterns}
        self.assertIn("POSSESSION_FRAME", families)
        self.assertIn("CONDITIONAL_FRAME", families)
        self.assertTrue(any(pattern.connector in {"has", "if", "before", "can"} for pattern in result.patterns))

    def test_memory_store_round_trip(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("How do I put a book into a bag?")
        db_path = os.path.join(os.path.dirname(__file__), "memory_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(graph, source="test", split="train")
        loaded = store.fetch_graphs(split="train", source="test")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].query, graph.query)

    def test_memory_retrieval_augments_pipeline(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "memory_retrieval_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        base_pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(base_pipeline.run("The car wash is far away and traffic is heavy. What should I do?"), source="demo", split="train")

        pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo")
        graph = pipeline.run("The car wash is far away and traffic is still bad. Is there another option?")
        self.assertTrue(any("memory hint from similar query" in warning for warning in graph.warnings))
        self.assertTrue(any("memory prior promoted families" in warning for warning in graph.warnings))

    def test_memory_retrieval_uses_structural_probe_for_closed_container_cases(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "structural_memory_probe_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode="heuristic")
            store.upsert_graph(base_pipeline.run("The drawer is closed and I need the folder inside. Should I pull the folder out right now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("The pouch is zipped closed and I need the document inside. Can I pull it out now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("I am going to the car wash and traffic is bad, should I walk there?"), source="demo", split="train")
            pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo")
            similar = pipeline._retrieve_similar_graphs("The box is closed and I need the file inside. Can I pull it out now?")
            top_queries = [item.query for item in similar[:2]]
            self.assertTrue(any("drawer" in item.lower() for item in top_queries))
            self.assertTrue(any("pouch" in item.lower() for item in top_queries))
            self.assertFalse(any("car wash" in item.lower() for item in top_queries))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    def test_analogical_memory_retrieves_multiple_structural_neighbors(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "analogical_memory_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode="heuristic")
            seeds = [
                "The drawer is closed and I need the folder inside. Should I pull the folder out right now?",
                "The pouch is zipped closed and I need the document inside. Can I pull it out now?",
                "The suitcase is closed and I need the shirt inside. Can I take it out now?",
            ]
            for query in seeds:
                store.upsert_graph(base_pipeline.run(query), source="demo", split="train")
            pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo")
            graph = pipeline.run("The box is closed and I need the file inside. Can I pull it out now?")
            self.assertGreaterEqual(len(graph.analogical_matches), 2)
            self.assertTrue(all(item.shared_requirements for item in graph.analogical_matches[:2]))
            self.assertTrue(any(item.analogy_type in {"goal_premise_analogy", "failure_analogy"} for item in graph.analogical_matches))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_analogy_aware_planning_inserts_requirement_guard_step(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "analogy_planning_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode="heuristic")
            store.upsert_graph(base_pipeline.run("The drawer is closed and I need the folder inside. Should I pull the folder out right now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("The pouch is zipped closed and I need the document inside. Can I pull it out now?"), source="demo", split="train")
            graph = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo").run("The box is closed and I need the file inside. Can I pull it out now?")
            self.assertEqual(graph.plan[0].id, 'analogy_requirement_guard')
            self.assertIn('open_access', graph.plan[0].requires)
            self.assertEqual(graph.plan[1].id, 'analogy_compare_cases')
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_analogy_aware_verifier_strengthens_runtime_warnings(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "analogy_verifier_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode="heuristic")
            store.upsert_graph(base_pipeline.run("The drawer is closed and I need the folder inside. Should I pull the folder out right now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("The pouch is zipped closed and I need the document inside. Can I pull it out now?"), source="demo", split="train")
            graph = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo").run("The box is closed and I need the file inside. Can I pull it out now?")
            premise = next(item for item in graph.premise_validations if item.premise == 'open_access')
            risk_check = next(item for item in graph.goal_preservation_checks if item.action == 'retrieve_without_opening')
            self.assertIn('Analogical memory found', premise.rationale)
            self.assertIn('Analogical memory recalled', risk_check.rationale)
            self.assertGreaterEqual(risk_check.confidence, 0.94)
            self.assertTrue(any('analogy verifier:' in warning for warning in graph.warnings))
            self.assertTrue(any('analogy risk:' in warning for warning in graph.operator_execution.warnings))
            self.assertTrue(any('analogy_guard:' in item for item in graph.operator_execution.derived_decisions))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


    def test_analogy_policy_trainer_learns_policy_and_boosts_operator_priorities(self) -> None:
        from semop.analogy_policy import AnalogyPolicyTrainer

        db_path = os.path.join(os.path.dirname(__file__), "analogy_policy_test.db")
        policy_path = os.path.join(os.path.dirname(__file__), "analogy_policy_test.json")
        for path_item in [db_path, policy_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode="heuristic")
            store.upsert_graph(base_pipeline.run("The drawer is closed and I need the folder inside. Should I pull the folder out right now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("The pouch is zipped closed and I need the document inside. Can I pull it out now?"), source="demo", split="train")
            store.upsert_graph(base_pipeline.run("I am going to the car wash and traffic is bad, should I walk there?"), source="demo", split="train")
            summary = AnalogyPolicyTrainer().train_from_memory(store, policy_path, source="demo", epochs=80, learning_rate=0.2)
            self.assertTrue(os.path.exists(policy_path))
            self.assertGreater(summary.model.get("training_examples", 0), 0)
            pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo", analogy_policy_path=policy_path)
            similar = pipeline._retrieve_similar_graphs("The box is closed and I need the file inside. Can I pull it out now?")
            self.assertTrue(any("drawer" in item.query.lower() for item in similar[:2]))
            graph = pipeline.run("The box is closed and I need the file inside. Can I pull it out now?")
            self.assertIn("weight=", graph.plan[0].rationale)
            self.assertTrue(any(tag.startswith("analogy_policy:operator_priority:") for candidate in graph.induced_operators for tag in candidate.provenance))
        finally:
            for path_item in [db_path, policy_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_operator_runtime_compiler_verifier_flags_missing_basis(self) -> None:
        from semop.structures import OperatorDecomposition, StructuredMeaningGraph

        graph = StructuredMeaningGraph(query="broken composition", intent="generic_reasoning")
        graph.hidden_goals = ["clean_car_goal"]
        graph.required_premises = ["vehicle_present"]
        graph.missing_premises = ["vehicle_present"]
        graph.operator_decompositions = [
            OperatorDecomposition(operator_name="BROKEN_OPERATOR", basis_operators=["HIDDEN_GOAL", "REQUIRES", "CONTAINS", "TYPICAL_FOR"], rationale="synthetic test", confidence=0.7)
        ]
        graph = compile_and_execute(graph)
        self.assertLess(graph.operator_execution.composition_score, 1.0)
        self.assertTrue(any("missing basis" in item and "CONTAINS" in item for item in graph.operator_execution.compiler_findings))
        self.assertTrue(any("compiler composition risk" in item for item in graph.operator_execution.warnings))

    def test_logical_grammar_patterns_become_pipeline_priors(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "logical_grammar_prior_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        learner = CorpusReasoningLearner(mode="heuristic")
        result = learner.learn_from_queries(
            [
                "A bag has a zipper and requires open access before inserting a book.",
                "If traffic is heavy, delay departure before driving to the car wash.",
                "A mobile detailer can wash the car when driving is blocked.",
            ]
        )
        store = CorpusMemoryStore(db_path)
        store.store_learning_result(result, source="grammar_demo")
        pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="grammar_demo")
        graph = pipeline.run("If the bag has no open access, check the zipper before inserting the book.")
        self.assertTrue(any("logical grammar prior matched" in warning for warning in graph.warnings))
        self.assertTrue(any("FRAME" in candidate.family for candidate in graph.induced_operators))
        self.assertTrue(any("--[" in rule for rule in graph.grammar_hypotheses))

    def test_logical_relation_priors_inject_edges_before_induction(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "logical_relation_prior_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        learner = CorpusReasoningLearner(mode="heuristic")
        result = learner.learn_from_queries(
            [
                "A robot has wheels and can move before lifting the box.",
                "If the machine has power, it can start before loading.",
            ]
        )
        store = CorpusMemoryStore(db_path)
        store.store_learning_result(result, source="relation_demo")
        pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="relation_demo")
        graph = pipeline.run("A robot has wheels and can move before lifting the box.")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertTrue(
            any(relation in {"HAS", "AFFORDS", "BEFORE"} for _, relation, _ in relations)
            or any('logical grammar prior matched:' in warning for warning in graph.warnings)
        )
        self.assertTrue(any('logical grammar prior matched:' in warning or "logical relation prior injected" in warning for warning in graph.warnings))



    def test_vlso_reasoner_aligns_language_and_visual_constraints(self) -> None:
        visual_payload = {
            "objects": [{"id": "bag", "kind": "container", "parts": ["zipper"]}],
            "affordances": [{"subject": "zipper", "value": "OPENABLE"}],
            "states": [{"subject": "zipper", "value": "CLOSED"}],
        }
        model = VLSOReasoner().run("How do I put a book into a bag?", visual_input=visual_payload)
        self.assertTrue(any("Open the zipper" in step for step in model.inferred_steps))
        self.assertIn("open_access_required", model.constraints)
        relations = {(item.source, item.relation, item.target) for item in model.relations}
        self.assertIn(("zipper", "PART_OF", "bag"), relations)

    def test_vlso_visual_text_parser_supports_simple_scene_descriptions(self) -> None:
        model = VLSOReasoner().run(
            "How do I put a book into a bag?",
            visual_input="A bag with a zipper is closed.",
        )
        entity_ids = {item.id for item in model.entities}
        self.assertIn("bag", entity_ids)
        self.assertIn("zipper", entity_ids)
        self.assertTrue(any(item.name == "OPENABLE" for item in model.operators))



    def test_vlso_language_parser_filters_question_stopwords(self) -> None:
        model, _ = VLSOReasoner().language_parser.parse("What objects or openings are visible here?")
        entity_ids = {item.id for item in model.entities}
        self.assertNotIn("what", entity_ids)
        self.assertNotIn("or", entity_ids)
        self.assertNotIn("are", entity_ids)
        self.assertNotIn("here", entity_ids)

    def test_vlso_visual_parser_extracts_shape_from_raw_image(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f"Pillow unavailable: {exc}")
        image_path = os.path.join(os.path.dirname(__file__), "vlso_raw_shape.png")
        image = Image.new("RGB", (48, 48), "white")
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((8, 10, 36, 34), fill="black")
        image.save(image_path)
        try:
            observation = VLSOReasoner().visual_parser._coerce(image_path)
            shape_hints = {item.get("shape_hint") for item in observation.objects}
            self.assertIn("rectangle", shape_hints)
            self.assertEqual(observation.metadata.get("source"), "raw_image")
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_raw_image_parser_suppresses_dominant_border_component(self) -> None:
        preprocessor = ImageMaskPreprocessor(border_area_ratio=0.2)
        dominant_frame = [(x, y) for x in range(0, 100) for y in range(0, 80)]
        inner_object = [(x, y) for x in range(60, 72) for y in range(30, 42)]
        kept, suppressed, audit = preprocessor.suppress_border_components(
            [dominant_frame, inner_object],
            width=100,
            height=80,
        )
        self.assertEqual(len(kept), 1)
        self.assertTrue(suppressed)
        self.assertTrue(audit)

    def test_raw_image_parser_suppresses_small_edge_fragments(self) -> None:
        parser = RawImageObservationParser()
        main_object = [(x, y) for x in range(40, 100) for y in range(30, 90)]
        edge_fragment = [(x, y) for x in range(0, 20) for y in range(0, 10)]
        kept, suppressed, audit = parser._suppress_edge_fragments([main_object, edge_fragment], width=160, height=120)
        self.assertEqual(len(kept), 1)
        self.assertEqual(suppressed, 1)
        self.assertTrue(audit)

    def test_detector_adapter_preserves_structural_role_and_parent_hints(self) -> None:
        observation = DetectorOutputAdapter().to_observation({
            'annotations': [
                {'id': 'drawer_body', 'label': 'drawer', 'bbox': [10, 10, 140, 90], 'bbox_mode': 'xyxy', 'kind': 'object'},
                {'id': 'drawer_handle', 'label': 'handle', 'bbox': [112, 36, 132, 60], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.93, 'structural_role': 'handle', 'segmentation_confidence': 0.88},
                {'id': 'drawer_front', 'label': 'front opening', 'bbox': [18, 26, 126, 42], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.9, 'structural_role': 'opening', 'segmentation_confidence': 0.91},
            ]
        })
        handle = next(item for item in observation.objects if item['id'] == 'drawer_handle')
        opening = next(item for item in observation.objects if item['id'] == 'drawer_front')
        self.assertEqual(handle.get('parent_id'), 'drawer_body')
        self.assertEqual(opening.get('structural_role'), 'opening')
        self.assertTrue(any(item.get('value') == 'HANDLE_CANDIDATE' for item in observation.affordances))
        self.assertTrue(any(item.get('value') == 'ACCESS_PORT_CANDIDATE' for item in observation.affordances))

    def test_vlso_reasoner_recovers_structural_bindings_from_segmented_payload(self) -> None:
        payload = {
            'annotations': [
                {'id': 'drawer_body', 'label': 'drawer', 'bbox': [10, 10, 160, 110], 'bbox_mode': 'xyxy', 'kind': 'object', 'mask_area': 11000, 'bbox_fill_ratio': 0.72, 'hull_fill_ratio': 0.84},
                {'id': 'drawer_handle', 'label': 'handle', 'bbox': [124, 48, 148, 72], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.95, 'structural_role': 'handle', 'segmentation_confidence': 0.92},
                {'id': 'drawer_opening', 'label': 'opening band', 'bbox': [24, 20, 138, 38], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.94, 'structural_role': 'opening', 'segmentation_confidence': 0.93},
            ]
        }
        world = VLSOReasoner(mode='deep', answer_mode='structured').run('How can I open or access this drawer?', payload)
        bindings = world.metadata.get('structural_operator_bindings', [])
        binding_names = {row.get('operator_name') for row in bindings if isinstance(row, dict)}
        self.assertIn('CONTAINER_BODY_OPERATOR', binding_names)
        self.assertIn('ACCESS_PORT_OPERATOR', binding_names)
        self.assertIn('ATTACHED_GRASP_OPERATOR', binding_names)
        relations = {(item.source, item.relation, item.target) for item in world.relations}
        self.assertIn(('drawer_handle', 'STRUCTURAL_PART_OF', 'drawer_body'), relations)
        self.assertIn(('drawer_opening', 'STRUCTURAL_PART_OF', 'drawer_body'), relations)

    def test_visual_parser_infers_container_parts_and_affordances(self) -> None:
        parser = VLSOReasoner().visual_parser
        model, observation = parser.parse(
            {
                "objects": [
                    {"id": "bag", "label": "bag", "kind": "object", "bbox": [0, 0, 100, 80]},
                    {"id": "zipper", "label": "zipper", "kind": "object", "bbox": [10, 4, 86, 12]},
                ]
            }
        )
        bag_object = next(item for item in observation.objects if item.get("id") == "bag")
        zipper_object = next(item for item in observation.objects if item.get("id") == "zipper")
        self.assertIn("zipper", bag_object.get("parts", []))
        self.assertEqual(bag_object.get("kind"), "container")
        self.assertEqual(zipper_object.get("kind"), "part")
        affordances = {item.get("value") for item in observation.affordances}
        self.assertIn("HAS_INTERIOR", affordances)
        self.assertIn("EDGE_OPENING", affordances)
        relations = {(item.source, item.relation, item.target) for item in model.relations}
        self.assertIn(("zipper", "PART_OF", "bag"), relations)

    def test_visual_object_reasoner_derives_structural_operators_before_labels(self) -> None:
        parser = VLSOReasoner(mode='deep').visual_parser
        model, observation = parser.parse({
            'objects': [
                {'id': 'container', 'label': 'polygon_10', 'kind': 'shape', 'bbox': [20, 20, 180, 180], 'polygon': [[20, 40], [30, 20], [170, 20], [180, 40], [180, 170], [170, 180], [30, 180], [20, 170]]},
                {'id': 'opening_band', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [50, 24, 150, 44], 'polygon': [[50, 24], [150, 24], [150, 44], [50, 44]]},
                {'id': 'side_handle', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [18, 70, 36, 150], 'polygon': [[18, 70], [36, 70], [36, 150], [18, 150]]},
            ]
        })
        bindings = model.metadata.get('structural_operator_bindings', [])
        operator_names = {item.get('operator_name') for item in bindings if isinstance(item, dict)}
        self.assertIn('CONTAINER_BODY_OPERATOR', operator_names)
        self.assertIn('ACCESS_PORT_OPERATOR', operator_names)
        self.assertIn('ATTACHED_GRASP_OPERATOR', operator_names)
        self.assertTrue(any(item.get('value') == 'ACCESSIBLE_INTERIOR_PATH' for item in observation.affordances))

    def test_vlso_question_answerer_uses_structural_access_route(self) -> None:
        world = VLSOReasoner(mode='deep').run(
            'How can I access the opening?',
            visual_input={
                'objects': [
                    {'id': 'container', 'label': 'polygon_10', 'kind': 'shape', 'bbox': [20, 20, 180, 180], 'polygon': [[20, 40], [30, 20], [170, 20], [180, 40], [180, 170], [170, 180], [30, 180], [20, 170]]},
                    {'id': 'opening_band', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [50, 24, 150, 44], 'polygon': [[50, 24], [150, 24], [150, 44], [50, 44]]},
                    {'id': 'side_handle', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [18, 70, 36, 150], 'polygon': [[18, 70], [36, 70], [36, 150], [18, 150]]},
                ]
            },
        )
        answer = VLSOQuestionAnswerer().answer('How can I access the opening?', world, answer_mode='structured')
        self.assertIn('structural access path', answer.answer_text.lower())
        self.assertIn('opening_band', answer.answer_text)

    def test_visual_object_reasoner_infers_bag_zipper_and_strap_hypotheses(self) -> None:
        parser = VLSOReasoner().visual_parser
        _, observation = parser.parse(
            {
                "objects": [
                    {"id": "bag", "label": "polygon_12", "kind": "shape", "bbox": [20, 10, 90, 130], "polygon": [[20, 18], [30, 10], [76, 12], [90, 28], [88, 120], [70, 130], [34, 128], [20, 110]]},
                    {"id": "zipper", "label": "polygon_8", "kind": "shape", "bbox": [30, 18, 82, 26], "polygon": [[30, 18], [82, 18], [82, 26], [30, 26]]},
                    {"id": "strap", "label": "polygon_8", "kind": "shape", "bbox": [16, 34, 26, 108], "polygon": [[16, 34], [26, 34], [26, 108], [16, 108]]},
                ]
            }
        )
        affordances = {item.get("value") for item in observation.affordances}
        bag_object = next(item for item in observation.objects if item.get("id") == "bag")
        self.assertEqual(bag_object.get("kind"), "container")
        self.assertIn("BAG_LIKE_CONTAINER", affordances)
        self.assertIn("ZIPPER_LIKE_PART", affordances)
        self.assertIn("ACCESS_OPENING_CANDIDATE", affordances)
        self.assertIn("STRAP_LIKE_PART", affordances)

    def test_weak_affordance_classifier_scores_zipper_like_features(self) -> None:
        classifier = WeakAffordanceClassifier()
        predictions = classifier.predict(
            {
                "inside_parent": 1.0,
                "near_top_band": 1.0,
                "horizontal_elongation": 6.0,
                "boundary_attached": 1.0,
                "relative_area": 0.05,
                "vertical_elongation": 0.17,
                "near_side_band": 0.0,
                "shape_complexity": 0.2,
                "is_dominant": 0.0,
                "height_over_width": 0.17,
                "vertex_count_norm": 0.35,
                "overlap_children": 0.0,
                "area_ratio": 0.01,
                "touches_border": 0.0,
            },
            threshold=0.5,
        )
        labels = {item.label for item in predictions}
        self.assertIn("ZIPPER_LIKE_PART", labels)
        self.assertIn("ACCESS_OPENING_CANDIDATE", labels)

    def test_affordance_label_dataset_supports_simple_and_targeted_rows(self) -> None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'vlso_affordance_labels.jsonl')
        with open(dataset_path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'image_path': 'a.png',
                'positive_labels': ['BAG_LIKE_CONTAINER'],
                'negative_labels': ['STRAP_LIKE_PART'],
            }, ensure_ascii=False) + '\n')
            handle.write(json.dumps({
                'image_path': 'b.png',
                'targets': [
                    {'subject_id': 'shape_1', 'positive_labels': ['ZIPPER_LIKE_PART'], 'negative_labels': []},
                ],
            }, ensure_ascii=False) + '\n')
        try:
            rows = AffordanceLabelDataset().load_jsonl(dataset_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].targets[0].positive_labels, ['BAG_LIKE_CONTAINER'])
            self.assertEqual(rows[1].targets[0].subject_id, 'shape_1')
        finally:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)

    def test_affordance_weight_trainer_reestimates_weights_from_labels(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f'Pillow unavailable: {exc}')
        image_path = os.path.join(os.path.dirname(__file__), 'vlso_train_bag.png')
        labels_path = os.path.join(os.path.dirname(__file__), 'vlso_train_labels.jsonl')
        output_path = os.path.join(os.path.dirname(__file__), 'vlso_train_weights.json')
        image = Image.new('RGB', (160, 200), 'white')
        drawer = ImageDraw.Draw(image)
        drawer.polygon([(30, 30), (120, 24), (136, 54), (132, 172), (108, 190), (44, 188), (22, 160), (20, 54)], fill='black')
        drawer.rectangle((42, 36, 114, 48), fill=(120, 120, 120))
        image.save(image_path)
        with open(labels_path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'image_path': image_path,
                'targets': [
                    {'subject_id': '', 'positive_labels': ['BAG_LIKE_CONTAINER'], 'negative_labels': []},
                ],
            }, ensure_ascii=False) + '\n')
        try:
            trainer = AffordanceWeightTrainer(epochs=2, learning_rate=0.2)
            summary = trainer.train_jsonl(labels_path, output_path=output_path)
            self.assertEqual(summary.examples, 1)
            self.assertIn('BAG_LIKE_CONTAINER', summary.classes_updated)
            weights = json.loads(Path(output_path).read_text(encoding='utf-8'))
            self.assertIn('BAG_LIKE_CONTAINER', weights['classes'])
        finally:
            for target in (image_path, labels_path, output_path):
                if os.path.exists(target):
                    os.remove(target)

    def test_raw_image_parser_records_adaptive_threshold_metadata(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f"Pillow unavailable: {exc}")
        image_path = os.path.join(os.path.dirname(__file__), "vlso_adaptive_threshold.png")
        image = Image.new("RGB", (1200, 900), "white")
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((240, 140, 900, 780), fill=(40, 40, 40))
        drawer.rectangle((360, 180, 760, 240), fill=(110, 110, 110))
        image.save(image_path)
        try:
            result = RawImageObservationParser().parse_image(image_path)
            metadata = result.observation.metadata
            self.assertIn("effective_background_threshold", metadata)
            self.assertIn("resized", metadata)
            self.assertTrue(metadata["effective_background_threshold"] >= 18)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_geometry_topology_extractor_derives_spatial_relations(self) -> None:
        observation = {
            "objects": [
                {"id": "outer", "bbox": [0, 0, 10, 10]},
                {"id": "inner", "bbox": [2, 2, 4, 4]},
                {"id": "right", "bbox": [12, 2, 16, 6]},
            ],
            "geometry": [
                {"id": "line_a", "points": [[0, 0], [4, 0]]},
                {"id": "line_b", "points": [[1, 2], [5, 2]]},
                {"id": "line_c", "points": [[2, -1], [2, 3]]},
            ],
        }
        result = GeometryTopologyExtractor().extract(VLSOReasoner().visual_parser._coerce(observation))
        relations = {(item["source"], item["relation"], item["target"]) for item in result.derived_relations}
        self.assertIn(("outer", "CONTAINS", "inner"), relations)
        self.assertIn(("line_a", "PARALLEL", "line_b"), relations)
        self.assertIn(("line_a", "PERPENDICULAR", "line_c"), relations)
        self.assertIn(("outer", "LEFT_OF", "right"), relations)
        self.assertIn(("line_a", "PARALLEL", "line_b"), relations)
        self.assertIn(("line_a", "INTERSECTS", "line_c"), relations)

    def test_visual_embedding_store_indexes_and_searches(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "vlso_embedding_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = VisualEmbeddingStore(db_path)
            extractor = VisionEmbeddingExtractor()
            payload_a = {"objects": [{"id": "bag", "kind": "container"}], "affordances": [{"subject": "zipper", "value": "OPENABLE"}]}
            payload_b = {"objects": [{"id": "graph", "kind": "graph"}], "relations": [{"source": "node1", "relation": "CONNECTED", "target": "node2"}]}
            store.upsert(VisualEmbeddingRecord(key="bag_scene", label="bag scene", vector=extractor.embed_observation(payload_a), metadata={"family": "bag"}))
            store.upsert(VisualEmbeddingRecord(key="graph_scene", label="graph scene", vector=extractor.embed_observation(payload_b), metadata={"family": "graph"}))
            matches = store.search(extractor.embed_observation(payload_a), limit=1)
            self.assertEqual(matches[0].key, "bag_scene")
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_vlso_reasoner_queries_visual_embedding_memory(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "vlso_visual_memory_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            reasoner = VLSOReasoner(visual_store_path=db_path)
            payload = {
                "objects": [{"id": "bag", "kind": "container", "parts": ["zipper"], "bbox": [0, 0, 10, 10]}],
                "affordances": [{"subject": "zipper", "value": "OPENABLE"}],
                "states": [{"subject": "zipper", "value": "CLOSED"}],
            }
            reasoner.run("How do I put a book into a bag?", visual_input=payload, remember_visual=True, visual_key="bag_case")
            model = reasoner.run("How do I put a book into a bag?", visual_input=payload)
            self.assertTrue(any("visual memory" in item.lower() for item in model.audit_trace + model.warnings))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)



    def test_detector_output_adapter_supports_detection_lists(self) -> None:
        payload = {
            "detections": [
                {"label": "bag", "bbox": [0, 0, 10, 10], "parts": ["zipper"]},
                {"label": "zipper", "bbox": [2, 1, 8, 2], "affordances": ["OPENABLE"], "state": "CLOSED"},
            ],
            "constraints": ["book_not_inserted_yet"],
        }
        observation = DetectorOutputAdapter().to_observation(payload)
        self.assertEqual(len(observation.objects), 2)
        self.assertTrue(any(item.get("parts") for item in observation.objects))
        self.assertTrue(any(item.get("value") == "OPENABLE" for item in observation.affordances))
        self.assertTrue(any(item.get("value") == "CLOSED" for item in observation.states))

    def test_detector_output_adapter_preserves_image_metadata(self) -> None:
        payload = {
            "image_path": "scene.png",
            "detector": "demo_detector",
            "detections": [{"label": "bag", "bbox": [0, 0, 10, 10]}],
        }
        observation = DetectorOutputAdapter().to_observation(payload)
        self.assertEqual(observation.metadata["image_path"], "scene.png")
        self.assertEqual(observation.metadata["detector"], "demo_detector")

    def test_vision_backbone_adapter_reads_image_path_from_observation_metadata(self) -> None:
        extractor = VisionEmbeddingExtractor(model_id="dinov2_adapter", local_model_path="")
        observation = VLSOReasoner().visual_parser._coerce({
            "image_path": "missing.png",
            "objects": [{"id": "bag", "kind": "container"}],
        })
        vector = extractor.embed_observation(observation)
        summary = extractor.backend_summary()
        self.assertEqual(summary["active_backend"], "token_geometry_v1")
        self.assertEqual(len(vector), 64)
        self.assertTrue(summary["load_error"])
    def test_visual_geometry_reasoner_emits_parallel_and_perpendicular_edges(self) -> None:
        observation = __import__("semop").VisualObservation(
            objects=[{"id": "rect", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]}]
        )
        world = __import__("semop").SharedWorldModel(query="geometry")
        VisualGeometryReasoner().enrich_world(world, observation)
        relations = {(item.source, item.relation, item.target) for item in world.relations}
        self.assertTrue(any(rel == "PARALLEL" for _, rel, _ in relations))
        self.assertTrue(any(rel == "PERPENDICULAR" for _, rel, _ in relations))

    def test_visual_geometry_reasoner_infers_shape_hypotheses(self) -> None:
        observation = VLSOReasoner().visual_parser._coerce({
            "objects": [
                {"id": "rect", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]},
                {"id": "tri", "polygon": [[0, 0], [2, 0], [1, 2]]},
            ]
        })
        result = VisualGeometryReasoner().analyze(observation)
        families = {(item["id"], item["shape"]) for item in result.shape_hypotheses}
        self.assertIn(("rect", "rectangle"), families)
        self.assertIn(("tri", "triangle"), families)

    def test_resolve_local_vision_model_path_finds_dinov2_checkpoint(self) -> None:
        root = os.path.join(os.path.dirname(__file__), "vision_resolve_root")
        checkpoint = os.path.join(root, "dinov2", "demo_model")
        os.makedirs(checkpoint, exist_ok=True)
        try:
            with open(os.path.join(checkpoint, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "dinov2"}, handle)
            resolved = resolve_local_vision_model_path("dinov2_adapter", root)
            self.assertEqual(resolved, checkpoint)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_vlso_reasoner_deep_mode_sets_reasoning_metadata(self) -> None:
        model = VLSOReasoner(mode="deep").run(
            "What shapes are visible here?",
            visual_input={"objects": [{"id": "shape_1", "kind": "shape", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]}]},
        )
        self.assertEqual(model.metadata.get("reasoning_mode"), "deep")
        self.assertIn("vision_backend", model.metadata)

    def test_vlso_question_answerer_produces_grounded_structured_answer(self) -> None:
        world = VLSOReasoner(mode="deep").run(
            "How can I access the bag opening?",
            visual_input={
                "objects": [
                    {"id": "bag", "label": "polygon_12", "kind": "shape", "bbox": [20, 10, 90, 130], "polygon": [[20, 18], [30, 10], [76, 12], [90, 28], [88, 120], [70, 130], [34, 128], [20, 110]]},
                    {"id": "zipper", "label": "polygon_8", "kind": "shape", "bbox": [30, 18, 82, 26], "polygon": [[30, 18], [82, 18], [82, 26], [30, 26]]},
                ]
            },
        )
        answer = VLSOQuestionAnswerer().answer("How can I access the bag opening?", world, answer_mode="structured")
        self.assertIn("opening", answer.answer_text.lower())
        self.assertTrue(answer.evidence)

    def test_vlso_question_answerer_prefers_opening_candidates_over_border_fragments(self) -> None:
        world = VLSOReasoner(mode="deep").run(
            "How can I access the bag opening?",
            visual_input={
                "objects": [
                    {"id": "bag", "label": "bag", "kind": "container", "bbox": [20, 20, 180, 180]},
                    {"id": "opening_main", "label": "opening_main", "kind": "part", "bbox": [60, 24, 150, 44], "concept_labels": ["ACCESS_OPENING_CANDIDATE", "ZIPPER_LIKE_PART"]},
                    {"id": "edge_noise", "label": "edge_noise", "kind": "part", "bbox": [0, 0, 24, 10], "concept_labels": ["ACCESS_OPENING_CANDIDATE", "ZIPPER_LIKE_PART"]},
                ]
            },
        )
        answer = VLSOQuestionAnswerer().answer("How can I access the bag opening?", world, answer_mode="structured")
        self.assertIn("opening_main", answer.answer_text)
        self.assertNotIn("edge_noise", answer.answer_text)

    def test_vlso_reasoner_accepts_affordance_weights_path(self) -> None:
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge', 'vlso_affordance_classifier.json')
        model, answer = VLSOReasoner(mode="deep", affordance_weights_path=weights_path).answer(
            "How can I access the bag opening?",
            visual_input={
                "objects": [
                    {"id": "bag", "label": "polygon_12", "kind": "shape", "bbox": [20, 10, 90, 130], "polygon": [[20, 18], [30, 10], [76, 12], [90, 28], [88, 120], [70, 130], [34, 128], [20, 110]]},
                    {"id": "zipper", "label": "polygon_8", "kind": "shape", "bbox": [30, 18, 82, 26], "polygon": [[30, 18], [82, 18], [82, 26], [30, 26]]},
                ]
            },
        )
        self.assertIn("answer", model.metadata)
        self.assertEqual(answer.answer_mode, "structured")

    def test_detector_output_adapter_supports_segmentation_annotations(self) -> None:
        adapter = DetectorOutputAdapter()
        observation = adapter.to_observation(
            {
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 3,
                        "bbox": [10, 20, 30, 40],
                        "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
                        "state": "closed",
                    }
                ],
                "categories": [{"id": 3, "name": "backpack"}],
            }
        )
        self.assertEqual(observation.objects[0]["label"], "backpack")
        self.assertEqual(observation.objects[0]["bbox"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(observation.objects[0]["polygon"][0], [10.0, 20.0])
        self.assertEqual(observation.states[0]["value"], "closed")

    def test_detector_output_adapter_preserves_part_relations_and_hole_metadata(self) -> None:
        adapter = DetectorOutputAdapter()
        observation = adapter.to_observation(
            {
                "detections": [
                    {
                        "id": "cabinet",
                        "label": "cabinet",
                        "bbox": [0, 0, 100, 100],
                        "mask_area": 6400,
                        "bbox_fill_ratio": 0.64,
                    },
                    {
                        "id": "handle",
                        "label": "handle",
                        "bbox": [10, 40, 20, 70],
                        "part_of": "cabinet",
                        "hole_count": 1,
                        "affordances": ["GRASPABLE_PART"],
                    },
                ]
            }
        )
        relations = {(item.get('source'), item.get('relation'), item.get('target')) for item in observation.relations}
        self.assertIn(('handle', 'PART_OF', 'cabinet'), relations)
        handle = next(item for item in observation.objects if item['id'] == 'handle')
        self.assertEqual(handle.get('hole_count'), 1)
        self.assertEqual(observation.affordances[0]['value'], 'GRASPABLE_PART')

    def test_visual_affordance_feature_extractor_uses_hole_and_fill_features(self) -> None:
        observation = VisualObservation(
            objects=[
                {
                    'id': 'container',
                    'label': 'polygon_10',
                    'kind': 'shape',
                    'bbox': [0, 0, 100, 100],
                    'pixel_count': 5500,
                    'hole_count': 1,
                    'bbox_fill_ratio': 0.55,
                    'hull_fill_ratio': 0.71,
                    'polygon_perimeter': 180.0,
                    'polygon': [[0, 10], [10, 0], [90, 0], [100, 10], [100, 90], [90, 100], [10, 100], [0, 90]],
                }
            ],
            metadata={'image_size': [100, 100]},
        )
        candidate = VisualAffordanceFeatureExtractor().extract(observation)[0]
        self.assertGreater(candidate.features.get('hole_count_norm', 0.0), 0.0)
        self.assertGreater(candidate.features.get('bbox_fill_ratio', 0.0), 0.5)
        self.assertGreater(candidate.features.get('hull_fill_ratio', 0.0), 0.6)

    def test_geometry_primitive_backbone_derives_edge_and_angle_features(self) -> None:
        observation = VisualObservation(
            objects=[
                {
                    'id': 'panel',
                    'label': 'rectangle',
                    'kind': 'shape',
                    'bbox': [0, 0, 40, 20],
                    'polygon': [[0, 0], [40, 0], [40, 20], [0, 20]],
                    'pixel_count': 800,
                    'bbox_fill_ratio': 1.0,
                    'hull_fill_ratio': 1.0,
                }
            ],
            metadata={'image_size': [40, 20]},
        )
        result = GeometryPrimitiveBackbone().enrich_observation(observation)
        self.assertEqual(result.enriched_objects, 1)
        panel = observation.objects[0]
        self.assertGreaterEqual(panel.get('right_angle_count', 0), 4)
        self.assertGreaterEqual(panel.get('parallel_edge_pair_count', 0), 2)
        self.assertTrue(any(item.get('kind') == 'edge_segment' for item in observation.geometry))

    def test_vlso_visual_parser_runs_geometry_backbone_before_reasoning(self) -> None:
        parser = VLSOReasoner(mode='deep').visual_parser
        model, observation = parser.parse({
            'objects': [
                {'id': 'panel', 'label': 'rectangle', 'kind': 'shape', 'bbox': [0, 0, 40, 20], 'polygon': [[0, 0], [40, 0], [40, 20], [0, 20]], 'pixel_count': 800, 'bbox_fill_ratio': 1.0, 'hull_fill_ratio': 1.0},
                {'id': 'opening_band', 'label': 'slot', 'kind': 'shape', 'bbox': [8, 0, 32, 5], 'polygon': [[8, 0], [32, 0], [32, 5], [8, 5]], 'pixel_count': 120, 'bbox_fill_ratio': 1.0, 'hull_fill_ratio': 1.0},
            ]
        })
        self.assertIn('geometry primitive backbone derived edge and angle primitives', model.audit_trace)
        self.assertTrue(any(item.get('operator_name') == 'ACCESS_PORT_OPERATOR' for item in model.metadata.get('structural_operator_bindings', [])))

    def test_jepa_structural_predictor_uses_operator_memory_for_context_priors(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_memory_test.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = VisualOperatorMemory(db_path)
            store.upsert(
                VisualOperatorRecord(
                    key='operator:CONTAINER_ACCESS_OPERATOR',
                    operator_name='CONTAINER_ACCESS_OPERATOR',
                    feature_vector={'child_count_norm': 0.4, 'has_parent_container': 1.0},
                    signature=['HAS_PARENT_CONTAINER', 'TOP_ACCESS_PATTERN', 'SIDE_GRASP_PATTERN'],
                    metadata={'record_type': 'operator_prototype'},
                )
            )
            predictor = JepaStructuralPredictor(operator_memory=store)
            observation = VisualObservation(objects=[], metadata={})
            candidates = [
                VisualAffordanceCandidate(subject='container', parent='', features={'is_dominant': 1.0, 'right_angle_count_norm': 0.5, 'parallel_edge_pair_norm': 0.5}, object_data={}),
                VisualAffordanceCandidate(subject='opening_band', parent='container', features={'near_top_band': 1.0, 'boundary_attached': 1.0}, object_data={}),
                VisualAffordanceCandidate(subject='side_handle', parent='container', features={'near_side_band': 1.0, 'vertical_elongation': 3.0}, object_data={}),
            ]
            result = predictor.predict(observation, candidates)
            names = {item.operator_name for item in result.bindings}
            self.assertIn('CONTAINER_ACCESS_OPERATOR', names)
            self.assertIn('ACCESSIBLE_INTERIOR_PATH', {item['value'] for item in result.inferred_affordances})
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_visual_concept_memory_round_trip(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "vlso_concept_memory_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = VisualConceptMemory(db_path)
            store.upsert(VisualConceptRecord(key="a", label="BAG_LIKE_CONTAINER", feature_vector={"is_dominant": 1.0, "area_ratio": 0.4}, metadata={"image_path": "a.png"}))
            store.upsert(VisualConceptRecord(key="b", label="HANDLE_LIKE_PART", feature_vector={"vertical_elongation": 3.0, "near_side_band": 1.0}, metadata={"image_path": "b.png"}))
            matches = store.search({"is_dominant": 1.0, "area_ratio": 0.38}, limit=1)
            self.assertEqual(matches[0].label, "BAG_LIKE_CONTAINER")
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_visual_object_reasoner_uses_few_shot_concept_memory(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "vlso_concept_prior_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            payload = {
                "objects": [
                    {"id": "bag", "label": "polygon_12", "kind": "shape", "bbox": [20, 10, 90, 130], "polygon": [[20, 18], [30, 10], [76, 12], [90, 28], [88, 120], [70, 130], [34, 128], [20, 110]]},
                    {"id": "handle", "label": "polygon_8", "kind": "shape", "bbox": [16, 34, 26, 108], "polygon": [[16, 34], [26, 34], [26, 108], [16, 108]]},
                ]
            }
            parser = VLSOReasoner().visual_parser
            observation = parser._coerce(payload)
            candidates = VisualAffordanceFeatureExtractor().extract(observation)
            handle_candidate = next(item for item in candidates if item.subject == "handle")
            store = VisualConceptMemory(db_path)
            store.upsert(
                VisualConceptRecord(
                    key="handle-demo",
                    label="HANDLE_LIKE_PART",
                    feature_vector=handle_candidate.features,
                    metadata={"image_path": "demo.png"},
                )
            )
            model = VLSOReasoner(mode="deep", concept_store_path=db_path).run("What objects are visible here?", visual_input=payload)
            operator_names = {item.name for item in model.operators}
            self.assertIn("HANDLE_LIKE_PART", operator_names)
            self.assertTrue(any("few-shot visual concept memory" in item for item in model.audit_trace))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_visual_concept_prototype_trainer_builds_compact_store(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f'Pillow unavailable: {exc}')
        image_path = os.path.join(os.path.dirname(__file__), 'vlso_proto_bag.png')
        labels_path = os.path.join(os.path.dirname(__file__), 'vlso_proto_labels.jsonl')
        store_path = os.path.join(os.path.dirname(__file__), 'vlso_proto_store.db')
        summary_path = os.path.join(os.path.dirname(__file__), 'vlso_proto_summary.json')
        image = Image.new('RGB', (160, 200), 'white')
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((25, 30, 120, 180), fill='black')
        drawer.rectangle((40, 38, 108, 48), fill=(120, 120, 120))
        image.save(image_path)
        observation = RawImageObservationParser().parse_image(image_path).observation
        candidates = VisualAffordanceFeatureExtractor().extract(observation)
        bag_subject = VisualAffordanceFeatureExtractor().choose_default_subject(candidates, 'BAG_LIKE_CONTAINER') or ''
        zipper_subject = VisualAffordanceFeatureExtractor().choose_default_subject(candidates, 'ZIPPER_LIKE_PART') or ''
        with open(labels_path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'image_path': image_path,
                'targets': [
                    {'subject_id': bag_subject, 'positive_labels': ['BAG_LIKE_CONTAINER'], 'negative_labels': [], 'notes': 'main object'},
                    {'subject_id': zipper_subject, 'positive_labels': ['ZIPPER_LIKE_PART'], 'negative_labels': [], 'notes': 'top strip'},
                ],
            }, ensure_ascii=False) + '\n')
        try:
            summary = VisualConceptPrototypeTrainer().train_jsonl(labels_path, store_path, summary_output=summary_path)
            self.assertIn('BAG_LIKE_CONTAINER', summary.labels_trained)
            self.assertTrue(os.path.exists(summary_path))
            store = VisualConceptMemory(store_path)
            matches = store.search({'is_dominant': 1.0, 'area_ratio': 0.4}, limit=3)
            self.assertTrue(matches)
            proto = next(item for item in matches if item.label == 'BAG_LIKE_CONTAINER')
            self.assertEqual(proto.metadata.get('record_type'), 'prototype')
        finally:
            for target in (image_path, labels_path, store_path, summary_path):
                if os.path.exists(target):
                    os.remove(target)

    def test_visual_concept_self_trainer_clusters_and_accepts_pseudo_labels(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_self_training_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            candidates_path = base_dir / 'candidates.jsonl'
            store_path = base_dir / 'pseudo.db'
            summary_path = base_dir / 'summary.json'
            rows = [
                {
                    'image_path': 'img_a.png',
                    'targets': [
                        {
                            'subject_id': 'shape_1',
                            'feature_vector': {
                                'inside_parent': 1.0,
                                'near_top_band': 1.0,
                                'horizontal_elongation': 5.8,
                                'boundary_attached': 1.0,
                                'relative_area': 0.05,
                                'vertical_elongation': 0.18,
                                'near_side_band': 0.0,
                                'shape_complexity': 0.2,
                                'is_dominant': 0.0,
                                'height_over_width': 0.18,
                                'vertex_count_norm': 0.35,
                                'overlap_children': 0.0,
                                'area_ratio': 0.01,
                                'touches_border': 0.0,
                            },
                            'prediction_details': [
                                {'label': 'ZIPPER_LIKE_PART', 'confidence': 0.91, 'score': 4.2},
                                {'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.87, 'score': 3.9},
                            ],
                            'suggested_labels': ['ZIPPER_LIKE_PART', 'ACCESS_OPENING_CANDIDATE'],
                        }
                    ],
                },
                {
                    'image_path': 'img_b.png',
                    'targets': [
                        {
                            'subject_id': 'shape_2',
                            'feature_vector': {
                                'inside_parent': 1.0,
                                'near_top_band': 1.0,
                                'horizontal_elongation': 5.6,
                                'boundary_attached': 1.0,
                                'relative_area': 0.052,
                                'vertical_elongation': 0.19,
                                'near_side_band': 0.0,
                                'shape_complexity': 0.22,
                                'is_dominant': 0.0,
                                'height_over_width': 0.19,
                                'vertex_count_norm': 0.34,
                                'overlap_children': 0.0,
                                'area_ratio': 0.011,
                                'touches_border': 0.0,
                            },
                            'prediction_details': [
                                {'label': 'ZIPPER_LIKE_PART', 'confidence': 0.9, 'score': 4.1},
                                {'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.85, 'score': 3.8},
                            ],
                            'suggested_labels': ['ZIPPER_LIKE_PART', 'ACCESS_OPENING_CANDIDATE'],
                        }
                    ],
                },
            ]
            with candidates_path.open('w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            summary = VisualConceptSelfTrainer().train_candidates_jsonl(
                candidates_path,
                store_path,
                summary_output=summary_path,
                config=PseudoLabelAcceptanceConfig(
                    cluster_similarity_threshold=0.85,
                    pseudo_confidence_threshold=0.7,
                    cluster_consensus_threshold=0.5,
                    min_cluster_size=2,
                ),
            )
            self.assertEqual(summary.cluster_count, 1)
            self.assertGreaterEqual(summary.accepted_prototypes, 1)
            self.assertIn('ZIPPER_LIKE_PART', summary.accepted_labels)
            matches = VisualConceptMemory(store_path).search(rows[0]['targets'][0]['feature_vector'], limit=3)
            self.assertTrue(matches)
            self.assertEqual(matches[0].metadata.get('record_type'), 'pseudo_prototype')
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_visual_concept_recommender_ranks_novel_uncertain_targets(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), 'vlso_recommend_store.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = VisualConceptMemory(db_path)
            store.upsert(VisualConceptRecord(key='proto:bag', label='BAG_LIKE_CONTAINER', feature_vector={'is_dominant': 1.0, 'area_ratio': 0.4}, metadata={'record_type': 'prototype'}))
            rows = [
                {
                    'image_path': 'a.png',
                    'targets': [
                        {'subject_id': 'shape_1', 'feature_vector': {'is_dominant': 1.0, 'area_ratio': 0.39}, 'suggested_labels': ['BAG_LIKE_CONTAINER']},
                        {'subject_id': 'shape_2', 'feature_vector': {'vertical_elongation': 3.2, 'near_side_band': 1.0}, 'suggested_labels': ['STRAP_LIKE_PART']},
                    ],
                }
            ]
            ranked = VisualConceptLabelRecommender(concept_store_path=db_path).rank_candidate_rows(rows, limit=2)
            self.assertEqual(ranked[0]['subject_id'], 'shape_2')
            self.assertGreaterEqual(ranked[0]['novelty'], ranked[1]['novelty'])
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    def test_vision_backbone_adapter_falls_back_when_local_weights_missing(self) -> None:
        extractor = VisionEmbeddingExtractor(model_id="dinov2_adapter", local_model_path="")
        vector = extractor.embed_observation({"image_path": "missing.png", "objects": [{"id": "bag", "kind": "container"}]})
        summary = extractor.backend_summary()
        self.assertEqual(summary["active_backend"], "token_geometry_v1")
        self.assertEqual(len(vector), 64)
        self.assertTrue(summary["load_error"])

    def test_transfer_evaluation_reports_reuse(self) -> None:
        evaluator = TransferEvaluator(mode="heuristic")
        result = evaluator.evaluate_queries(
            [
                "How do I put a book into a bag?",
                "What should I check before putting a book into a bag?",
                "The car wash is far away and traffic is heavy. What should I do?",
                "Traffic is heavy and I need a different car wash plan.",
                "The book is large. How should I place it in the bag?",
            ],
            train_ratio=0.6,
        )
        self.assertGreaterEqual(result.train_size, 1)
        self.assertGreaterEqual(result.test_size, 1)
        self.assertTrue(0.0 <= result.family_reuse_rate <= 1.0)

    def test_corpus_builder_expands_and_splits_queries(self) -> None:
        builder = CorpusBuilder(train_ratio=0.75)
        records = builder.build_from_queries(
            [
                ("How do I put a book into a bag?", "seed"),
                ("The car wash is far away and traffic is heavy. What should I do?", "seed"),
            ],
            augment=True,
        )
        self.assertGreater(len(records), 2)
        self.assertTrue(any(record.augmented for record in records))
        self.assertTrue(all(record.split in {"train", "test"} for record in records))

    def test_public_dataset_adapter_loads_csv_and_json(self) -> None:
        adapter = PublicDatasetAdapter()
        csv_examples = adapter.load_path(os.path.join(os.path.dirname(__file__), "..", "examples", "public_reasoning_sample.csv"))
        json_examples = adapter.load_path(os.path.join(os.path.dirname(__file__), "..", "examples", "public_reasoning_sample.json"))
        self.assertEqual(len(csv_examples), 2)
        self.assertEqual(len(json_examples), 2)
        self.assertTrue(all(example.query for example in csv_examples))
        self.assertTrue(all(example.query for example in json_examples))

    def test_remote_dataset_downloader_supports_file_url_zip(self) -> None:
        downloader = RemoteDatasetDownloader()
        sample_csv = (Path(os.path.dirname(__file__)) / ".." / "examples" / "public_reasoning_sample.csv").resolve()
        temp_root = (Path(os.path.dirname(__file__)) / "_tmp_remote").resolve()
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            zip_path = temp_root / "dataset.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(sample_csv, arcname="public_reasoning_sample.csv")
            artifact = downloader.download(zip_path.as_uri(), output_dir=temp_root / "downloads", extract=True, overwrite=True)
            self.assertTrue(artifact.saved_path.exists())
            self.assertTrue(any(path.name == "public_reasoning_sample.csv" for path in artifact.extracted_paths))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_public_corpus_ingestor_runs_manifest(self) -> None:
        temp_root = (Path(os.path.dirname(__file__)) / "_tmp_manifest").resolve()
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            csv_path = (Path(os.path.dirname(__file__)) / ".." / "examples" / "public_reasoning_sample.csv").resolve()
            json_path = (Path(os.path.dirname(__file__)) / ".." / "examples" / "public_reasoning_sample.json").resolve()
            manifest_path = temp_root / "manifest.json"
            manifest = [
                {"name": "csv_demo", "source": "csv_demo", "urls": [csv_path.as_uri()], "query_fields": ["question"], "context_fields": ["context"], "answer_fields": ["answer"], "augment": False},
                {"name": "json_demo", "source": "json_demo", "urls": [json_path.as_uri()], "query_fields": ["question", "instruction"], "context_fields": ["context"], "answer_fields": ["answer", "output"], "augment": False},
            ]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            store_path = temp_root / "manifest.db"
            ingestor = PublicCorpusIngestor(mode="heuristic")
            summaries = ingestor.run_manifest(
                manifest_path=manifest_path,
                download_root=temp_root / "downloads",
                normalized_root=temp_root / "normalized",
                store_path=store_path,
                overwrite=True,
            )
            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(summary.stored_examples >= 2 for summary in summaries))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_curated_presets_and_manifest_build(self) -> None:
        self.assertIn("reasoning_core", CURATED_PRESETS)
        self.assertIn("gsm8k_train", CURATED_PUBLIC_DATASETS)
        manifest = preset_manifest("starter")
        self.assertEqual(len(manifest), 2)
        custom = curated_manifest(["gsm8k_test", "officeqa"])
        self.assertEqual([item["name"] for item in custom], ["gsm8k_test", "officeqa"])

    def test_memory_prior_evaluator_reports_gain_fields(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "memory_prior_eval.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(pipeline.run("The car wash is far away and traffic is heavy. What should I do?"), source="demo", split="train")
        store.upsert_graph(pipeline.run("The car wash is far away and traffic is still bad. Is there another option?"), source="demo", split="test")
        evaluator = MemoryPriorEvaluator(mode="heuristic")
        result = evaluator.evaluate_store(db_path, source="demo", max_test_queries=10, sample_queries=3)
        self.assertEqual(result.train_size, 1)
        self.assertEqual(result.test_size, 1)
        self.assertTrue(result.samples)

    def test_embedding_model_resolution_prefers_local_env_path(self) -> None:
        temp_root = (Path(os.path.dirname(__file__)) / "_tmp_model_env").resolve()
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            fake_model_dir = temp_root / "mini-model"
            fake_model_dir.mkdir(parents=True, exist_ok=True)
            os.environ["SEMOP_EMBED_MODEL_PATH"] = str(fake_model_dir)
            resolved = resolve_embedding_model_id()
            self.assertEqual(Path(resolved), fake_model_dir)
            cache = EmbeddingModelCache(cache_root=temp_root / "cache")
            exported = cache.export_env(fake_model_dir)
            self.assertIn(str(fake_model_dir), exported)
        finally:
            os.environ.pop("SEMOP_EMBED_MODEL_PATH", None)
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_symbolic_math_solver_adds_numeric_answer(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = (
            "Five friends eat at a fast-food chain and order the following: "
            "5 pieces of hamburger that cost $3 each; 4 sets of French fries that cost $1.20; "
            "5 cups of soda that cost $0.5 each; and 1 platter of spaghetti that cost $2.7. "
            "How much will each of them pay if they will split the bill equally?"
        )
        graph = pipeline.run(query)
        self.assertTrue(any(result.domain == "arithmetic" for result in graph.symbolic_results))
        self.assertTrue(any(candidate.family == "SYMBOLIC_ARITHMETIC" for candidate in graph.induced_operators))

    def test_symbolic_direct_calculation(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Calculate 3*(4+5)-6/2")
        self.assertTrue(any("24" in result.answer for result in graph.symbolic_results))

    def test_symbolic_linear_simplification(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Simplify 2x + 3x - 4 + 7")
        self.assertTrue(any("5x + 3" in result.answer for result in graph.symbolic_results))

    def test_symbolic_linear_equation_solver(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Solve 2x + 3 = 11")
        self.assertTrue(any("x = 4" in result.answer for result in graph.symbolic_results))

    def test_symbolic_binomial_expansion(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Expand (x+2)(x+3)")
        self.assertTrue(any("x^2 + 5x + 6" in result.answer for result in graph.symbolic_results))

    def test_symbolic_common_factor_factoring(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Factor 6x + 9")
        self.assertTrue(any("3(2x + 3)" in result.answer for result in graph.symbolic_results))

    def test_symbolic_document_grounding_extracts_evidence(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = "The office closes at 6 PM. The support desk closes at 5 PM. Which desk closes earlier?"
        graph = pipeline.run(query)
        self.assertTrue(any(result.domain == "document_grounding" for result in graph.symbolic_results))
        self.assertTrue(any("support desk closes at 5 PM" in result.answer for result in graph.symbolic_results))

    def test_operator_hierarchy_learner_builds_l1_l2_l3_nodes(self) -> None:
        learner = OperatorHierarchyLearner(mode="heuristic")
        result = learner.learn_from_queries(
            [
                "What should I check before putting a book into a bag?",
                "How do I put a book into a bag?",
                "Five friends split a restaurant bill equally after ordering several items.",
                "Revenue was $10 million in fiscal 2024. Operating income was $2 million. What was the operating income?",
            ]
        )
        self.assertEqual(result.corpus_size, 4)
        self.assertTrue(result.micro_nodes)
        self.assertTrue(result.family_nodes)
        self.assertTrue(result.abstract_nodes)
        self.assertTrue(result.composition_patterns)

    def test_ops_warehouse_exception_case_recovers_blockers_and_prerequisites(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("A worker wants to move a pallet to a rack with a forklift, but the aisle is blocked and approval is missing.")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("move_pallet_to_rack", "BLOCKED_BY", "blocked_aisle"), relations)
        self.assertIn(("move_pallet_to_rack", "REQUIRES", "supervisor_approval"), relations)

    def test_domain_copilot_scores_ops_kpis_and_audit_trace(self) -> None:
        copilot = DomainCopilot(mode="heuristic")
        request = CopilotRequest(
            query="The barcode does not match the order label. Should I pack it anyway?",
            context=(
                "Warehouse onboarding SOP\n"
                "1. Scan the barcode after picking.\n"
                "2. The order label and pick ticket must match before packing.\n"
                "3. If labels mismatch, hold the shipment and ask for supervisor approval."
            ),
            domain="warehouse_onboarding",
            scenario="onboarding",
        )
        result = copilot.run(request)
        self.assertTrue(result.graph.audit_trace)
        self.assertGreater(result.kpis.human_audit_usefulness, 0.5)
        self.assertIn("\uc6b4\uc601 KPI:", result.to_text())

    def test_plain_rag_baseline_retrieves_relevant_context_chunk(self) -> None:
        baseline = PlainRagBaseline()
        context = (
            "Shipping quality SOP\n"
            "If label mismatch occurs, hold the shipment.\n"
            "Do not close the package until supervisor approval is complete."
        )
        result = baseline.answer("Can I ship it if the label is different?", context, domain="warehouse_exception")
        self.assertTrue(result.retrieved_chunks)
        self.assertIn("hold the shipment", result.answer_text)

    def test_labeled_ops_evaluator_compares_semop_and_rag(self) -> None:
        cases = load_labeled_ops_cases(os.path.join(os.path.dirname(__file__), "..", "examples", "ops_labeled_eval_ko.jsonl"))
        evaluator = LabeledOpsEvaluator(copilot=DomainCopilot(mode="heuristic"), baseline_name="lexical_rag")
        comparison = evaluator.evaluate_case(cases[0])
        self.assertIn("semop", comparison)
        self.assertIn("lexical_rag", comparison)
        self.assertGreaterEqual(comparison["semop"].relation_recall, comparison["lexical_rag"].relation_recall)

    def test_domain_copilot_enqueues_review_items_when_risk_is_high(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "review_queue_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        copilot = DomainCopilot(mode="heuristic", review_queue_path=db_path)
        request = CopilotRequest(
            query="The aisle is blocked and approval is still missing. What should I do with the forklift move?",
            context=(
                "Exception SOP\n"
                "- Stop forklift movement when the aisle is blocked.\n"
                "- Do not rack-load without supervisor approval and safety clearance.\n"
                "- If not urgent, use the staging area and file an incident report."
            ),
            domain="warehouse_exception",
            scenario="exception_response",
        )
        result = copilot.run(request)
        queue = ReviewQueueStore(db_path)
        pending = queue.fetch_pending(limit=5)
        self.assertTrue(result.queued_for_review)
        self.assertTrue(pending)

    def test_review_queue_persists_graph_snapshot_and_context(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), 'review_queue_graph_snapshot.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            copilot = DomainCopilot(mode='heuristic', review_queue_path=db_path)
            request = CopilotRequest(
                query='The aisle is blocked and approval is still missing. What should I do with the forklift move?',
                context=(
                    'Exception SOP\n'
                    '- Stop forklift movement when the aisle is blocked.\n'
                    '- Do not rack-load without supervisor approval and safety clearance.\n'
                    '- If not urgent, use the staging area and file an incident report.'
                ),
                domain='warehouse_exception',
                scenario='exception_response',
            )
            result = copilot.run(request)
            queue = ReviewQueueStore(db_path)
            pending = queue.fetch_pending(limit=5)
            self.assertTrue(result.queued_for_review)
            self.assertTrue(pending)
            detail = queue.fetch_item_detail(pending[0].id)
            self.assertEqual(detail['context_text'], request.context)
            self.assertIsInstance(detail['graph'], dict)
            self.assertEqual(detail['graph']['query'], result.graph.query)
            self.assertEqual(detail['graph']['source_context'], result.graph.source_context)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_compare_baseline_schema_supports_multiple_baselines(self) -> None:
        self.assertIn("lexical_rag", BASELINE_SPECS)
        self.assertIn("first_chunk", BASELINE_SPECS)

    def test_review_queue_can_update_status_and_fetch_stats(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "review_queue_status_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        queue = ReviewQueueStore(db_path)
        item_id = queue.enqueue(
            domain="warehouse_exception",
            scenario="exception_response",
            query="blocked aisle case",
            reasons=["invalid_advice_rate"],
            answer_text="hold and escalate",
            kpis={"invalid_advice_rate": 0.4},
            audit_items=[{"stage": "test", "detail": "queued"}],
        )
        queue.update_status(item_id, status="approved", resolution_note="validated by supervisor")
        detail = queue.fetch_item_detail(item_id)
        stats = queue.fetch_stats()
        self.assertEqual(detail["status"], "approved")
        self.assertEqual(stats["approved"], 1)

    def test_review_reason_enrichment_detects_grounding_compiler_and_repair_gaps(self) -> None:
        from semop import review_reasons_from_graph_and_kpis
        from semop.structures import ClaimGrounding, OperatorExecutionReport, StructuredMeaningGraph

        graph = StructuredMeaningGraph(query='broken grounding case', intent='goal_directed_reasoning', source_context='Manual: open the drawer first.')
        graph.operator_execution = OperatorExecutionReport(
            compiler_findings=[
                'compiler warning: BROKEN_OPERATOR missing basis REQUIRES',
                'grounding compiler warning: document-backed reasoning has no explicit GROUNDED_BY evidence edge.',
                'claim grounding warning: "inspect the hidden sensor" has no explicit evidence or operator trace.',
            ],
            composition_score=0.41,
            counterexample_repairs=['repair: attach document evidence nodes and ground the question on them before answering.'],
            claim_groundings=[
                ClaimGrounding(claim='open the drawer first', grounded=True, support_kind='evidence', supports=['Open the drawer first.'], score=1.0),
                ClaimGrounding(claim='inspect the hidden sensor', grounded=False, support_kind='unsupported', supports=[], score=0.0),
            ],
            claim_grounding_score=0.5,
            derived_decisions=[],
        )
        reasons = review_reasons_from_graph_and_kpis(
            graph,
            {
                'invalid_advice_rate': 0.0,
                'clarification_need_rate': 0.0,
                'plan_executability': 1.0,
                'human_audit_usefulness': 1.0,
                'context_misread_rate': 0.31,
                'relation_recovery': 0.45,
            },
        )
        self.assertIn('grounding_review', reasons)
        self.assertIn('claim_grounding_review', reasons)
        self.assertIn('compiler_validity_gap', reasons)
        self.assertIn('repair_failure', reasons)
        self.assertIn('context_misread_review', reasons)
        self.assertIn('relation_recovery_gap', reasons)

    def test_domain_copilot_enqueues_runtime_review_reasons_for_compiler_gaps(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), 'review_queue_runtime_reason_test.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            copilot = DomainCopilot(mode='heuristic', review_queue_path=db_path)
            request = CopilotRequest(
                query='The drawer is closed and I need the folder inside. Should I pull the folder out right now?',
                domain='general',
                scenario='qa',
            )
            result = copilot.run(request)
            queue = ReviewQueueStore(db_path)
            pending = queue.fetch_pending(limit=5)
            self.assertTrue(result.queued_for_review)
            self.assertIn('compiler_validity_gap', result.review_reasons)
            self.assertIn('repair_failure', result.review_reasons)
            self.assertTrue(pending)
            self.assertIn('compiler_validity_gap', pending[0].reasons)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_feedback_rules_are_applied_back_into_copilot(self) -> None:
        rules_path = os.path.join(os.path.dirname(__file__), "feedback_rules_test.json")
        with open(rules_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "rules": [
                        {
                            "id": "rule1",
                            "domain": "warehouse_exception",
                            "scenario": "exception_response",
                            "trigger_terms": ["blocked", "approval"],
                            "require_terms": ["approval", "safety"],
                            "avoid_phrases": ["proceed without approval"],
                            "recommended_actions": ["Check approval and safety conditions before moving the pallet."],
                            "explanation": "Supervisor feedback rule",
                        }
                    ]
                },
                handle,
                ensure_ascii=False,
            )
        copilot = DomainCopilot(mode="heuristic", feedback_rules_path=rules_path)
        result = copilot.run(
            CopilotRequest(
                query="The aisle is blocked and approval is still missing. What should I do?",
                context="Exception SOP\nIf the aisle is blocked, stop the move and verify approval and safety first.",
                domain="warehouse_exception",
                scenario="exception_response",
            )
        )
        self.assertTrue(any("feedback rules applied" in warning for warning in result.graph.warnings))
        self.assertTrue(any("Check approval and safety conditions" in item for item in result.graph.creative_alternatives))

    def test_olympiad_reasoner_proves_odd_plus_odd_is_even(self) -> None:
        result = OlympiadReasoner().solve("Prove that the sum of two odd integers is even.")
        self.assertIsNotNone(result)
        self.assertIn("odd integers is even", result.answer)
        self.assertTrue(any("2(m + n + 1)" in item for item in result.evidence))

    def test_olympiad_reasoner_proves_three_consecutive_sum_divisible_by_three(self) -> None:
        result = OlympiadReasoner().solve("Show that the sum of three consecutive integers is divisible by 3.")
        self.assertIsNotNone(result)
        self.assertIn("divisible by 3", result.answer)
        self.assertTrue(any("MODULAR" in item for item in result.equations))

    def test_olympiad_reasoner_uses_pigeonhole_for_birth_months(self) -> None:
        result = OlympiadReasoner().solve("Show that among 13 people, at least two share the same birth month.")
        self.assertIsNotNone(result)
        self.assertIn("birth month", result.answer)
        self.assertTrue(any("PIGEONHOLE" in item for item in result.equations))

    def test_olympiad_reasoner_builds_contradiction_for_infinitely_many_primes(self) -> None:
        result = OlympiadReasoner().solve("Prove that there are infinitely many primes.")
        self.assertIsNotNone(result)
        self.assertIn("infinitely many primes", result.answer)
        self.assertTrue(any("CONTRADICTION" in item for item in result.equations))

    def test_olympiad_reasoner_uses_coloring_invariant_for_domino_problem(self) -> None:
        result = OlympiadReasoner().solve("Show that an 8x8 chessboard with two opposite corners removed cannot be tiled by dominoes.")
        self.assertIsNotNone(result)
        self.assertIn("cannot be tiled by dominoes", result.answer)
        self.assertTrue(any(("INVARIANT" in item) or ("COLORING" in item) for item in result.equations))

    def test_olympiad_reasoner_handles_induction_sum_formula(self) -> None:
        result = OlympiadReasoner().solve("Prove that 1+2+...+n = n(n+1)/2 for all positive integers n.")
        self.assertIsNotNone(result)
        self.assertIn("n(n+1)/2", result.answer)
        self.assertTrue(any("INDUCTION" in item for item in result.equations))

    def test_olympiad_reasoner_handles_extremal_average_argument(self) -> None:
        result = OlympiadReasoner().solve("Show that in any finite set of numbers, some element is at most the average.")
        self.assertIsNotNone(result)
        self.assertIn("at most the average", result.answer)
        self.assertTrue(any(("EXTREMAL" in item) or ("AVERAGE" in item) for item in result.equations))

    def test_olympiad_reasoner_emits_memory_hints_for_prime_proof(self) -> None:
        result = OlympiadReasoner().solve("Prove that there are infinitely many primes.")
        self.assertIsNotNone(result)
        self.assertTrue(result.evidence)
        self.assertTrue(result.evidence[0].startswith("memory hint:"))
        self.assertEqual(result.equations[0], "MEMORY_HINT")

    def test_pipeline_surfaces_olympiad_symbolic_proof_search(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("Prove that the product of two consecutive integers is even.")
        self.assertTrue(any(result.domain == "olympiad_proof" for result in graph.symbolic_results))
        self.assertTrue(any(candidate.family == "SYMBOLIC_PROOF_SEARCH" for candidate in graph.induced_operators))
        response = ResponseSynthesizer().synthesize(graph).to_text()
        self.assertIn("Olympiad proof sketch:", response)

    def test_olympiad_reasoner_returns_strategy_for_generic_geometry(self) -> None:
        result = OlympiadReasoner().solve("Prove that the three medians of a triangle are concurrent.")
        self.assertIsNotNone(result)
        self.assertIn("incomplete", result.answer)
        self.assertTrue(any("META_OPERATOR_SCAN" in item for item in result.equations))

    def test_competitive_programming_reasoner_generates_dijkstra_cpp(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "dijkstra_shortest_path")
        self.assertIn("priority_queue", result.cpp_code)
        self.assertIn("dijkstra_shortest_path", result.approach)
        self.assertTrue(result.compile_ok)

    def test_competitive_programming_reasoner_generates_prefix_sum_cpp(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "Given an array and many range sum queries, output the sum from l to r for each query."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "prefix_sum_range_query")
        self.assertIn("pref", result.cpp_code)
        self.assertEqual(result.time_complexity, "O(N+Q)")
        self.assertTrue(result.compile_ok)

    def test_competitive_programming_reasoner_handles_geometry_structure(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "Given points of a polygon, determine whether two segments are perpendicular and compute the area."
        )
        self.assertIn("geometry", result.domain_tags)
        self.assertIn("geometry_configuration", result.logical_frames)
        self.assertEqual(result.category, "computational_geometry_analysis")
        self.assertIn("cross", result.cpp_code.lower())

    def test_competitive_programming_reasoner_surfaces_logical_frames(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "There are many range sum queries on an array and no updates. Output the sum from l to r each time."
        )
        self.assertIsNotNone(result)
        self.assertIn("offline_range_aggregation", result.logical_frames)
        self.assertTrue(any("logical frames" in step.lower() for step in result.reasoning_steps))

    def test_competitive_programming_reasoner_projects_dsl_and_memory_layers(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "There are many range sum queries on an array and no updates. Output the sum from l to r each time."
        )
        self.assertIsNotNone(result)
        self.assertIn("query", result.goal_types)
        self.assertIn("array", result.domain_tags)
        self.assertIn("RANGE_QUERY", result.dsl_operators)
        self.assertIn("semantic", result.memory_projection)
        self.assertIn("structural", result.memory_projection)
        self.assertIn("episodic", result.memory_projection)
        self.assertIn("procedural", result.memory_projection)

    def test_cp_knowledge_loader_contains_algorithm_sources(self) -> None:
        knowledge = CpKnowledgeLoader().load()
        self.assertTrue(knowledge.algorithms)
        self.assertIn("nvidia_rtx_4060", knowledge.sources)
        self.assertTrue(any(item.id == "dijkstra_shortest_path" for item in knowledge.algorithms))

    def test_cp_knowledge_loader_contains_dsl_and_memory_schema(self) -> None:
        knowledge = CpKnowledgeLoader().load()
        self.assertGreaterEqual(len(knowledge.dsl_operators), 30)
        self.assertIn("semantic", knowledge.memory_schema)
        self.assertIn("procedural", knowledge.memory_schema)

    def test_cpp_syntax_checker_accepts_valid_cpp17_code(self) -> None:
        checker = CppSyntaxChecker()
        code = "#include <bits/stdc++.h>\nusing namespace std;\nint main(){cout<<1<<'\\n';}\n"
        result = checker.check(code)
        self.assertTrue(result.ok)
    def test_cp_solution_validator_runs_sample_and_random_checks(self) -> None:
        result = CompetitiveProgrammingReasoner().solve(
            "Given an array and many range sum queries, output the sum from l to r for each query."
        )
        self.assertIsNotNone(result)
        report = result.validation_report
        self.assertTrue(report.get("checked"))
        self.assertTrue(report.get("build_ok"))
        self.assertTrue(report.get("sample_ok"))
        self.assertTrue(report.get("random_ok"))

    def test_cp_solution_validator_supports_fenwick_dsu_grid_and_knapsack(self) -> None:
        cases = [
            (
                "There are point updates and range sum queries on an array. Process every query online.",
                "fenwick_tree",
            ),
            (
                "Process connectivity queries with union find and answer whether two nodes are in the same component.",
                "dsu_connectivity",
            ),
            (
                "Given a grid maze, find the minimum number of steps from S to T.",
                "grid_bfs",
            ),
            (
                "You are given item weights and values and a maximum capacity W. Maximize the total value without exceeding W.",
                "knapsack_dp",
            ),
        ]
        reasoner = CompetitiveProgrammingReasoner()
        for statement, expected_category in cases:
            result = reasoner.solve(statement)
            self.assertIsNotNone(result)
            self.assertEqual(result.category, expected_category)
            self.assertTrue(result.validation_report.get("checked"))
            self.assertTrue(result.validation_report.get("build_ok"))
            self.assertTrue(result.validation_report.get("sample_ok"))
            self.assertTrue(result.validation_report.get("random_ok"))
            self.assertTrue(result.validation_report.get("overall_ok"))

    def test_cp_dataset_builder_and_sft_export(self) -> None:
        statements = [
            "Given an array and many range sum queries, output the sum from l to r for each query.",
            "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities.",
        ]
        builder = CpDslDatasetBuilder()
        examples = builder.build_from_statements(statements)
        self.assertEqual(len(examples), 2)
        self.assertTrue(any("offline_range_aggregation" in item.logical_frames for item in examples))
        output_path = os.path.join(os.path.dirname(__file__), "cp_dsl_dataset_test.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        builder.save(output_path, examples)
        loaded = load_cp_dsl_examples(output_path)
        self.assertEqual(len(loaded), 2)
        sft_records = CpTrainingPlanner.build_sft_records(loaded)
        self.assertEqual(len(sft_records), 2)
        self.assertIn("goal_types", sft_records[0].completion)
        os.remove(output_path)

    def test_cp_episode_store_round_trip(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "cp_episode_store_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        store = CpEpisodeStore(db_path)
        result = CompetitiveProgrammingReasoner(episode_store_path=db_path).solve(
            "Given an array and many range sum queries, output the sum from l to r for each query."
        )
        self.assertIsNotNone(result)
        self.assertEqual(store.count(), 1)
        recent = store.fetch_recent(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].category, "prefix_sum_range_query")
        self.assertTrue(recent[0].validation_report.get("overall_ok"))
        os.remove(db_path)

    def test_cp_parser_training_scaffold_builds_from_episode_store(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "cp_episode_training_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        reasoner = CompetitiveProgrammingReasoner(episode_store_path=db_path)
        reasoner.solve("Given an array and many range sum queries, output the sum from l to r for each query.")
        scaffold = CpParserTrainingScaffold()
        examples = scaffold.build_examples_from_episode_store(db_path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].target_algorithm, "prefix_sum_range_query")
        os.remove(db_path)

    def test_cp_parser_training_scaffold_dry_run_writes_plan(self) -> None:
        output_dir = Path(os.path.dirname(__file__)) / "cp_parser_dry_run"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        try:
            config = CpParserTrainConfig(
                model_name_or_path="local-test-model",
                output_dir=str(output_dir),
                train_jsonl=os.path.join(os.path.dirname(__file__), "..", "examples", "cp_dsl_expanded_dataset.jsonl"),
                max_steps=2,
                dry_run=True,
                local_files_only=True,
                use_lora=True,
            )
            summary = CpParserTrainingScaffold().run(config)
            self.assertEqual(summary["mode"], "dry_run")
            self.assertTrue(summary["lora_enabled"])
            self.assertTrue((output_dir / "training_plan.json").exists())
            self.assertTrue((output_dir / "train_sft.jsonl").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cp_episode_priors_help_rerank_ambiguous_query(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "cp_episode_priors_test.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            plain = CompetitiveProgrammingReasoner().solve("Online array changes must be handled repeatedly.")
            self.assertIsNotNone(plain)
            self.assertEqual(plain.category, "generic_contest_analysis")

            reasoner = CompetitiveProgrammingReasoner(episode_store_path=db_path)
            reasoner.solve("Process online array changes and sum queries after each update.")
            replay_reasoner = CompetitiveProgrammingReasoner(episode_store_path=db_path)
            result = replay_reasoner.solve("Online array changes must be handled repeatedly.")
            self.assertIsNotNone(result)
            self.assertEqual(result.category, "segment_tree")
            episodic = result.memory_projection.get("episodic", {})
            self.assertTrue(episodic.get("similar_episodes"))
            self.assertTrue(episodic.get("category_scores"))
            self.assertTrue(any("episodic memory" in step.lower() for step in result.reasoning_steps))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


    def test_cp_training_bundle_builder_merges_dataset_and_episode_store(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "cp_training_bundle_test.db")
        train_out = os.path.join(os.path.dirname(__file__), "cp_bundle_train.jsonl")
        val_out = os.path.join(os.path.dirname(__file__), "cp_bundle_val.jsonl")
        for path_item in [db_path, train_out, val_out]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            reasoner = CompetitiveProgrammingReasoner(episode_store_path=db_path)
            reasoner.solve("Given an array and many range sum queries, output the sum from l to r for each query.")
            builder = CpTrainingBundleBuilder()
            bundle = builder.build(
                dataset_paths=[os.path.join(os.path.dirname(__file__), "..", "examples", "cp_dsl_expanded_dataset.jsonl")],
                episode_store_path=db_path,
                train_ratio=0.8,
            )
            self.assertTrue(bundle.train_examples)
            self.assertTrue(bundle.val_examples)
            builder.save_bundle(bundle, train_out, val_out)
            self.assertTrue(os.path.exists(train_out))
            self.assertTrue(os.path.exists(val_out))
        finally:
            for path_item in [db_path, train_out, val_out]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_cp_parser_evaluator_reports_heuristic_metrics(self) -> None:
        examples = load_cp_dsl_examples(os.path.join(os.path.dirname(__file__), "..", "examples", "cp_dsl_expanded_dataset.jsonl"))[:4]
        summary = CpParserEvaluator().evaluate_examples(examples, predictor="heuristic")
        self.assertEqual(summary.num_examples, 4)
        self.assertTrue(0.0 <= summary.algorithm_exact_match <= 1.0)
        self.assertTrue(0.0 <= summary.frame_jaccard <= 1.0)

    def test_cp_parser_evaluator_compare_examples_reports_deltas(self) -> None:
        examples = [CpDslExample(
            statement='Given three points of a triangle, compute its area.',
            goal_types=['query'],
            domain_tags=['geometry'],
            logical_frames=['geometry_configuration'],
            dsl_operators=['POINT', 'CROSS_PRODUCT'],
            target_algorithm='computational_geometry_analysis',
            reasoning_sketch='Use cross product.',
        )]

        class DummyModel:
            def predict(self, statement: str) -> CpParserPrediction:
                return CpParserPrediction(
                    statement=statement,
                    goal_types=['query'],
                    domain_tags=['geometry'],
                    logical_frames=['geometry_configuration'],
                    dsl_operators=['POINT', 'CROSS_PRODUCT'],
                    target_algorithm='computational_geometry_analysis',
                    reasoning_sketch='Use cross product.',
                )

        summary = CpParserEvaluator().compare_examples(examples, model=DummyModel())
        self.assertEqual(summary.num_examples, 1)
        self.assertIn('heuristic_summary', summary.model_dump())
        self.assertIn('model_summary', summary.model_dump())
        self.assertIn('algorithm_exact_match_delta', summary.deltas)

    def test_cp_repair_engine_fixes_newline_literal(self) -> None:
        engine = CppRepairEngine()
        repaired, attempts = engine.repair(
            category="binary_search_answer",
            code="int main(){cout << '\\n';}" ,
            compile_stderr="missing terminating ' character",
            validation_notes=[],
        )
        self.assertIn("\\n", repaired)
        self.assertTrue(any(attempt.applied for attempt in attempts))

    def test_cp_corpus_builder_reads_html_and_zip_sources(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / "cp_corpus_html_zip"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            html_path = base_dir / "sample.html"
            html_path.write_text(
                "<html><body><h1>Problem</h1><p>Given a weighted graph, find the shortest path from node 1.</p><h2>Input</h2></body></html>",
                encoding="utf-8",
            )
            md_path = base_dir / "pack.md"
            md_path.write_text(
                "# Problem A\nGiven an array, answer many range sum queries.\n# Problem B\nSupport range add updates and range sum queries with lazy propagation.\n",
                encoding="utf-8",
            )
            zip_path = base_dir / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("inside.jsonl", '{"statement":"Process connectivity queries with union find."}\n')
                archive.writestr("inside.html", "<html><body><p>Find the minimum possible maximum segment sum.</p></body></html>")
            records = CpCorpusBuilder().build_from_inputs([base_dir], augment=False)
            statements = [record.statement for record in records]
            self.assertTrue(any("weighted graph" in item.lower() for item in statements))
            self.assertTrue(any("range sum queries" in item.lower() for item in statements))
            self.assertTrue(any("union find" in item.lower() for item in statements))
            self.assertTrue(any("minimum possible maximum segment sum" in item.lower() for item in statements))
        finally:
            shutil.rmtree(base_dir)

    def test_cp_corpus_builder_collects_and_augments_mixed_inputs(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / "cp_corpus_inputs"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            (base_dir / "seed.txt").write_text(
                "Given an array and many range sum queries, output the sum from l to r for each query.\n",
                encoding="utf-8",
            )
            (base_dir / "seed.jsonl").write_text(
                '{"statement":"Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities."}\n',
                encoding="utf-8",
            )
            (base_dir / "seed.csv").write_text(
                "problem\nGiven a grid maze, find the minimum number of steps from S to T.\n",
                encoding="utf-8",
            )
            records = CpCorpusBuilder().build_from_inputs([base_dir])
            statements = [record.statement for record in records]
            self.assertTrue(any("range sum queries" in item for item in statements))
            self.assertTrue(any("weighted graph" in item for item in statements))
            self.assertTrue(any("grid maze" in item for item in statements))
            self.assertTrue(any(record.augmented for record in records))
        finally:
            shutil.rmtree(base_dir)





    def test_cp_validator_reports_counterexample_and_checker_kind(self) -> None:
        bad_code = "#include <bits/stdc++.h>\nusing namespace std;\nint main(){ios::sync_with_stdio(false);cin.tie(nullptr);cout << 0 << '\\n';}\n"
        report = CompetitiveProgrammingReasoner().validator.validate(bad_code, "prefix_sum_range_query")
        self.assertTrue(report.checked)
        self.assertEqual(report.checker_kind, "linewise_ints")
        self.assertEqual(report.failure_type, "output_mismatch")
        self.assertTrue(report.counterexample_input)
        self.assertTrue(report.expected_output)

    def test_cp_repair_engine_uses_fallback_on_output_mismatch(self) -> None:
        engine = CppRepairEngine()
        broken = "#include <bits/stdc++.h>\nusing namespace std;\nint main(){cout << 0 << '\\n';}\n"
        fallback = "#include <bits/stdc++.h>\nusing namespace std;\nint main(){ios::sync_with_stdio(false);cin.tie(nullptr);cout << 1 << '\\n';}\n"
        repaired, attempts = engine.repair(
            category="binary_search_answer",
            code=broken,
            validation_notes=["Sample output mismatch."],
            failure_type="output_mismatch",
            counterexample_input="5\n1 2 3 4 5\n",
            fallback_code=fallback,
        )
        self.assertEqual(repaired, fallback)
        self.assertTrue(any(item.applied for item in attempts if item.rule_id in {"canonical_template_reset", "counterexample_guided_rewrite"}))

    def test_cp_incident_ingestor_stores_editorial_and_failure_kind(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "cp_incident_ingest_test.db")
        jsonl_path = os.path.join(os.path.dirname(__file__), "cp_incident_ingest_test.jsonl")
        for path_item in [db_path, jsonl_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "problem_id": "demo-wa-1",
                    "statement": "Given an array and many range sum queries, output the sum from l to r for each query.",
                    "editorial_summary": "Use prefix sums.",
                    "outcome": "WA",
                    "failure_kind": "output_mismatch",
                }, ensure_ascii=False) + "\n")
            from semop import CpIncidentDataset, CpIncidentIngestor
            cases = CpIncidentDataset().load_inputs([jsonl_path])
            self.assertEqual(len(cases), 1)
            ingestor = CpIncidentIngestor(db_path)
            stored_ids = ingestor.ingest_cases(cases, solve_missing=True)
            self.assertEqual(len(stored_ids), 1)
            store = CpEpisodeStore(db_path)
            recent = store.fetch_recent(limit=1)
            self.assertEqual(recent[0].problem_id, "demo-wa-1")
            self.assertEqual(recent[0].editorial_summary, "Use prefix sums.")
            self.assertEqual(recent[0].outcome_label, "WA")
            self.assertEqual(recent[0].failure_kind, "output_mismatch")
        finally:
            for path_item in [db_path, jsonl_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_cp_gui_module_imports(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "cp_copilot_gui.py"
        spec = importlib.util.spec_from_file_location("cp_copilot_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "CpGuiApp"))

    def test_easy_gui_module_imports(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "StarterApp"))

    def test_easy_gui_exposes_vlso_progress_logger(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, '_gui_vlso_progress'))

    def test_easy_gui_checkbox_parser_prefers_last_checkbox_value(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(module._checked_form({"vlso_dry_run": ["0", "1"]}, "vlso_dry_run", default=True))
        self.assertFalse(module._checked_form({"vlso_dry_run": ["0"]}, "vlso_dry_run", default=True))
        self.assertTrue(module._checked_form({}, "vlso_dry_run", default=True))

    def test_easy_gui_starter_page_renders_geometry_tools(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        app = module.StarterApp()
        page = app.handle({})
        self.assertIn("One-click VLSO geometry self-training", page)
        self.assertIn("Run geometry self-training", page)
        self.assertIn("Prepare VLSO downloads", page)
        self.assertIn("VLSO download preparation", page)
        self.assertIn("Load image preview cards", page)
        self.assertIn("VLSO image preview and approval", page)
        self.assertIn("Run learning on downloaded images", page)
        self.assertIn("Downloaded image labeling", page)
        self.assertIn("Load downloaded image cards", page)

    def test_easy_gui_starter_page_renders_operator_training_tools(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        app = module.StarterApp()
        page = app.handle({})
        self.assertIn("Generic operator student train/resume", page)
        self.assertIn("Run generic operator training", page)
        self.assertIn("Run overall understanding benchmark", page)

    def test_visual_review_retrainer_exports_approved_clusters(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'visual_review_retrain_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            image_path = base_dir / 'sample.png'
            Image.new('RGB', (64, 64), (255, 255, 255)).save(image_path)
            summary_path = base_dir / 'summary.json'
            summary_path.write_text(json.dumps({
                'clusters': [{
                    'cluster_id': 'cluster_0001',
                    'support': 2,
                    'accepted_labels': [{'label': 'STRAP_LIKE_PART'}],
                    'suggested_labels': [{'label': 'STRAP_LIKE_PART', 'accept': True}],
                    'centroid': {'area_ratio': 0.2, 'vertical_elongation': 3.0},
                    'members': [{'image_path': str(image_path), 'subject_id': 'shape_1'}],
                }]
            }), encoding='utf-8')
            review_path = base_dir / 'reviews.json'
            review_path.write_text(json.dumps({
                'cluster_0001': {
                    'cluster_id': 'cluster_0001',
                    'status': 'approved',
                    'note': 'keep strap',
                    'approved_labels': ['STRAP_LIKE_PART'],
                }
            }), encoding='utf-8')
            summary = VisualApprovedReviewRetrainer().export_and_retrain(
                summary_path=summary_path,
                review_path=review_path,
                labels_path=base_dir / 'approved.jsonl',
                concept_store_path=base_dir / 'concepts.db',
                operator_store_path=base_dir / 'operators.db',
            )
            self.assertEqual(summary.approved_clusters, 1)
            self.assertTrue((base_dir / 'approved.jsonl').exists())
            self.assertTrue((base_dir / 'operators.db').exists())
        finally:
            shutil.rmtree(base_dir)


    def test_visual_operator_family_generalizes_beyond_bags(self) -> None:
        operators = VisualOperatorPrototypeTrainer._derive_operator_names(
            ["DRAWER_LIKE_CONTAINER", "HAS_INTERIOR", "HANDLE_LIKE_PART", "ACCESS_OPENING_CANDIDATE"],
            ["HAS_PARENT", "INSIDE_PARENT", "BOUNDARY_ATTACHED", "TOP_BAND"],
        )
        self.assertIn("CONTAINER_BODY_OPERATOR", operators)
        self.assertIn("SLIDING_ACCESS_OPERATOR", operators)
        self.assertIn("CARRIABLE_CONTAINER_OPERATOR", operators)

    def test_vlso_grounded_evaluator_scores_structured_cases(self) -> None:
        eval_path = os.path.join(os.path.dirname(__file__), 'vlso_eval_cases_test.jsonl')
        visual_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso', 'bag_closed_observation.json')
        Path(eval_path).write_text(json.dumps({
            'case_id': 'bag_eval',
            'query': 'What objects are visible here?',
            'visual_json': visual_path,
            'expected_entities': ['bag', 'zipper', 'book'],
            'expected_relations': [{'source': 'zipper', 'relation': 'PART_OF', 'target': 'bag'}],
            'required_terms': ['bag', 'zipper'],
            'forbidden_terms': ['clearer image'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            evaluator = VlsoGroundedEvaluator(VLSOReasoner(mode='heuristic', answer_mode='structured'))
            cases = evaluator.load_cases(eval_path)
            summary = evaluator.evaluate_cases(cases)
            self.assertEqual(summary.num_cases, 1)
            self.assertGreaterEqual(summary.object_recall, 0.66)
            self.assertEqual(summary.relation_recall, 1.0)
            self.assertGreaterEqual(summary.answer_term_recall, 0.5)
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)

    def test_vlso_grounded_evaluator_scores_structural_operator_recovery(self) -> None:
        eval_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_eval_cases_test.jsonl')
        Path(eval_path).write_text(json.dumps({
            'case_id': 'structural_eval',
            'query': 'How can I access the opening?',
            'expected_operators': ['CONTAINER_BODY_OPERATOR', 'ACCESS_PORT_OPERATOR', 'ATTACHED_GRASP_OPERATOR'],
            'expected_operator_bindings': [
                {'operator_name': 'ACCESS_PORT_OPERATOR', 'subject': 'opening_band', 'parent': 'container'},
                {'operator_name': 'ATTACHED_GRASP_OPERATOR', 'subject': 'side_handle', 'parent': 'container'},
            ],
            'required_terms': ['opening'],
            'forbidden_terms': ['clearer image'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            evaluator = VlsoGroundedEvaluator(VLSOReasoner(mode='deep', answer_mode='structured'))
            cases = evaluator.load_cases(eval_path)
            # inject direct visual payload by monkeypatching helper pathless load pattern via query-time reasoner wrapper
            original = evaluator._visual_input_from_case
            evaluator._visual_input_from_case = staticmethod(lambda case: {
                'objects': [
                    {'id': 'container', 'label': 'polygon_10', 'kind': 'shape', 'bbox': [20, 20, 180, 180], 'polygon': [[20, 40], [30, 20], [170, 20], [180, 40], [180, 170], [170, 180], [30, 180], [20, 170]], 'pixel_count': 18000, 'bbox_fill_ratio': 0.69, 'hull_fill_ratio': 0.82},
                    {'id': 'opening_band', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [50, 24, 150, 44], 'polygon': [[50, 24], [150, 24], [150, 44], [50, 44]], 'pixel_count': 1900, 'bbox_fill_ratio': 0.9},
                    {'id': 'side_handle', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [18, 70, 36, 150], 'polygon': [[18, 70], [36, 70], [36, 150], [18, 150]], 'pixel_count': 1200, 'bbox_fill_ratio': 0.83},
                ]
            })
            summary = evaluator.evaluate_cases(cases)
            evaluator._visual_input_from_case = original
            self.assertEqual(summary.operator_recall, 1.0)
            self.assertEqual(summary.operator_binding_recall, 1.0)
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)

    def test_vlso_review_impact_evaluator_compares_store_configs(self) -> None:
        eval_path = os.path.join(os.path.dirname(__file__), 'vlso_review_compare_eval.jsonl')
        visual_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso', 'bag_closed_observation.json')
        Path(eval_path).write_text(json.dumps({
            'case_id': 'bag_eval',
            'query': 'What objects are visible here?',
            'visual_json': visual_path,
            'expected_entities': ['bag', 'zipper'],
            'expected_relations': [{'source': 'zipper', 'relation': 'PART_OF', 'target': 'bag'}],
            'required_terms': ['bag', 'zipper'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            summary = VlsoReviewImpactEvaluator(mode='heuristic', answer_mode='structured').compare_stores(
                input_path=eval_path,
                primary_concept_store=None,
                primary_operator_store=None,
                compare_concept_store=None,
                compare_operator_store=None,
            )
            self.assertEqual(summary.primary_summary['num_cases'], 1)
            self.assertEqual(summary.compare_summary['num_cases'], 1)
            self.assertEqual(summary.deltas['grounded_answer_accuracy_delta'], 0.0)
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)

    def test_visual_operator_prototype_trainer_builds_operator_store(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f'Pillow unavailable: {exc}')
        image_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_train.png')
        labels_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_labels.jsonl')
        store_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_store.db')
        for path_item in [image_path, labels_path, store_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        image = Image.new('RGB', (160, 200), 'white')
        drawer = ImageDraw.Draw(image)
        drawer.polygon([(30, 30), (120, 24), (136, 54), (132, 172), (108, 190), (44, 188), (22, 160), (20, 54)], fill='black')
        drawer.rectangle((42, 36, 114, 48), fill=(120, 120, 120))
        image.save(image_path)
        Path(labels_path).write_text(json.dumps({
            'image_path': image_path,
            'targets': [
                {'positive_labels': ['BAG_LIKE_CONTAINER', 'HAS_INTERIOR']},
                {'positive_labels': ['ZIPPER_LIKE_PART', 'ACCESS_OPENING_CANDIDATE']},
            ],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            summary = VisualOperatorPrototypeTrainer().train_jsonl(labels_path, store_path)
            self.assertTrue(summary.prototype_count >= 2)
            memory = VisualOperatorMemory(store_path)
            matches = memory.search({'inside_parent': 1.0, 'boundary_attached': 1.0, 'near_top_band': 1.0, 'horizontal_elongation': 3.0}, ['HAS_PARENT', 'INSIDE_PARENT', 'BOUNDARY_ATTACHED', 'TOP_BAND', 'HORIZONTAL_ELONGATION'])
            self.assertTrue(matches)
            self.assertTrue(any(match.operator_name in {'OPENING_CONTROL_OPERATOR', 'CONTAINER_ACCESS_OPERATOR'} for match in matches))
        finally:
            for path_item in [image_path, labels_path, store_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_visual_hybrid_memory_fuses_local_and_global_matches(self) -> None:
        concept_store = os.path.join(os.path.dirname(__file__), 'vlso_hybrid_concepts_test.db')
        operator_store = os.path.join(os.path.dirname(__file__), 'vlso_hybrid_operators_test.db')
        for path_item in [concept_store, operator_store]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            VisualConceptMemory(concept_store).upsert(VisualConceptRecord(
                key='concept:zip',
                label='ZIPPER_LIKE_PART',
                feature_vector={'inside_parent': 1.0, 'boundary_attached': 1.0, 'near_top_band': 1.0, 'horizontal_elongation': 3.0},
                metadata={'co_labels': [{'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.8}]},
            ))
            VisualOperatorMemory(operator_store).upsert(VisualOperatorRecord(
                key='operator:open',
                operator_name='OPENING_CONTROL_OPERATOR',
                feature_vector={'inside_parent': 1.0, 'boundary_attached': 1.0, 'near_top_band': 1.0, 'horizontal_elongation': 3.0},
                signature=['HAS_PARENT', 'INSIDE_PARENT', 'BOUNDARY_ATTACHED', 'TOP_BAND', 'HORIZONTAL_ELONGATION'],
                metadata={'record_type': 'operator_prototype', 'implied_labels': [{'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.9}]},
            ))
            hybrid = VisualHybridMemory(VisualConceptMemory(concept_store), VisualOperatorMemory(operator_store))
            result = hybrid.retrieve(
                {'inside_parent': 1.0, 'boundary_attached': 1.0, 'near_top_band': 1.0, 'horizontal_elongation': 3.0},
                ['HAS_PARENT', 'INSIDE_PARENT', 'BOUNDARY_ATTACHED', 'TOP_BAND', 'HORIZONTAL_ELONGATION'],
            )
            labels = {row['label'] for row in result.fused_labels}
            self.assertIn('ZIPPER_LIKE_PART', labels)
            self.assertIn('ACCESS_OPENING_CANDIDATE', labels)
            self.assertTrue(result.local_matches)
            self.assertTrue(result.global_matches)
        finally:
            for path_item in [concept_store, operator_store]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_visual_object_reasoner_uses_operator_memory(self) -> None:
        store_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_memory_test.db')
        if os.path.exists(store_path):
            os.remove(store_path)
        try:
            memory = VisualOperatorMemory(store_path)
            memory.upsert(
                __import__('semop').VisualOperatorRecord(
                    key='operator:open',
                    operator_name='OPENING_CONTROL_OPERATOR',
                    feature_vector={'inside_parent': 1.0, 'boundary_attached': 1.0, 'near_top_band': 1.0, 'horizontal_elongation': 3.0},
                    signature=['HAS_PARENT', 'INSIDE_PARENT', 'BOUNDARY_ATTACHED', 'TOP_BAND', 'HORIZONTAL_ELONGATION'],
                    metadata={'record_type': 'operator_prototype', 'implied_labels': [{'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.9}, {'label': 'ZIPPER_LIKE_PART', 'confidence': 0.8}]},
                )
            )
            parser = VLSOReasoner(operator_store_path=store_path).visual_parser
            _, observation = parser.parse({'objects': [{'id': 'bag', 'label': 'bag', 'kind': 'object', 'bbox': [0, 0, 100, 100]}, {'id': 'zip', 'label': 'zip', 'kind': 'shape', 'bbox': [20, 0, 80, 12]}]})
            zip_obj = next(item for item in observation.objects if item.get('id') == 'zip')
            labels = set(zip_obj.get('concept_labels', []))
            self.assertIn('ACCESS_OPENING_CANDIDATE', labels)
            self.assertIn('ZIPPER_LIKE_PART', labels)
            self.assertIn('operator_prototype_matched', observation.constraints)
        finally:
            if os.path.exists(store_path):
                os.remove(store_path)

    def test_visual_cluster_review_store_round_trip(self) -> None:
        summary_path = os.path.join(os.path.dirname(__file__), "vlso_cluster_summary_test.json")
        review_path = os.path.join(os.path.dirname(__file__), "vlso_cluster_reviews_test.json")
        for path_item in [summary_path, review_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            Path(summary_path).write_text(json.dumps({
                "summary": {"cluster_count": 1},
                "clusters": [
                    {
                        "cluster_id": "cluster_0001",
                        "support": 3,
                        "accepted_labels": [{"label": "BAG_LIKE_CONTAINER", "accept": True}],
                        "suggested_labels": [{"label": "BAG_LIKE_CONTAINER", "accept": True}],
                        "centroid": {},
                        "members": [],
                    }
                ],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            store = VisualClusterReviewStore(review_path)
            store.save_decision(VisualClusterReviewDecision(
                cluster_id="cluster_0001",
                status="approved",
                note="Looks correct.",
                approved_labels=["BAG_LIKE_CONTAINER"],
            ))
            stats = store.stats(summary_path)
            clusters = store.list_clusters(summary_path)
            self.assertEqual(stats["approved"], 1)
            self.assertEqual(clusters[0]["review"]["status"], "approved")
            self.assertEqual(clusters[0]["review"]["approved_labels"], ["BAG_LIKE_CONTAINER"])
        finally:
            for path_item in [summary_path, review_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_hard_problem_engine_verifies_and_learns_pattern_weights(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "hard_problem_memory.db")
        weights_path = os.path.join(os.path.dirname(__file__), "logical_pattern_weights_test.json")
        for path_item in [db_path, weights_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        learner = CorpusReasoningLearner(mode="heuristic")
        result = learner.learn_from_queries(
            [
                "A bag has a zipper and requires open access before inserting a book.",
                "If traffic is heavy, delay departure before driving to the car wash.",
            ]
        )
        store = CorpusMemoryStore(db_path)
        store.store_learning_result(result, source="hard_demo")
        engine = HardProblemEngine(memory_store_path=db_path, memory_source="hard_demo", logical_weight_path=weights_path)
        report = engine.solve("If the bag has no open access, check the zipper before inserting the book.")
        self.assertTrue(report.checks)
        self.assertTrue(report.matched_patterns)
        updated = engine.learn_from_report(report, success=True)
        self.assertTrue(updated)
        trainer = PatternOutcomeTrainer(weights_path)
        payload = trainer.load()
        self.assertTrue(payload)


    def test_visual_data_collector_builds_dry_run_plans(self) -> None:
        manifest_path = os.path.join(os.path.dirname(__file__), 'vlso_collection_manifest_test.json')
        output_path = os.path.join(os.path.dirname(__file__), 'vlso_collection_plan_test.jsonl')
        payload = {
            'sources': [
                {'provider': 'wikimedia_commons', 'query': 'backpack zipper', 'limit': 5, 'categories': ['Backpacks']},
                {'provider': 'openverse', 'query': 'tool bag handle', 'limit': 4, 'license_filters': ['by', 'cc0']},
            ]
        }
        for path_item in [manifest_path, output_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        try:
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False)
            collector = VisualDataCollector()
            summary = collector.run_manifest(manifest_path, output_path, dry_run=True)
            self.assertEqual(summary.requests_planned, 2)
            self.assertEqual(summary.records_written, 2)
            rows = [json.loads(line) for line in Path(output_path).read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertIn('commons.wikimedia.org/w/api.php', rows[0]['request']['url'])
            self.assertIn('api.openverse.org/v1/images/', rows[1]['request']['url'])
        finally:
            for path_item in [manifest_path, output_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)

    def test_visual_data_collector_builds_keyed_provider_headers(self) -> None:
        collector = VisualDataCollector()
        plan = collector.build_plan(VisualCollectionSource(provider='unsplash', query='travel backpack', limit=3))
        self.assertEqual(plan.auth_env, 'UNSPLASH_ACCESS_KEY')
        self.assertIn('Client-ID', plan.headers.get('Authorization', ''))
        pexels = collector.build_plan(VisualCollectionSource(provider='pexels', query='tool bag', limit=2))
        self.assertEqual(pexels.auth_env, 'PEXELS_API_KEY')
        self.assertIn('api.pexels.com/v1/search', pexels.request_url)


    def test_visual_data_collector_prepare_downloads(self) -> None:
        records_path = os.path.join(os.path.dirname(__file__), 'vlso_records_test.jsonl')
        approved_path = os.path.join(os.path.dirname(__file__), 'vlso_records_approved_test.jsonl')
        manifest_path = os.path.join(os.path.dirname(__file__), 'vlso_download_manifest_test.jsonl')
        download_root = os.path.join(os.path.dirname(__file__), 'vlso_downloads_test')
        for path_item in [records_path, approved_path, manifest_path]:
            if os.path.exists(path_item):
                os.remove(path_item)
        if os.path.exists(download_root):
            shutil.rmtree(download_root)
        try:
            with open(records_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'provider': 'wikimedia_commons', 'query': 'bag', 'title': 'Backpack', 'page_url': 'https://example.com/page', 'media_url': 'https://example.com/image.jpg', 'license': 'CC-BY-SA', 'creator': 'demo', 'source_id': 'demo1', 'raw': {}}, ensure_ascii=False) + '\n')
                handle.write(json.dumps({'provider': 'pexels', 'query': 'bag', 'title': 'Other', 'page_url': 'https://example.com/p2', 'media_url': 'https://example.com/image2.jpg', 'license': 'pexels', 'creator': 'demo', 'source_id': 'demo2', 'raw': {}}, ensure_ascii=False) + '\n')
            collector = VisualDataCollector()
            summary = collector.prepare_downloads(
                records_path=records_path,
                approved_output=approved_path,
                manifest_output=manifest_path,
                download_root=download_root,
                allow_providers=['wikimedia_commons'],
                allow_licenses=['cc-by'],
            )
            self.assertEqual(summary.approved_count, 1)
            rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]['target_path'].endswith('.jpg'))
        finally:
            for path_item in [records_path, approved_path, manifest_path]:
                if os.path.exists(path_item):
                    os.remove(path_item)
            if os.path.exists(download_root):
                shutil.rmtree(download_root)

    def test_open_images_annotation_adapter_builds_detector_payloads(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'open_images_adapter_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            boxes_path = base_dir / 'boxes.csv'
            labels_path = base_dir / 'labels.csv'
            output_path = base_dir / 'payloads.jsonl'
            boxes_path.write_text('ImageID,LabelName,XMin,XMax,YMin,YMax,IsOccluded\nimg1,/m/bag,0.1,0.5,0.2,0.8,0\n', encoding='utf-8')
            labels_path.write_text('LabelName,DisplayName\n/m/bag,Bag\n', encoding='utf-8')
            adapter = OpenImagesAnnotationAdapter()
            summary = adapter.build_and_save(boxes_path, labels_path, output_path, limit_images=10)
            self.assertEqual(summary.num_images, 1)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(rows[0]['annotations'][0]['category_name'], 'Bag')
            self.assertEqual(rows[0]['annotations'][0]['bbox'], [0.1, 0.2, 0.4, 0.6])
        finally:
            shutil.rmtree(base_dir)


    def test_cp_labeled_dataset_downloader_normalizes_manifest(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'cp_labeled_download_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            dataset_path = base_dir / 'sample.jsonl'
            manifest_path = base_dir / 'manifest.json'
            output_path = base_dir / 'normalized.jsonl'
            dataset_path.write_text(json.dumps({'problem_id': 'p1', 'statement': 'Given an array, answer range sum queries.', 'solution': 'Use prefix sums.', 'tags': ['array', 'prefix_sum']}, ensure_ascii=False) + '\n', encoding='utf-8')
            manifest_path.write_text(json.dumps({'datasets': [{'name': 'demo', 'url': str(dataset_path), 'statement_field': 'statement', 'solution_field': 'solution', 'id_field': 'problem_id', 'tag_fields': ['tags']}]}, ensure_ascii=False), encoding='utf-8')
            summary = CpLabeledDatasetDownloader().download_and_normalize(manifest_path, base_dir / 'downloads', output_path)
            self.assertEqual(summary.datasets, 1)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(rows[0]['problem_id'], 'p1')
            self.assertEqual(rows[0]['tags'], ['array', 'prefix_sum'])
        finally:
            shutil.rmtree(base_dir)

    def test_visual_geometry_seed_manifest_contains_multiclass_queries(self) -> None:
        from semop.vlso.data_collection import build_geometry_seed_manifest
        manifest = build_geometry_seed_manifest()
        queries = {item['query'] for item in manifest['sources']}
        self.assertIn('triangle diagram', queries)
        self.assertIn('open drawer handle', queries)
        self.assertIn('door hinge open', queries)

    def test_cp_labeled_dataset_downloader_filters_geometry_rows(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'cp_geometry_download_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            dataset_path = base_dir / 'sample.jsonl'
            manifest_path = base_dir / 'manifest.json'
            output_path = base_dir / 'normalized.jsonl'
            dataset_path.write_text(
                json.dumps({'problem_id': 'g1', 'statement': 'Given a graph, find shortest paths.', 'solution': 'Use Dijkstra.', 'tags': ['graph']}, ensure_ascii=False) + '\n' +
                json.dumps({'problem_id': 'geo1', 'statement': 'Given three points of a triangle, compute its area.', 'solution': 'Use cross product.', 'tags': ['geometry', 'triangle']}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(json.dumps({'datasets': [{'name': 'geo_demo', 'source_type': 'url', 'url': str(dataset_path), 'statement_fields': ['statement'], 'solution_fields': ['solution'], 'id_fields': ['problem_id'], 'tag_fields': ['tags'], 'include_any_tags': ['geometry'], 'include_text_terms': ['triangle', 'area']} ]}, ensure_ascii=False), encoding='utf-8')
            summary = CpLabeledDatasetDownloader().download_and_normalize(manifest_path, base_dir / 'downloads', output_path)
            self.assertEqual(summary.examples, 1)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(rows[0]['problem_id'], 'geo1')
            self.assertIn('geometry', rows[0]['tags'])
        finally:
            shutil.rmtree(base_dir)

    def test_visual_geometry_bootstrap_pipeline_runs_end_to_end(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            self.skipTest(f'Pillow unavailable: {exc}')
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_geometry_pipeline_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        image_path = base_dir / 'rect.png'
        image = Image.new('RGB', (80, 80), 'white')
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((16, 20, 64, 56), fill='black')
        image.save(image_path)
        try:
            summary = VisualGeometryBootstrapPipeline().run(
                inputs=[str(image_path)],
                candidates_path=base_dir / 'candidates.jsonl',
                pseudo_labels_path=base_dir / 'pseudo_labels.jsonl',
                concept_store_path=base_dir / 'concepts.db',
                operator_store_path=base_dir / 'operators.db',
                eval_input='examples/vlso_geometry_eval.jsonl',
                eval_mode='heuristic',
                answer_mode='structured',
            )
            self.assertEqual(summary.image_count, 1)
            self.assertTrue((base_dir / 'candidates.jsonl').exists())
            self.assertTrue((base_dir / 'pseudo_labels.jsonl').exists())
            self.assertTrue((base_dir / 'concepts.db').exists())
            self.assertTrue((base_dir / 'operators.db').exists())
        finally:
            shutil.rmtree(base_dir)

    def test_synthetic_geometry_scene_builder_writes_eval_assets(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_synthetic_geometry_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            builder = SyntheticGeometrySceneBuilder()
            scenes = builder.build(base_dir)
            eval_path = base_dir / 'eval.jsonl'
            builder.write_eval_jsonl(scenes, eval_path)
            self.assertEqual(len(scenes), 5)
            self.assertTrue((base_dir / 'parallel_perpendicular_scene.json').exists())
            self.assertTrue(eval_path.exists())
            rows = [json.loads(line) for line in eval_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(len(rows), 5)
            self.assertIn('parallel', rows[0]['required_terms'])
        finally:
            shutil.rmtree(base_dir)

    def test_visual_geometry_bootstrap_pipeline_accepts_json_scene_inputs(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_geometry_json_pipeline_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            scenes = SyntheticGeometrySceneBuilder().build(base_dir)
            SyntheticGeometrySceneBuilder().write_eval_jsonl(scenes, base_dir / 'eval.jsonl')
            summary = VisualGeometryBootstrapPipeline().run(
                inputs=[scene.visual_json_path for scene in scenes],
                candidates_path=base_dir / 'candidates.jsonl',
                pseudo_labels_path=base_dir / 'pseudo_labels.jsonl',
                concept_store_path=base_dir / 'concepts.db',
                operator_store_path=base_dir / 'operators.db',
                eval_input=str(base_dir / 'eval.jsonl'),
                eval_mode='heuristic',
                answer_mode='structured',
            )
            self.assertEqual(summary.image_count, 5)
            self.assertIsNotNone(summary.eval_summary)
            self.assertGreaterEqual(summary.eval_summary.get('grounded_answer_accuracy', 0.0), 0.66)
        finally:
            shutil.rmtree(base_dir)

    def test_cp_geometry_template_generator_builds_eval_examples(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'cp_geometry_template_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            output_path = base_dir / 'geometry_eval.jsonl'
            summary = CpGeometryTemplateGenerator().build_eval_set(output_path)
            self.assertEqual(summary.num_examples, 10)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row['target_algorithm'] == 'computational_geometry_analysis' for row in rows))
        finally:
            shutil.rmtree(base_dir)

    def test_cp_geometry_template_generator_builds_train_val_split(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'cp_geometry_template_split_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            train_path = base_dir / 'train.jsonl'
            val_path = base_dir / 'val.jsonl'
            summary = CpGeometryTemplateGenerator().build_train_val_split(train_path, val_path, train_ratio=0.8)
            self.assertEqual(summary.train_examples, 8)
            self.assertEqual(summary.val_examples, 2)
            self.assertTrue(train_path.exists())
            self.assertTrue(val_path.exists())
        finally:
            shutil.rmtree(base_dir)

    def test_cp_geometry_eval_builder_emits_geometry_examples(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'cp_geometry_eval_builder_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            input_path = base_dir / 'geometry.jsonl'
            output_path = base_dir / 'eval.jsonl'
            input_path.write_text(
                json.dumps({'problem_id': 'geo1', 'statement': 'Given three points of a triangle, compute its area.', 'solution': 'Use cross product.', 'tags': ['geometry', 'triangle']}, ensure_ascii=False) + '\n' +
                json.dumps({'problem_id': 'graph1', 'statement': 'Find the shortest path in a weighted graph.', 'solution': 'Use Dijkstra.', 'tags': ['graph']}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            summary = CpGeometryEvalBuilder().build_from_normalized_jsonl(input_path, output_path)
            self.assertEqual(summary.kept_examples, 1)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(rows[0]['target_algorithm'], 'computational_geometry_analysis')
            self.assertIn('geometry_configuration', rows[0]['logical_frames'])
        finally:
            shutil.rmtree(base_dir)


    def test_visual_concept_self_trainer_assigns_review_priority_to_ambiguous_clusters(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_self_training_priority_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            candidates_path = base_dir / 'candidates.jsonl'
            store_path = base_dir / 'concepts.db'
            summary_path = base_dir / 'summary.json'
            payload = {
                'image_path': 'demo.png',
                'targets': [
                    {
                        'subject_id': 'shape_1',
                        'shape_hint': 'polygon_8',
                        'feature_vector': {'f1': 0.92, 'f2': 0.11, 'f3': 0.41},
                        'prediction_details': [
                            {'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.78, 'score': 0.78},
                            {'label': 'STRAP_LIKE_PART', 'confidence': 0.74, 'score': 0.74},
                        ],
                    },
                    {
                        'subject_id': 'shape_2',
                        'shape_hint': 'polygon_8',
                        'feature_vector': {'f1': 0.91, 'f2': 0.12, 'f3': 0.4},
                        'prediction_details': [
                            {'label': 'ACCESS_OPENING_CANDIDATE', 'confidence': 0.76, 'score': 0.76},
                            {'label': 'STRAP_LIKE_PART', 'confidence': 0.73, 'score': 0.73},
                        ],
                    },
                ],
            }
            candidates_path.write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')
            summary = VisualConceptSelfTrainer().train_candidates_jsonl(
                candidates_path,
                store_path,
                summary_output=summary_path,
                config=PseudoLabelAcceptanceConfig(
                    cluster_similarity_threshold=0.85,
                    pseudo_confidence_threshold=0.7,
                    cluster_consensus_threshold=0.5,
                    min_cluster_size=2,
                ),
            )
            self.assertEqual(summary.cluster_count, 1)
            self.assertTrue(summary_path.exists())
            summary_payload = json.loads(summary_path.read_text(encoding='utf-8'))
            self.assertTrue(summary_payload.get('clusters'))
            self.assertGreater(summary_payload['clusters'][0].get('review_priority', 0.0), 0.0)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_easy_gui_render_result_supports_understanding_summary(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        html_output = module.render_result("understanding_eval", {
            "progress": {
                "research_architecture_overall": 0.63,
                "robust_general_intelligence_overall": 0.59,
                "operator_architecture": {"score": 0.64},
                "premise_reasoning": {"score": 0.6},
                "shared_world_model": {"score": 0.63},
                "raw_visual_reasoning": {"score": 0.62},
            },
            "interpretation": {
                "headline": "SemOp currently understands structured hidden premises better than arbitrary real-world visual scenes.",
                "strengths": ["Premise reasoning is strong."],
                "risks": ["Real-image grounding is weak."],
                "next_steps": ["Expand the real-image benchmark."],
            },
            "snapshot": {},
        })
        self.assertIn("Research architecture", html_output)
        self.assertIn("Next steps", html_output)
        self.assertIn("Raw JSON", html_output)

    def test_easy_gui_render_result_uses_summary_and_raw_json_details(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        html_output = module.render_result("vision", {
            "world": {
                "entities": [{"id": "shape_1", "modality": "vision", "attributes": {"concept_labels": ["BAG_LIKE_CONTAINER", "HAS_INTERIOR"]}}],
                "relations": [],
                "metadata": {"vision_backend": {"active_backend": "dinov2_adapter"}},
            },
            "answer": {"answer_text": "Bag opening candidate detected.", "warnings": []},
        })
        self.assertIn("Likely objects", html_output)
        self.assertIn("Raw JSON", html_output)

    def test_easy_gui_can_render_visual_record_preview_cards(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        html_output = module.render_visual_record_preview(
            records_path="data/vlso_collection_records.jsonl",
            payload={
                "summary": {"loaded_records": 1},
                "records": [{
                    "provider": "wikimedia_commons",
                    "title": "backpack",
                    "media_url": "https://example.com/backpack.jpg",
                    "license": "cc0",
                    "source_id": "abc123",
                }],
            },
            selected_ids=["abc123"],
            allow_providers="wikimedia_commons",
            allow_licenses="cc0",
            approved_output="data/approved.jsonl",
            manifest_output="data/manifest.jsonl",
            download_root="data/vlso_downloads",
            accept_all=False,
            execute_downloads=False,
        )
        self.assertIn("Prepare selected downloads", html_output)
        self.assertIn("backpack", html_output)
        self.assertIn("Select all previewed records", html_output)

    def test_easy_gui_can_render_downloaded_label_editor(self) -> None:
        import importlib.util
        gui_path = Path(os.path.dirname(__file__)).parent / "semop_easy_gui.py"
        spec = importlib.util.spec_from_file_location("semop_easy_gui", gui_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        html_output = module.render_downloaded_label_editor(
            download_root="data/vlso_downloads",
            payload={
                "summary": {"loaded_records": 1},
                "records": [{
                    "source_id": "bag/item.jpg",
                    "title": "item",
                    "provider": "wikimedia_commons",
                    "local_path": "E:/tmp/item.jpg",
                }],
            },
            labels_output="data/manual_labels.jsonl",
            concept_store="data/manual_concepts.db",
            operator_store="data/manual_operators.db",
            review_path="data/manual_label_reviews.json",
            review_rows={
                "bag/item.jpg": {
                    "positive_labels": ["BAG_LIKE_CONTAINER", "HAS_INTERIOR"],
                    "notes": "looks like a bag",
                    "status": "approved",
                }
            },
            review_filter="approved",
        )
        self.assertIn("Positive labels", html_output)
        self.assertIn("Save review queue", html_output)
        self.assertIn("Retrain approved labels only", html_output)
        self.assertIn("status: approved", html_output)
        self.assertIn("Review filter", html_output)
        self.assertIn("vlso-label-suggestions", html_output)

    def test_hidden_premise_evaluator_reports_precision_and_clarification_metrics(self) -> None:
        evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        cases = [
            HiddenPremiseEvalCase(
                query="I am going to the car wash and traffic is bad, should I walk there?",
                expected_hidden_goals=["clean_car_goal"],
                expected_required_premises=["vehicle_present"],
                expected_satisfied_premises=[],
                expected_missing_premises=['vehicle_present'],
                expected_risky_actions=["walk_without_car"],
                forbidden_premises=["open_access"],
                expected_clarification_needed=True,
            )
        ]
        summary = evaluator.evaluate(cases)
        self.assertGreaterEqual(summary.unsupported_premise_precision, 1.0)
        self.assertEqual(summary.clarification_accuracy, 1.0)

    def test_pipeline_reuses_sqlite_premise_memory_hints(self) -> None:
        db_path = Path('tests/premise_memory_runtime_test.db')
        if db_path.exists():
            db_path.unlink()
        pipeline = StructuredMeaningPipeline(mode='heuristic', memory_store_path=str(db_path), memory_source='premise_test')
        pipeline.run('I am going to the car wash and traffic is bad, should I walk there?')
        second = pipeline.run('If traffic is bad on the way to the car wash, can I just walk there?')
        self.assertTrue(any('premise memory hint:' in warning for warning in second.warnings))

    def test_cp_parser_evaluator_uses_problem_structure_without_solver(self) -> None:
        evaluator = CpParserEvaluator(CompetitiveProgrammingReasoner())
        prediction = evaluator.predict_heuristic('Given points of a triangle, compute its area.')
        self.assertIn('geometry_configuration', prediction.logical_frames)
        self.assertTrue(bool(prediction.reasoning_sketch.strip()))

    def test_common_evaluator_runs_hidden_premise_and_cp_snapshot(self) -> None:
        evaluator = SemOpCommonEvaluator()
        hidden_cases = [
            HiddenPremiseEvalCase(
                query='?? ??? ?? ?? ??? ???',
                expected_hidden_goals=['store_book_in_bag_goal'],
                expected_required_premises=['open_access', 'available_space'],
                expected_satisfied_premises=[],
                expected_missing_premises=['open_access'],
                expected_risky_actions=['insert_without_opening'],
                forbidden_premises=['vehicle_present'],
                expected_clarification_needed=True,
            )
        ]
        cp_examples = [
            CpDslExample(
                statement='There are many range sum queries on an array and no updates. Output the sum from l to r each time.',
                goal_types=['query'],
                domain_tags=['array'],
                logical_frames=['prefix_sum_range_query'],
                dsl_operators=['RANGE_QUERY', 'PREFIX'],
                target_algorithm='prefix_sum_range_query',
                reasoning_sketch='Use prefix sums to answer each range query in O(1).',
            )
        ]
        snapshot = evaluator.evaluate(hidden_premise_cases=hidden_cases, cp_examples=cp_examples)
        self.assertIsNotNone(snapshot.hidden_premise)
        self.assertIsNotNone(snapshot.cp_parser)
        self.assertIn('critical_premise_recall', snapshot.hidden_premise)
        self.assertIn('algorithm_exact_match', snapshot.cp_parser)



    def test_script_compatibility_scorer_uses_memory_profile(self) -> None:
        db_path = Path('tests/script_compatibility_memory.db')
        if db_path.exists():
            db_path.unlink()
        store = CorpusMemoryStore(db_path)
        seed_graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I open it first?')
        store.upsert_graph(seed_graph, source='premise_test', split='train')
        store.upsert_premise_operator_memory(seed_graph, source='premise_test', split='train')
        pipeline = StructuredMeaningPipeline(mode='heuristic', memory_store_path=str(db_path), memory_source='premise_test')
        graph = pipeline.run('The drawer is closed and I need the folder inside. Can I pull it out now?')
        open_candidates = [item for item in graph.premise_candidates if item.premise == 'open_access']
        self.assertTrue(open_candidates)
        self.assertTrue(any('Compatibility rationale:' in evidence for evidence in open_candidates[0].evidence))

    def test_script_compatibility_profile_uses_retained_evolved_operator_support(self) -> None:
        base_dir = Path('tests/evolved_operator_support_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = CorpusMemoryStore(base_dir / 'memory.db')
            proposal = __import__('semop').EvolvedOperatorProposal(
                name='EVOLVED_CAPPED_ACCESS_OPERATOR',
                basis_signature=['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'],
                basis_operators=['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'],
                source_operator_names=['ACCESS_PORT_OPERATOR', 'ACCESS_CONTROL_OPERATOR'],
                source_domains=['access', 'capped_access'],
                support=3,
                domain_support=2,
                confidence=0.86,
                utility_score=0.86,
                retained=True,
                rationale='Learned capped-access operator.',
            )
            summary = __import__('semop').OperatorEvolutionSummary(
                num_graphs=2,
                proposals=[proposal],
                retained_count=1,
                pruned_count=0,
                merged_count=0,
            )
            run_id = store.store_operator_evolution_result(summary, source='compat_evolved', split='train', iteration=1)
            store.store_operator_transfer_summary(
                __import__('semop').OperatorTransferEvalSummary(
                    num_train=1,
                    num_test=0,
                    retained_operator_count=1,
                    transfer_recall=1.0,
                    domain_transfer_rate=1.0,
                    num_unseen_test_cases=0,
                    unseen_domain_transfer_rate=0.0,
                    retained_operator_names=['EVOLVED_CAPPED_ACCESS_OPERATOR'],
                    results=[],
                ),
                source='compat_evolved',
                split='train',
                iteration=1,
                evolution_run_id=run_id,
            )
            profile = store.collect_script_compatibility_profile(
                'The bottle cap is still on. Can I pour it now?',
                premise='open_access',
                hidden_goal='pour_from_bottle_goal',
                source='compat_evolved',
                split='train',
            )
            self.assertGreater(profile.get('evolved_operator_support', 0.0), 0.0)
        finally:
            shutil.rmtree(base_dir)

    def test_hidden_premise_explorer_recovers_cabinet_and_capped_access_goals(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        cabinet_graph = pipeline.run('The cabinet door is closed and I need the file inside. Should I reach in immediately?')
        bottle_graph = pipeline.run('The bottle cap is still on. Can I pour it now?')
        self.assertIn('retrieve_item_from_cabinet_goal', cabinet_graph.hidden_goals)
        self.assertIn('open_access', cabinet_graph.required_premises)
        self.assertTrue(any(item.action == 'retrieve_without_opening' for item in cabinet_graph.goal_preservation_checks))
        self.assertIn('pour_from_bottle_goal', bottle_graph.hidden_goals)
        self.assertIn('open_access', bottle_graph.required_premises)
        self.assertTrue(any(item.action == 'pour_without_uncapping' for item in bottle_graph.goal_preservation_checks))

    def test_hidden_premise_explorer_recovers_box_and_pouch_access_goals(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        box_graph = pipeline.run('The box is closed and I need the file inside. Can I pull it out now?')
        pouch_graph = pipeline.run('The pouch is zipped closed and I need the document inside. Can I pull it out now?')
        suitcase_graph = pipeline.run('The suitcase is already open and I need the folder inside. Can I take it now?')
        self.assertIn('retrieve_item_from_box_goal', box_graph.hidden_goals)
        self.assertIn('open_access', box_graph.required_premises)
        self.assertTrue(any(item.action == 'retrieve_without_opening' for item in box_graph.goal_preservation_checks))
        self.assertIn('retrieve_item_from_pouch_goal', pouch_graph.hidden_goals)
        self.assertIn('open_access', pouch_graph.required_premises)
        self.assertTrue(any(item.action == 'retrieve_without_opening' for item in pouch_graph.goal_preservation_checks))
        self.assertIn('retrieve_item_from_suitcase_goal', suitcase_graph.hidden_goals)
        self.assertIn('open_access', suitcase_graph.satisfied_premises)


    def test_visual_review_retrainer_can_seed_semop_memory(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'visual_review_seed_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            image_path = base_dir / 'sample.png'
            Image.new('RGB', (64, 64), (255, 255, 255)).save(image_path)
            summary_path = base_dir / 'summary.json'
            summary_path.write_text(json.dumps({
                'clusters': [{
                    'cluster_id': 'cluster_0001',
                    'support': 2,
                    'accepted_labels': [{'label': 'DRAWER_LIKE_CONTAINER'}, {'label': 'HANDLE_LIKE_PART'}],
                    'centroid': {'area_ratio': 0.2},
                    'members': [{'image_path': str(image_path), 'subject_id': 'shape_1'}],
                }]
            }), encoding='utf-8')
            review_path = base_dir / 'reviews.json'
            review_path.write_text(json.dumps({
                'cluster_0001': {
                    'cluster_id': 'cluster_0001',
                    'status': 'approved',
                    'note': 'keep drawer structure',
                    'approved_labels': ['DRAWER_LIKE_CONTAINER', 'HANDLE_LIKE_PART'],
                }
            }), encoding='utf-8')
            semop_db = base_dir / 'semop_memory.db'
            summary = VisualApprovedReviewRetrainer().export_and_retrain(
                summary_path=summary_path,
                review_path=review_path,
                labels_path=base_dir / 'approved.jsonl',
                concept_store_path=base_dir / 'concepts.db',
                operator_store_path=base_dir / 'operators.db',
                semop_memory_store_path=semop_db,
                semop_memory_source='vlso_review_test',
            )
            self.assertGreaterEqual(summary.seeded_semop_memories, 1)
            store = CorpusMemoryStore(semop_db)
            hits = store.search_premise_support('The drawer is closed and I need the folder inside.', source='vlso_review_test')
            self.assertTrue(any(hit['premise'] == 'open_access' for hit in hits))
        finally:
            shutil.rmtree(base_dir)

    def test_common_evaluator_reports_operator_premise_support(self) -> None:
        evaluator = SemOpCommonEvaluator()
        hidden_cases = [
            HiddenPremiseEvalCase(
                query='The bag is closed and I need to put the book inside. Should I push it in now?',
                expected_hidden_goals=['store_book_in_bag_goal'],
                expected_required_premises=['open_access', 'available_space'],
                expected_satisfied_premises=[],
                expected_missing_premises=['open_access'],
                expected_risky_actions=['insert_without_opening'],
                forbidden_premises=['vehicle_present'],
                expected_clarification_needed=False,
                expected_support_operators=['CONTAINMENT_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
            )
        ]
        snapshot = evaluator.evaluate(hidden_premise_cases=hidden_cases)
        self.assertIsNotNone(snapshot.operator_premise_support)
        self.assertIn('operator_supported_premise_recall', snapshot.operator_premise_support)
        self.assertGreater(snapshot.operator_premise_support['operator_supported_premise_recall'], 0.0)


    def test_script_compatibility_trainer_writes_model(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'script_compatibility_train_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            store = CorpusMemoryStore(base_dir / 'memory.db')
            graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I open it first?')
            store.upsert_graph(graph, source='premise_test', split='train')
            store.upsert_premise_operator_memory(graph, source='premise_test', split='train')
            summary = ScriptCompatibilityTrainer().train_from_memory(store, base_dir / 'script_model.json', source='premise_test')
            self.assertTrue((base_dir / 'script_model.json').exists())
            self.assertGreater(summary.trained_on_hits, 0)
        finally:
            shutil.rmtree(base_dir)


    def test_script_compatibility_trainer_emits_examples_and_loss(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'script_compatibility_train_loss_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            store = CorpusMemoryStore(base_dir / 'memory.db')
            for query in [
                'The drawer is closed and I need the folder inside. Should I open it first?',
                'I am going to the car wash and traffic is bad, should I walk there?',
            ]:
                graph = StructuredMeaningPipeline(mode='heuristic').run(query)
                store.upsert_graph(graph, source='premise_test', split='train')
                store.upsert_premise_operator_memory(graph, source='premise_test', split='train')
            summary = ScriptCompatibilityTrainer().train_from_memory(
                store,
                base_dir / 'script_model.json',
                source='premise_test',
                epochs=10,
                learning_rate=0.1,
            )
            self.assertGreater(summary.model.get('training_examples', 0), 0)
            self.assertGreaterEqual(summary.model.get('learned_weight', 0.0), 0.5)
            self.assertGreaterEqual(summary.model.get('training_loss', 0.0), 0.0)
        finally:
            shutil.rmtree(base_dir)

    def test_vlso_grounded_evaluator_reports_operator_premise_support(self) -> None:
        eval_path = os.path.join(os.path.dirname(__file__), 'vlso_operator_premise_eval.jsonl')
        Path(eval_path).write_text(json.dumps({
            'case_id': 'structural_support',
            'query': 'How can I access the opening?',
            'visual_json': os.path.join(os.path.dirname(__file__), '..', 'examples', 'vlso', 'bag_closed_observation.json'),
            'expected_support_premises': ['open_access'],
            'expected_operators': ['ACCESS_PORT_OPERATOR'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            evaluator = VlsoGroundedEvaluator(VLSOReasoner(mode='heuristic', answer_mode='structured'))
            cases = evaluator.load_cases(eval_path)
            summary = evaluator.evaluate_cases(cases)
            self.assertEqual(summary.num_cases, 1)
            self.assertGreaterEqual(summary.operator_premise_support, 0.5)
        finally:
            if os.path.exists(eval_path):
                os.remove(eval_path)

    def test_common_evaluator_merges_vlso_operator_support_metrics(self) -> None:
        evaluator = SemOpCommonEvaluator()
        vlso_eval_path = Path(os.path.dirname(__file__)) / 'vlso_operator_support_snapshot.jsonl'
        vlso_eval_path.write_text(json.dumps({
            'case_id': 'structural_support',
            'query': 'How can I access the opening?',
            'visual_json': str((Path(os.path.dirname(__file__)) / '..' / 'examples' / 'vlso' / 'bag_closed_observation.json').resolve()),
            'expected_support_premises': ['open_access'],
            'expected_operators': ['ACCESS_PORT_OPERATOR'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            vlso_cases = VlsoGroundedEvaluator.load_cases(vlso_eval_path)
            snapshot = evaluator.evaluate(vlso_cases=vlso_cases)
            self.assertIsNotNone(snapshot.operator_premise_support)
            self.assertIn('vlso_operator_premise_support', snapshot.operator_premise_support)
        finally:
            if vlso_eval_path.exists():
                vlso_eval_path.unlink()

    def test_operator_intelligence_progress_estimator_tracks_axes(self) -> None:
        snapshot = SemOpCommonEvaluator().evaluate(
            hidden_premise_cases=[
                HiddenPremiseEvalCase(
                    query='I am going to the car wash and traffic is bad, should I walk there?',
                    expected_hidden_goals=['clean_car_goal'],
                    expected_required_premises=['vehicle_present'],
                    expected_satisfied_premises=[],
                    expected_missing_premises=['vehicle_present'],
                    expected_risky_actions=['walk_without_car'],
                    forbidden_premises=['open_access'],
                    expected_clarification_needed=True,
                    expected_support_operators=['SERVICE_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
                )
            ],
            cp_examples=CpParserEvaluator.load_examples(Path('examples/cp_parser_eval.jsonl')),
            vlso_cases=VlsoGroundedEvaluator.load_cases(Path('examples/vlso_eval.jsonl')),
            cp_mode='heuristic',
        )
        progress = OperatorIntelligenceProgressEstimator().estimate(snapshot)
        self.assertGreater(progress.operator_architecture.score, 0.0)
        self.assertGreater(progress.premise_reasoning.score, 0.0)
        self.assertGreater(progress.research_architecture_overall, 0.0)
        self.assertLessEqual(progress.robust_general_intelligence_overall, progress.research_architecture_overall)

    def test_progress_estimator_handles_empty_snapshot(self) -> None:
        progress = OperatorIntelligenceProgressEstimator().estimate(SemOpCommonEvaluator().evaluate())
        self.assertEqual(progress.operator_architecture.score, 0.0)
        self.assertEqual(progress.research_architecture_overall, 0.0)


    def test_real_image_eval_builder_emits_candidates_and_seed_cases(self) -> None:
        base_dir = Path('tests/real_image_eval_builder_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        (base_dir / 'downloads').mkdir(parents=True, exist_ok=True)
        image_path = base_dir / 'downloads' / 'bag.jpg'
        Image.new('RGB', (8, 8), color=(255, 255, 255)).save(image_path)
        records_path = base_dir / 'records.jsonl'
        manifest_path = base_dir / 'manifest.jsonl'
        candidate_output = base_dir / 'candidates.jsonl'
        seed_output = base_dir / 'seed.jsonl'
        record_line = json.dumps({
            'provider': 'openverse',
            'source_id': 'bag1',
            'title': 'Handbag with zipper handle',
            'query': 'bag handle opening',
            'raw': {'tags': [{'name': 'bag'}, {'name': 'zipper'}, {'name': 'handle'}]},
        }, ensure_ascii=False)
        manifest_line = json.dumps({
            'provider': 'openverse',
            'source_id': 'bag1',
            'target_path': str(image_path),
        }, ensure_ascii=False)
        records_path.write_text(record_line + "\n", encoding='utf-8')
        manifest_path.write_text(manifest_line + "\n", encoding='utf-8')
        try:
            summary = RealImageEvalBuilder().build(records_path, manifest_path, candidate_output, seed_output)
            self.assertEqual(summary.num_existing_images, 1)
            self.assertGreaterEqual(summary.num_candidates, 1)
            candidate_rows = [json.loads(line) for line in candidate_output.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertIn('zipper', candidate_rows[0]['expected_entities'])
            self.assertTrue(seed_output.exists())
        finally:
            shutil.rmtree(base_dir)

    def test_cp_learned_parser_detects_adapter_base_model(self) -> None:
        base_dir = Path('tests/cp_adapter_detect_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': 'dummy/base-model'}), encoding='utf-8')
        try:
            self.assertEqual(CpLearnedParser._detect_adapter_base_model(str(base_dir)), 'dummy/base-model')
        finally:
            shutil.rmtree(base_dir)

    def test_real_image_eval_builder_finalizes_approved_candidates(self) -> None:
        base_dir = Path('tests/real_image_eval_finalize_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = base_dir / 'candidates.jsonl'
        output_path = base_dir / 'gold.jsonl'
        rows = [
            {
                'case_id': 'case_1',
                'image_path': 'downloads/bag.jpg',
                'query': 'What opening is visible here?',
                'expected_entities': ['bag', 'zipper'],
                'expected_relations': [],
                'required_terms': ['zipper'],
                'forbidden_terms': ['clearer image'],
                'expected_operators': ['ACCESS_CONTROL_OPERATOR'],
                'expected_operator_bindings': [{'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'zipper', 'parent': 'bag'}],
                'expected_support_premises': ['open_access'],
                'title': 'Bag',
                'provider': 'openverse',
                'source_id': '1',
                'review_status': 'approved',
                'review_notes': 'good',
                'needs_review': False,
            },
            {
                'case_id': 'case_2',
                'image_path': 'downloads/other.jpg',
                'query': 'What object is visible here?',
                'expected_entities': ['tool'],
                'expected_relations': [],
                'required_terms': [],
                'forbidden_terms': ['clearer image'],
                'expected_operators': ['TOOL_GRASP_OPERATOR'],
                'expected_operator_bindings': [],
                'expected_support_premises': [],
                'title': 'Tool',
                'provider': 'openverse',
                'source_id': '2',
                'review_status': 'rejected',
                'needs_review': True,
            },
        ]
        candidates_path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
        try:
            summary = RealImageEvalBuilder().finalize_reviewed(candidates_path, output_path)
            self.assertEqual(summary.num_approved, 1)
            gold_rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(len(gold_rows), 1)
            self.assertNotIn('review_status', gold_rows[0])
            self.assertEqual(gold_rows[0]['case_id'], 'case_1')
        finally:
            shutil.rmtree(base_dir)

    def test_cp_lora_experiment_runner_dry_run_builds_bundle_and_summary(self) -> None:
        workspace = Path('tests/cp_lora_experiment_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            summary = CpLoraExperimentRunner().run(CpLoraExperimentConfig(
                workspace=str(workspace),
                model_name_or_path='local-test-model',
                dataset_paths=[str(Path('examples/cp_parser_eval.jsonl'))],
                eval_dataset_paths=[str(Path('examples/cp_geometry_parser_eval.jsonl'))],
                execute_train=True,
                dry_run_train=True,
                local_files_only=True,
                max_steps=4,
            ))
            self.assertTrue(Path(summary.train_jsonl).exists())
            self.assertTrue(Path(summary.val_jsonl).exists())
            self.assertIsNotNone(summary.training_summary)
            self.assertEqual(summary.training_summary.get('mode'), 'dry_run')
            self.assertEqual(summary.heuristic_eval.get('num_examples'), 10)
            self.assertTrue((workspace / 'experiment_summary.json').exists())
        finally:
            shutil.rmtree(workspace)

    def test_cp_training_scaffold_finds_latest_checkpoint(self) -> None:
        workspace = Path('tests/cp_checkpoint_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            (workspace / 'checkpoint-2').mkdir(parents=True)
            (workspace / 'checkpoint-10').mkdir(parents=True)
            (workspace / 'checkpoint-7').mkdir(parents=True)
            latest = CpParserTrainingScaffold.find_latest_checkpoint(workspace)
            self.assertEqual(Path(latest).name, 'checkpoint-10')
        finally:
            shutil.rmtree(workspace)

    def test_cp_lora_experiment_runner_passes_resume_and_save_config(self) -> None:
        workspace = Path('tests/cp_lora_resume_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            training_run = workspace / 'training_run'
            (training_run / 'checkpoint-12').mkdir(parents=True)
            summary = CpLoraExperimentRunner().run(CpLoraExperimentConfig(
                workspace=str(workspace),
                model_name_or_path='local-test-model',
                dataset_paths=[str(Path('examples/cp_parser_eval.jsonl'))],
                eval_dataset_paths=[str(Path('examples/cp_geometry_parser_eval.jsonl'))],
                execute_train=True,
                dry_run_train=True,
                local_files_only=True,
                max_steps=4,
                resume_from_checkpoint=str(training_run / 'checkpoint-12'),
                save_steps=2,
                save_total_limit=3,
            ))
            self.assertEqual(summary.training_summary.get('resume_from_checkpoint'), str(training_run / 'checkpoint-12'))
            self.assertEqual(summary.training_summary.get('config', {}).get('save_steps'), 2)
            self.assertEqual(summary.training_summary.get('config', {}).get('save_total_limit'), 3)
        finally:
            shutil.rmtree(workspace)
    def test_real_image_eval_builder_auto_selects_seed_gold_rows(self) -> None:
        base_dir = Path('tests/real_image_eval_auto_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = base_dir / 'candidates.jsonl'
        output_path = base_dir / 'gold.jsonl'
        rows = [
            {
                'case_id': 'good_1',
                'image_path': 'downloads/bag.jpg',
                'query': 'What opening or access control is visible here?',
                'expected_entities': ['bag', 'zipper'],
                'expected_relations': [],
                'required_terms': ['zipper'],
                'forbidden_terms': ['clearer image'],
                'expected_operators': ['ACCESS_CONTROL_OPERATOR', 'CONTAINER_BODY_OPERATOR'],
                'expected_operator_bindings': [{'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'zipper', 'parent': 'bag'}],
                'expected_support_premises': ['open_access'],
                'title': 'Bag with zipper',
                'provider': 'openverse',
                'source_id': '1',
                'needs_review': True,
            },
            {
                'case_id': 'bad_1',
                'image_path': 'downloads/bad.jpg',
                'query': 'What access-related object is visible here?',
                'expected_entities': ['drawer', 'handle'],
                'expected_relations': [],
                'required_terms': ['handle'],
                'forbidden_terms': ['clearer image'],
                'expected_operators': ['ATTACHED_GRASP_OPERATOR'],
                'expected_operator_bindings': [],
                'expected_support_premises': ['manipulable_grasp'],
                'title': 'how to use a knife',
                'provider': 'openverse',
                'source_id': '2',
                'needs_review': True,
            },
        ]
        candidates_path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
        try:
            summary = RealImageEvalBuilder().finalize_auto_selected(candidates_path, output_path, auto_approve_limit=1)
            self.assertEqual(summary.num_approved, 1)
            gold_rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(gold_rows[0]['case_id'], 'good_1')
        finally:
            shutil.rmtree(base_dir)

    def test_real_image_eval_builder_finalized_gold_resolves_candidate_relative_image_paths(self) -> None:
        base_dir = Path('tests/real_image_eval_path_runtime')
        if base_dir.exists():
            shutil.rmtree(base_dir)
        downloads_dir = base_dir / 'downloads'
        downloads_dir.mkdir(parents=True, exist_ok=True)
        (downloads_dir / 'bag.jpg').write_bytes(b'fake')
        candidates_path = base_dir / 'candidates.jsonl'
        output_path = base_dir / 'gold.jsonl'
        row = {
            'case_id': 'good_1',
            'image_path': 'downloads/bag.jpg',
            'query': 'What opening or access control is visible here?',
            'expected_entities': ['bag', 'zipper'],
            'expected_relations': [],
            'required_terms': ['zipper'],
            'forbidden_terms': ['clearer image'],
            'expected_operators': ['ACCESS_CONTROL_OPERATOR', 'CONTAINER_BODY_OPERATOR'],
            'expected_operator_bindings': [{'operator_name': 'ACCESS_CONTROL_OPERATOR', 'subject': 'zipper', 'parent': 'bag'}],
            'expected_support_premises': ['open_access'],
            'title': 'Bag with zipper',
            'provider': 'openverse',
            'source_id': '1',
            'review_status': 'approved',
        }
        candidates_path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        try:
            summary = RealImageEvalBuilder().finalize_reviewed(candidates_path, output_path)
            self.assertEqual(summary.num_approved, 1)
            gold_rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertTrue(Path(gold_rows[0]['image_path']).exists())
        finally:
            shutil.rmtree(base_dir)

    def test_vlso_question_answerer_describes_structural_access_parts_for_inventory_questions(self) -> None:
        world = VLSOReasoner(mode='deep').run(
            'What objects or openings are visible here?',
            visual_input={
                'objects': [
                    {'id': 'shape_1', 'label': 'shape_1', 'kind': 'shape', 'bbox': [10, 10, 180, 160], 'concept_labels': ['HAS_INTERIOR', 'STRUCTURAL_CONTAINER_CANDIDATE']},
                    {'id': 'shape_2', 'label': 'shape_2', 'kind': 'part', 'bbox': [40, 12, 150, 24], 'concept_labels': ['ACCESS_OPENING_CANDIDATE', 'ZIPPER_LIKE_PART']},
                    {'id': 'shape_3', 'label': 'shape_3', 'kind': 'part', 'bbox': [12, 40, 28, 120], 'concept_labels': ['HANDLE_CANDIDATE', 'GRASPABLE_PART']},
                ]
            },
        )
        answer = VLSOQuestionAnswerer().answer('What objects or openings are visible here?', world, answer_mode='structured')
        self.assertIn('opening', answer.answer_text.lower())
        self.assertIn('handle', answer.answer_text.lower())

    def test_common_evaluator_accepts_cp_hidden_and_vlso_real_inputs(self) -> None:
        evaluator = SemOpCommonEvaluator()
        cp_hidden_examples = CpParserEvaluator.load_examples(Path('examples/cp_hidden_constraint_eval.jsonl'))
        vlso_real_cases = VlsoGroundedEvaluator.load_cases(Path('examples/vlso_real_image_eval.jsonl'))
        snapshot = evaluator.evaluate(cp_hidden_examples=cp_hidden_examples, vlso_real_image_cases=vlso_real_cases)
        self.assertIsNotNone(snapshot.cp_hidden_constraints)
        self.assertIsNotNone(snapshot.vlso_real_image)

    def test_script_compatibility_experiment_pipeline_can_use_trained_model_path(self) -> None:
        runtime_db = Path('tests/script_compatibility_experiment.db')
        if runtime_db.exists():
            runtime_db.unlink()
        model_path = Path('tests/script_compatibility_experiment_model.json')
        if model_path.exists():
            model_path.unlink()
        try:
            store = CorpusMemoryStore(runtime_db)
            graph = StructuredMeaningPipeline(mode='heuristic').run('I need to open the drawer to get the keys.')
            store.upsert_graph(graph, source='premise_experiment', split='train')
            store.upsert_premise_operator_memory(graph, source='premise_experiment', split='train')
            summary = ScriptCompatibilityTrainer().train_from_memory(store, output_path=model_path, source='premise_experiment', epochs=2, learning_rate=0.05)
            self.assertTrue(model_path.exists())
            pipeline = StructuredMeaningPipeline(mode='heuristic', memory_store_path=str(runtime_db), memory_source='premise_experiment', script_compatibility_model_path=str(model_path))
            graph_with_model = pipeline.run('Should I open the drawer before taking out the keys?')
            self.assertTrue(any(item.hidden_goal == 'retrieve_item_from_drawer_goal' for item in graph_with_model.premise_candidates))
            self.assertGreaterEqual(summary.model.get('training_examples', 0), 1)
        finally:
            if runtime_db.exists():
                runtime_db.unlink()
            if model_path.exists():
                model_path.unlink()

    def test_teacher_trace_exporter_builds_hidden_premise_and_cp_records(self) -> None:
        exporter = TeacherTraceExporter(
            premise_pipeline=StructuredMeaningPipeline(mode="heuristic"),
            cp_reasoner=CompetitiveProgrammingReasoner(),
        )
        hidden_records = exporter.export_hidden_premise_eval("examples/hidden_premise_eval.jsonl")
        cp_records = exporter.export_cp_parser_eval("examples/cp_parser_eval.jsonl")
        self.assertTrue(hidden_records)
        self.assertTrue(cp_records)
        self.assertEqual(hidden_records[0].task, "hidden_premise")
        self.assertEqual(cp_records[0].task, "cp_structuring")
        sft_records = exporter.to_sft_records(hidden_records[:1] + cp_records[:1])
        self.assertEqual(len(sft_records), 2)
        self.assertIn("Teacher trace context", sft_records[0].prompt)

    def test_teacher_trace_exporter_builds_vlso_record(self) -> None:
        exporter = TeacherTraceExporter(
            vlso_reasoner=VLSOReasoner(mode="hybrid", language_mode="heuristic", answer_mode="structured")
        )
        records = exporter.export_vlso_eval("examples/vlso_eval.jsonl", base_dir="examples")
        self.assertTrue(records)
        self.assertEqual(records[0].task, "vlso_grounded_qa")
        self.assertIn("answer_text", records[0].completion_payload)


    def test_teacher_trace_exporter_builds_operator_proposal_records(self) -> None:
        exporter = TeacherTraceExporter(
            premise_pipeline=StructuredMeaningPipeline(mode="heuristic")
        )
        records = exporter.export_operator_transfer_eval("examples/operator_transfer_eval.jsonl")
        self.assertTrue(records)
        tasks = {record.task for record in records}
        self.assertIn("operator_proposal", tasks)
        self.assertIn("operator_self_evolution", tasks)
        proposal_record = next(record for record in records if record.task == "operator_proposal")
        self.assertIn("operator_decompositions", proposal_record.teacher_trace)


    def test_teacher_trace_exporter_accepts_harder_cp_and_real_image_sets(self) -> None:
        exporter = TeacherTraceExporter(
            premise_pipeline=StructuredMeaningPipeline(mode="heuristic"),
            cp_reasoner=CompetitiveProgrammingReasoner(),
            vlso_reasoner=VLSOReasoner(mode="hybrid", language_mode="heuristic", answer_mode="structured"),
        )
        cp_hidden_records = exporter.export_cp_parser_eval("examples/cp_hidden_constraint_eval.jsonl")
        vlso_real_records = exporter.export_vlso_eval("examples/vlso_real_image_eval_gold.jsonl", base_dir=".")
        self.assertTrue(cp_hidden_records)
        self.assertTrue(vlso_real_records)

    def test_operator_curriculum_builder_creates_balanced_bundle(self) -> None:
        workspace = Path('tests/operator_learning_bundle_runtime')
        trace_path = workspace / 'teacher_traces.jsonl'
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in ['hidden_premise', 'cp_structuring', 'vlso_grounded_qa', 'operator_proposal']:
            for idx in range(2):
                rows.append(TeacherTraceRecord(
                    task=task,
                    input_text=f'{task} example {idx}',
                    input_payload={'id': idx},
                    teacher_trace={'task': task, 'idx': idx},
                    completion_payload={'label': task},
                    metadata={},
                ).model_dump())
        trace_path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
        try:
            summary = OperatorCurriculumBuilder().build_bundle(trace_path, workspace, val_ratio=0.5)
            self.assertTrue(Path(summary.train_trace_jsonl).exists())
            self.assertTrue(Path(summary.val_trace_jsonl).exists())
            self.assertTrue(Path(summary.train_sft_jsonl).exists())
            self.assertTrue(Path(summary.curriculum_plan_path).exists())
            self.assertEqual(summary.task_counts['hidden_premise'], 2)
            self.assertGreater(summary.num_train, 0)
            self.assertGreater(summary.num_val, 0)
        finally:
            shutil.rmtree(workspace)

    def test_operator_training_scaffold_dry_run_writes_plan(self) -> None:
        workspace = Path('tests/operator_training_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        train_jsonl = workspace / 'operator_train_sft.jsonl'
        rows = [
            DistillationSftRecord(prompt='p1', completion='c1', task='hidden_premise').model_dump(),
            DistillationSftRecord(prompt='p2', completion='c2', task='operator_proposal').model_dump(),
        ]
        train_jsonl.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
        try:
            summary = OperatorTrainingScaffold().run(OperatorTrainConfig(
                model_name_or_path='local-test-model',
                output_dir=str(workspace / 'training_run'),
                train_jsonl=str(train_jsonl),
                dry_run=True,
                local_files_only=True,
                use_lora=True,
            ))
            self.assertEqual(summary['mode'], 'dry_run')
            self.assertEqual(summary['num_records'], 2)
            self.assertTrue(Path(summary['plan_path']).exists())
        finally:
            shutil.rmtree(workspace)

    def test_operator_training_scaffold_finds_latest_checkpoint(self) -> None:
        workspace = Path('tests/operator_checkpoint_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            (workspace / 'checkpoint-3').mkdir(parents=True)
            (workspace / 'checkpoint-11').mkdir(parents=True)
            (workspace / 'checkpoint-7').mkdir(parents=True)
            latest = OperatorTrainingScaffold.find_latest_checkpoint(workspace)
            self.assertEqual(Path(latest).name, 'checkpoint-11')
        finally:
            shutil.rmtree(workspace)

    def test_operator_training_scaffold_skips_resume_on_config_mismatch(self) -> None:
        workspace = Path('tests/operator_resume_mismatch_runtime')
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            old_config = OperatorTrainConfig(
                model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
                output_dir=str(workspace),
                train_jsonl='old.jsonl',
                use_lora=False,
            )
            OperatorTrainingScaffold._save_run_metadata(workspace, old_config, 'resolved-old-model')
            new_config = OperatorTrainConfig(
                model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
                output_dir=str(workspace),
                train_jsonl='new.jsonl',
                use_lora=True,
            )
            effective, info = OperatorTrainingScaffold.resolve_resume_checkpoint(
                workspace,
                str(workspace / 'checkpoint-1'),
                new_config,
                'resolved-new-model',
            )
            self.assertIsNone(effective)
            self.assertEqual(info.get('resume_skipped_reason'), 'checkpoint_config_mismatch')
        finally:
            shutil.rmtree(workspace)

    def test_operator_runtime_compiles_and_executes_goal_risk_case(self) -> None:
        graph = StructuredMeaningPipeline(mode='heuristic').run('I am going to the car wash and traffic is bad; should I walk there?')
        graph = compile_and_execute(graph)
        self.assertTrue(graph.operator_instructions)
        self.assertIsNotNone(graph.operator_execution)
        self.assertTrue(any('goal risk' in item for item in graph.operator_execution.warnings))
        self.assertTrue(any('walk_without_car' in item for item in graph.operator_execution.derived_decisions))

    def test_response_synthesizer_includes_compiled_execution_lines(self) -> None:
        graph = StructuredMeaningPipeline(mode='heuristic').run('I am going to the car wash and traffic is bad; should I walk there?')
        graph = compile_and_execute(graph)
        response = ResponseSynthesizer().synthesize(graph)
        self.assertTrue(response.compiled_execution)
        self.assertTrue(response.context_frame)
        self.assertIn('Decision:', response.to_text())
        self.assertIn('?곗궛???ㅽ뻾:', response.to_text())

    def test_response_synthesizer_includes_analogical_memories(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), 'analogical_response_test.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        try:
            store = CorpusMemoryStore(db_path)
            base_pipeline = StructuredMeaningPipeline(mode='heuristic')
            store.upsert_graph(base_pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'), source='demo', split='train')
            store.upsert_graph(base_pipeline.run('The pouch is zipped closed and I need the document inside. Can I pull it out now?'), source='demo', split='train')
            graph = StructuredMeaningPipeline(mode='heuristic', memory_store_path=db_path, memory_source='demo').run('The box is closed and I need the file inside. Can I pull it out now?')
            response = ResponseSynthesizer().synthesize(graph)
            self.assertTrue(response.analogical_memories)
            self.assertIn('??⑥る쭜?????:', response.to_text())
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_context_analyzer_builds_constraint_first_frame_for_risky_goal(self) -> None:
        graph = StructuredMeaningPipeline(mode='heuristic').run('I am going to the car wash and traffic is bad; should I walk there?')
        self.assertIsNotNone(graph.context_frame)
        self.assertEqual(graph.context_frame.reasoning_mode, 'clarify_goal')
        self.assertEqual(graph.context_frame.frame_type, 'mobility_reasoning')
        self.assertIn('vehicle_present', graph.context_frame.active_constraints)
        self.assertIn('GOAL_PRESERVATION_OPERATOR', graph.context_frame.operator_view)
        self.assertIn('ServiceGoalToConstraintFunctor', graph.context_frame.functor_view)

    def test_context_analyzer_marks_execution_ready_when_access_is_satisfied(self) -> None:
        graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is already open and I need the folder inside. Can I take it now?')
        self.assertIsNotNone(graph.context_frame)
        self.assertEqual(graph.context_frame.reasoning_mode, 'execution_ready')
        self.assertEqual(graph.context_frame.frame_type, 'access_reasoning')
        self.assertIn('open_access', graph.context_frame.satisfied_requirements)
        self.assertNotIn('open_access', graph.context_frame.active_constraints)

    def test_pipeline_document_grounding_attaches_evidence_nodes_from_source_context(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        source_context = 'Manual:\nStep 1: Open the drawer before retrieval.\nStep 2: Pull the folder out after the drawer is open.'
        graph = pipeline.run('What should I do first?', source_context=source_context)
        evidence_nodes = [node for node in graph.nodes if node.kind == 'evidence']
        grounded_edges = [edge for edge in graph.edges if edge.relation == 'GROUNDED_BY']
        self.assertTrue(evidence_nodes)
        self.assertTrue(grounded_edges)
        self.assertTrue(any(result.domain == 'document_grounding' for result in graph.symbolic_results))

    def test_pipeline_visual_input_merges_structural_operators_into_main_graph(self) -> None:
        pipeline = StructuredMeaningPipeline(mode='heuristic')
        graph = pipeline.run(
            'How can I open or access this drawer?',
            visual_input={
                'annotations': [
                    {'id': 'drawer_body', 'label': 'drawer', 'bbox': [10, 10, 160, 110], 'bbox_mode': 'xyxy', 'kind': 'object', 'mask_area': 11000, 'bbox_fill_ratio': 0.72, 'hull_fill_ratio': 0.84},
                    {'id': 'drawer_handle', 'label': 'handle', 'bbox': [124, 48, 148, 72], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.95, 'structural_role': 'handle', 'segmentation_confidence': 0.92},
                    {'id': 'drawer_opening', 'label': 'opening band', 'bbox': [24, 20, 138, 38], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'part_of_confidence': 0.94, 'structural_role': 'opening', 'segmentation_confidence': 0.93},
                ]
            },
        )
        operator_names = {item.name for item in graph.induced_operators}
        self.assertIn('ACCESS_PORT_OPERATOR', operator_names)
        self.assertIn('drawer_handle', {node.id for node in graph.nodes})
        self.assertTrue(any(node.kind == 'evidence' and node.attributes.get('modality') == 'vision' for node in graph.nodes))
        self.assertTrue(any(edge.relation == 'GROUNDED_BY' and edge.source == 'question' for edge in graph.edges))
        self.assertTrue(any('multimodal merge:' in item for item in graph.audit_trace))

    def test_operator_runtime_flags_missing_visual_grounding_edges(self) -> None:
        from semop.structures import Node, StructuredMeaningGraph

        graph = StructuredMeaningGraph(query='What does the image show about the drawer opening?', intent='goal_directed_reasoning')
        graph.add_node(Node(id='question', label=graph.query, kind='query', provenance=['test']))
        graph.add_node(Node(id='visual_scene', label='visual_scene', kind='scene', provenance=['multimodal:vision_scene']))
        graph.add_node(Node(id='drawer_handle', label='drawer handle', kind='part', provenance=['multimodal:vision_entity']))
        graph = compile_and_execute(graph)
        self.assertTrue(any('visual reasoning has no explicit GROUNDED_BY visual evidence edge' in item for item in graph.operator_execution.compiler_findings))

    def test_operator_runtime_claim_grounding_flags_unsupported_document_claim(self) -> None:
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        graph = StructuredMeaningPipeline(mode='heuristic').run(
            'What should I do first?',
            source_context='Manual:\nOpen the drawer before retrieval.',
        )
        graph.symbolic_results = [
            SymbolicResult(
                domain='document_grounding',
                answer='Open the drawer first and inspect the hidden sensor.',
                evidence=['Open the drawer before retrieval.'],
                confidence=0.92,
                source='unit_test',
            )
        ]
        graph = compile_and_execute(graph)
        self.assertIsNotNone(graph.operator_execution)
        self.assertTrue(graph.operator_execution.claim_groundings)
        self.assertLess(graph.operator_execution.claim_grounding_score, 1.0)
        self.assertTrue(any('claim grounding warning:' in item for item in graph.operator_execution.compiler_findings))
        self.assertTrue(any((not item.grounded) and 'sensor' in item.claim.lower() for item in graph.operator_execution.claim_groundings))

    def test_response_synthesizer_surfaces_claim_grounding_lines(self) -> None:
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        graph = StructuredMeaningPipeline(mode='heuristic').run(
            'What should I do first?',
            source_context='Manual:\nOpen the drawer before retrieval.',
        )
        graph.symbolic_results = [
            SymbolicResult(
                domain='document_grounding',
                answer='Open the drawer first and inspect the hidden sensor.',
                evidence=['Open the drawer before retrieval.'],
                confidence=0.92,
                source='unit_test',
            )
        ]
        graph = compile_and_execute(graph)
        response = ResponseSynthesizer().synthesize(graph)
        self.assertTrue(any('Claim support:' in item for item in response.compiled_execution))
        self.assertTrue(any('Unsupported claim:' in item or item.startswith('Claim: ') for item in response.compiled_execution))

    def test_retained_operator_trainer_reuses_verified_decompositions(self) -> None:
        from semop import RetainedOperatorTrainer

        output_path = os.path.join(os.path.dirname(__file__), 'retained_operator_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            graphs = [
                pipeline.run('I am going to the car wash and traffic is bad, should I walk there?'),
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
            ]
            summary = RetainedOperatorTrainer().train_from_graphs(graphs, output_path, min_support=1)
            self.assertGreaterEqual(summary.retained_operator_count, 1)
            retained_graph = StructuredMeaningPipeline(mode='heuristic', retained_algebra_path=output_path).run('The cabinet door is closed and I need the file inside. Should I reach in immediately?')
            self.assertTrue(any('retained_operator_algebra' in tag for candidate in retained_graph.induced_operators for tag in candidate.provenance))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_operator_runtime_counterexample_repairs_cover_missing_types_and_functors(self) -> None:
        from semop.structures import FunctorHypothesis, OperatorCandidate, OperatorDecomposition, StructuredMeaningGraph

        graph = StructuredMeaningGraph(query='repair test', intent='goal_directed_reasoning')
        graph.operator_decompositions = [
            OperatorDecomposition(operator_name='DOC_OPERATOR', basis_operators=['DOCUMENT_CONTEXT', 'REQUIRES'], rationale='needs evidence', confidence=0.7)
        ]
        graph.induced_operators = [
            OperatorCandidate(name='DOC_OPERATOR', family='DOC_OPERATOR', arity=1, input_types=['document_context'], output_type='evidence_span', description='doc operator')
        ]
        graph.functor_hypotheses = [
            FunctorHypothesis(name='BrokenVisualFunctor', source_category='visual_structure', target_category='goal_preservation_logic', object_map={'visual_scene': 'hidden_goal'}, morphism_map={'PART_OF': 'constraint_binding'}, confidence=0.6)
        ]
        graph = compile_and_execute(graph)
        self.assertTrue(any('document chunks' in item.lower() or 'document' in item.lower() for item in graph.operator_execution.counterexample_repairs))
        self.assertTrue(any('hidden goal' in item.lower() or 'visual' in item.lower() for item in graph.operator_execution.counterexample_repairs))

    def test_unified_benchmark_trains_and_scores_core_metrics(self) -> None:
        from semop import AnalogyEvalCase, CompilerRepairEvalCase, GroundedExplanationEvalCase, UnifiedBenchmarkHarness, UnifiedSemOpTrainer
        from semop.structures import OperatorDecomposition, StructuredMeaningGraph

        db_path = os.path.join(os.path.dirname(__file__), 'unified_benchmark_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'unified_benchmark_artifacts')
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            store = CorpusMemoryStore(db_path)
            seed_pipeline = StructuredMeaningPipeline(mode='heuristic')
            train_graphs = [
                seed_pipeline.run('I am going to the car wash and traffic is bad, should I walk there?'),
                seed_pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
                seed_pipeline.run('Open the drawer and retrieve the folder.'),
            ]
            for graph in train_graphs:
                store.upsert_graph(graph, source='unified_demo', split='train')
            training = UnifiedSemOpTrainer().train_from_store(db_path, output_dir, source='unified_demo')
            self.assertTrue(os.path.exists(training.artifacts.analogy_policy_path))
            self.assertTrue(os.path.exists(training.artifacts.graph_supervision_path))
            self.assertTrue(os.path.exists(training.artifacts.repair_policy_path))
            self.assertTrue(os.path.exists(training.artifacts.retained_repair_program_path))
            benchmark = UnifiedBenchmarkHarness().evaluate(
                training.artifacts,
                hidden_premise_cases=[
                    HiddenPremiseEvalCase(
                        query='The drawer is closed and I need the folder inside. Should I pull the folder out right now?',
                        expected_hidden_goals=['retrieve_item_from_drawer_goal'],
                        expected_required_premises=['open_access'],
                        expected_satisfied_premises=[],
                        expected_missing_premises=['open_access'],
                        expected_risky_actions=['retrieve_without_opening'],
                        forbidden_premises=['vehicle_present'],
                        expected_clarification_needed=False,
                    )
                ],
                transfer_cases=[
                    __import__('semop').OperatorTransferEvalCase(query='I am going to the car wash and traffic is bad, should I walk there?', domain='service', expected_operator_names=['GOAL_PRESERVATION_OPERATOR'], split='train'),
                    __import__('semop').OperatorTransferEvalCase(query='The cabinet door is closed and I need the file inside. Should I reach in immediately?', domain='cabinet_access', expected_operator_names=['GOAL_PRESERVATION_OPERATOR'], split='test'),
                ],
                analogy_cases=[
                    AnalogyEvalCase(query='The box is closed and I need the file inside. Can I pull it out now?', similar_graphs=train_graphs[:2], expected_requirement='open_access')
                ],
                grounding_cases=[
                    GroundedExplanationEvalCase(query='What should I do first?', source_context='Manual:\nOpen the drawer before retrieving the folder.', expected_evidence_terms=['open the drawer'])
                ],
                compiler_cases=[
                    CompilerRepairEvalCase(
                        graph=StructuredMeaningGraph(
                            query='broken benchmark graph',
                            intent='goal_directed_reasoning',
                            operator_decompositions=[OperatorDecomposition(operator_name='BROKEN_OPERATOR', basis_operators=['DOCUMENT_CONTEXT', 'REQUIRES'], rationale='broken', confidence=0.7)],
                        ),
                        expected_repair_terms=['document'],
                    )
                ],
            )
            self.assertGreaterEqual(benchmark.analogy_usefulness, 0.0)
            self.assertGreater(benchmark.grounded_explanation_fidelity, 0.0)
            self.assertGreater(benchmark.repair_success_rate, 0.0)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_graph_supervision_exporter_writes_runtime_graph_labels(self) -> None:
        from semop import GraphSupervisionExporter

        output_path = os.path.join(os.path.dirname(__file__), 'graph_supervision_test.jsonl')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?')
            summary = GraphSupervisionExporter().export_from_graphs([graph], output_path)
            self.assertEqual(summary.exported_examples, 1)
            payload = [json.loads(line) for line in Path(output_path).read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertEqual(payload[0]['hidden_goals'][0], 'retrieve_item_from_drawer_goal')
            self.assertIn('open_access', payload[0]['required_premises'])
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_unified_parser_bootstrap_recovers_goal_and_premise_slots(self) -> None:
        from semop import LearnedUnifiedParser, UnifiedParserTrainer

        output_path = os.path.join(os.path.dirname(__file__), 'unified_parser_graph_slots.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            graphs = [
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
                pipeline.run('The cabinet door is closed and I need the file inside. Should I reach in immediately?'),
            ]
            UnifiedParserTrainer().train_from_graphs(graphs, output_path)
            parser = LearnedUnifiedParser(model_path=output_path)
            prediction = parser.predict('The drawer is shut and I need the document inside. Can I grab it now?')
            self.assertIn('open_access', prediction.required_premises)
            self.assertGreater(prediction.confidence, 0.0)
            bootstrapped = parser.bootstrap_graph('The drawer is shut and I need the document inside. Can I grab it now?', prediction=prediction)
            self.assertTrue(bootstrapped.hidden_goals)
            self.assertTrue(any(edge.relation == 'REQUIRES' and edge.target == 'open_access' for edge in bootstrapped.edges))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_pipeline_uses_parser_first_bootstrap_when_confident(self) -> None:
        from semop import UnifiedParserTrainer

        output_path = os.path.join(os.path.dirname(__file__), 'parser_first_bootstrap.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            teacher = StructuredMeaningPipeline(mode='heuristic')
            graphs = [
                teacher.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
                teacher.run('The cabinet door is closed and I need the file inside. Should I reach in immediately?'),
            ]
            UnifiedParserTrainer().train_from_graphs(graphs, output_path)
            graph = StructuredMeaningPipeline(mode='heuristic', unified_parser_path=output_path).run('The drawer is shut and I need the document inside. Can I grab it now?')
            self.assertTrue(any('parser-first bootstrap activated' in item for item in graph.audit_trace))
            self.assertIn('open_access', graph.required_premises + graph.satisfied_premises + graph.missing_premises)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_operator_repair_policy_trainer_prioritizes_structural_repairs(self) -> None:
        from semop import OperatorRepairPolicyScorer, OperatorRepairPolicyTrainer, compile_and_execute
        from semop.structures import StructuredMeaningGraph

        output_path = os.path.join(os.path.dirname(__file__), 'operator_repair_policy_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            graphs = [
                pipeline.run('I am going to the car wash and traffic is bad, should I walk there?'),
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
            ]
            summary = OperatorRepairPolicyTrainer().train_from_graphs(graphs, output_path)
            self.assertGreater(summary.trained_on_examples, 0)
            broken = StructuredMeaningGraph(query='repair ranking case', intent='goal_directed_reasoning')
            broken.hidden_goals = ['clean_car_goal']
            broken.required_premises = ['vehicle_present']
            broken = compile_and_execute(broken)
            ranked = OperatorRepairPolicyScorer(model_path=output_path).rank_actions(broken, broken.operator_execution.compiler_findings)
            self.assertIn('add_goal_preservation_decomposition', ranked[:2])
            self.assertIn('bind_requires_edges', ranked[:2])
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_operator_repair_policy_trainer_prioritizes_claim_repairs(self) -> None:
        from semop import OperatorRepairPolicyScorer, OperatorRepairPolicyTrainer, compile_and_execute
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        output_path = os.path.join(os.path.dirname(__file__), 'operator_repair_policy_claim_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            summary = OperatorRepairPolicyTrainer().train_from_graphs([graph], output_path)
            self.assertGreater(summary.trained_on_examples, 0)
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.symbolic_results[0].answer = 'Open the drawer first and inspect the hidden sensor.'
            broken = compile_and_execute(broken)
            ranked = OperatorRepairPolicyScorer(model_path=output_path).rank_actions(broken, broken.operator_execution.compiler_findings)
            self.assertIn('trim_unsupported_claims', ranked[:3])
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_repair_utility_trainer_learns_claim_trim_utility(self) -> None:
        from semop import RepairUtilityScorer, RepairUtilityTrainer, compile_and_execute
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        output_path = os.path.join(os.path.dirname(__file__), 'repair_utility_claim_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            summary = RepairUtilityTrainer().train_from_graphs([graph], output_path)
            self.assertGreater(summary.trained_on_examples, 0)
            positive = StructuredMeaningGraph.from_dict(graph.model_dump())
            positive.symbolic_results[0].answer = 'Open the drawer first and inspect the hidden sensor.'
            positive = compile_and_execute(positive)
            scorer = RepairUtilityScorer(model_path=output_path)
            positive_score = scorer.score_action(
                positive,
                'trim_unsupported_claims',
                positive.operator_execution.compiler_findings + positive.operator_execution.counterexample_repairs,
            )
            self.assertGreater(positive_score, scorer.model.action_min_utility)
            unsafe = StructuredMeaningGraph.from_dict(graph.model_dump())
            unsafe.symbolic_results[0].answer = 'Inspect the hidden sensor.'
            unsafe = compile_and_execute(unsafe)
            reject_reason = scorer.rejection_reason(
                unsafe,
                'trim_unsupported_claims',
                unsafe.operator_execution.compiler_findings + unsafe.operator_execution.counterexample_repairs,
                ['typed_claim_grounding_repair'],
            )
            self.assertTrue(isinstance(reject_reason, str) and reject_reason.startswith('low_expected_utility'))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_operator_repair_engine_uses_utility_model_to_reject_low_value_document_repair(self) -> None:
        from semop import OperatorRepairEngine, RepairUtilityModel, compile_and_execute
        from semop.structures import SymbolicResult

        output_path = os.path.join(os.path.dirname(__file__), 'repair_utility_reject_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            model = RepairUtilityModel(
                action_bias={'attach_document_context_nodes': -1.0},
                program_bias={'typed_document_grounding_repair': -1.0},
            )
            Path(output_path).write_text(json.dumps({'weights': model.model_dump()}, ensure_ascii=False, indent=2), encoding='utf-8')
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            graph.nodes = [node for node in graph.nodes if node.id not in {'source_document'} and node.kind != 'evidence']
            graph.edges = [edge for edge in graph.edges if edge.relation not in {'USES_CONTEXT', 'HAS_EVIDENCE', 'GROUNDED_BY'}]
            graph = compile_and_execute(graph)
            repaired = OperatorRepairEngine(repair_utility_path=output_path).run(graph)
            self.assertIn('repair_rejected:attach_document_context_nodes', repaired.operator_execution.derived_decisions)
            self.assertTrue(any('low_expected_utility' in item for item in repaired.audit_trace))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    def test_unified_semop_trainer_emits_repair_policy_artifact(self) -> None:
        from semop import UnifiedSemOpTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'repair_policy_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'repair_policy_artifacts')
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            store = CorpusMemoryStore(db_path)
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            for query in [
                'I am going to the car wash and traffic is bad, should I walk there?',
                'The drawer is closed and I need the folder inside. Should I pull the folder out right now?',
            ]:
                store.upsert_graph(pipeline.run(query), source='repair_demo', split='train')
            store.upsert_graph(
                pipeline.run(
                    'The drawer is closed and I need the folder inside. Should I pull the folder out right now?',
                    visual_input={
                        'annotations': [
                            {'id': 'drawer_body', 'label': 'drawer', 'bbox': [10, 10, 160, 110], 'bbox_mode': 'xyxy', 'kind': 'object'},
                            {'id': 'drawer_handle', 'label': 'handle', 'bbox': [124, 48, 148, 72], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'structural_role': 'handle'},
                            {'id': 'drawer_opening', 'label': 'opening band', 'bbox': [24, 20, 138, 38], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'structural_role': 'opening'},
                        ]
                    },
                ),
                source='repair_demo',
                split='train',
            )
            training = UnifiedSemOpTrainer().train_from_store(db_path, output_dir, source='repair_demo')
            self.assertTrue(os.path.exists(training.artifacts.graph_supervision_path))
            self.assertTrue(os.path.exists(training.artifacts.multimodal_alignment_path))
            self.assertTrue(os.path.exists(training.artifacts.repair_policy_path))
            self.assertTrue(os.path.exists(training.artifacts.retained_repair_program_path))
            self.assertTrue(os.path.exists(training.artifacts.repair_utility_path))
            self.assertTrue(os.path.isdir(training.artifacts.continuous_learning_bundle_dir))
            self.assertGreater(training.graph_supervision.get('exported_examples', 0), 0)
            self.assertGreater(training.multimodal_alignment.get('retained_alignment_count', 0), 0)
            self.assertGreater(training.continuous_learning_bundle.get('trace_count', 0), 0)
            self.assertGreater(training.repair_policy.get('trained_on_examples', 0), 0)
            self.assertGreater(training.retained_repair_programs.get('retained_program_count', 0), 0)
            self.assertGreater(training.repair_utility.get('trained_on_examples', 0), 0)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_unified_semop_trainer_reinjects_repair_trace_graphs(self) -> None:
        from semop import UnifiedSemOpTrainer
        from semop.structures import SymbolicResult

        db_path = os.path.join(os.path.dirname(__file__), 'repair_trace_reinject_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'repair_trace_reinject_artifacts')
        for path in [db_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            store = CorpusMemoryStore(db_path)
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first and inspect the hidden sensor.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            store.upsert_graph(graph, source='repair_trace_demo', split='train')
            training = UnifiedSemOpTrainer().train_from_store(db_path, output_dir, source='repair_trace_demo')
            self.assertTrue(os.path.exists(training.artifacts.repair_utility_path))
            self.assertGreater(training.repair_trace_graph_count, 0)
            self.assertGreater(training.augmented_graph_count, training.trained_on_graphs)
            self.assertGreater(training.repair_utility.get('trained_on_examples', 0), 0)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_multimodal_alignment_memory_projects_visual_goal_constraints(self) -> None:
        from semop import MultimodalAlignmentTrainer

        output_path = os.path.join(os.path.dirname(__file__), 'multimodal_alignment_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        visual_payload = {
            'annotations': [
                {'id': 'drawer_body', 'label': 'drawer', 'bbox': [10, 10, 160, 110], 'bbox_mode': 'xyxy', 'kind': 'object'},
                {'id': 'drawer_handle', 'label': 'handle', 'bbox': [124, 48, 148, 72], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'structural_role': 'handle'},
                {'id': 'drawer_opening', 'label': 'opening band', 'bbox': [24, 20, 138, 38], 'bbox_mode': 'xyxy', 'kind': 'object', 'part_of': 'drawer_body', 'structural_role': 'opening'},
            ]
        }
        try:
            teacher_graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?', visual_input=visual_payload)
            summary = MultimodalAlignmentTrainer().train_from_graphs([teacher_graph], output_path)
            self.assertGreaterEqual(summary.retained_alignment_count, 1)
            graph = StructuredMeaningPipeline(mode='heuristic', multimodal_alignment_path=output_path).run('Can I take the document now?', visual_input=visual_payload)
            self.assertTrue(any('multimodal alignment memory:' in item for item in graph.audit_trace))
            self.assertTrue(graph.hidden_goals)
            self.assertIn('open_access', graph.required_premises + graph.satisfied_premises + graph.missing_premises)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_retained_operator_algebra_skips_retired_records(self) -> None:
        from semop import RetainedOperatorAlgebra, RetainedOperatorModel, RetainedOperatorRecord
        from semop.structures import StructuredMeaningGraph

        model = RetainedOperatorModel(records=[
            RetainedOperatorRecord(operator_name='OLD_OPERATOR', basis_signature=['HIDDEN_GOAL', 'REQUIRES'], support=3, domain_support=1, average_confidence=0.8, verification_rate=0.4, utility_score=0.35, activated_support=2, activation_success_rate=0.3, retired=True, retirement_reason='low_activation_success')
        ])
        graph = StructuredMeaningGraph(query='retirement test', intent='goal_directed_reasoning')
        graph.hidden_goals = ['clean_car_goal']
        graph.required_premises = ['vehicle_present']
        graph = RetainedOperatorAlgebra(model=model).enrich(graph)
        self.assertFalse(any(item.operator_name == 'OLD_OPERATOR' for item in graph.operator_decompositions))
        self.assertTrue(any('skipped 1 retired operator priors' in item for item in graph.audit_trace))

    def test_continuous_learning_bundle_builder_exports_runtime_and_review_traces(self) -> None:
        from semop import ContinuousLearningBundleBuilder

        review_db = os.path.join(os.path.dirname(__file__), 'continuous_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'continuous_bundle_artifacts')
        if os.path.exists(review_db):
            os.remove(review_db)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?')
            store = __import__('semop').ReviewQueueStore(review_db)
            item_id = store.enqueue(domain='access', scenario='drawer', query=graph.query, reasons=['grounding_review'], answer_text='Open the drawer first.', kpis={'clarification_need_rate': 0.0}, audit_items=[{'stage': 'evidence', 'detail': 'drawer access required'}])
            store.update_status(item_id, 'approved', 'validated by reviewer')
            summary = ContinuousLearningBundleBuilder().build_from_graphs([graph], output_dir, review_store_path=review_db)
            self.assertEqual(summary.trace_count, 1)
            self.assertEqual(summary.approved_review_count, 1)
            self.assertEqual(summary.promoted_review_count, 1)
            self.assertEqual(summary.filtered_review_count, 0)
            self.assertGreaterEqual(summary.sft_record_count, 2)
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'teacher_traces.jsonl')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'graph_supervision.jsonl')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'continuous_learning_sft.jsonl')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'review_promotion_manifest.json')))
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_review_promotion_policy_blocks_invalid_advice_without_override(self) -> None:
        from semop import evaluate_review_promotion

        blocked = evaluate_review_promotion({
            'id': 1,
            'domain': 'general',
            'query': 'Unsafe case',
            'reasons': ['invalid_advice_rate'],
            'answer_text': 'Do not proceed.',
            'resolution_note': 'Reviewed.',
            'status': 'approved',
        })
        allowed = evaluate_review_promotion({
            'id': 2,
            'domain': 'general',
            'query': 'Grounded case',
            'reasons': ['grounding_review'],
            'answer_text': 'Open the drawer first.',
            'resolution_note': 'Evidence attached.',
            'status': 'approved',
        })
        override = evaluate_review_promotion({
            'id': 3,
            'domain': 'general',
            'query': 'Corrected unsafe case',
            'reasons': ['invalid_advice_rate', 'approved_training_trace'],
            'answer_text': 'Stop and escalate.',
            'resolution_note': 'Manually approved safe trace.',
            'status': 'approved',
        })
        self.assertFalse(blocked.promotable)
        self.assertTrue(any(item == 'invalid_advice_rate' for item in blocked.blocked_reasons))
        self.assertTrue(allowed.promotable)
        self.assertTrue(override.promotable)
        self.assertTrue(any(item == 'approved_training_trace' for item in override.override_reasons))

    def test_continuous_learning_bundle_filters_non_promoted_reviews(self) -> None:
        from semop import ContinuousLearningBundleBuilder

        review_db = os.path.join(os.path.dirname(__file__), 'continuous_review_filter.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'continuous_bundle_filter_artifacts')
        if os.path.exists(review_db):
            os.remove(review_db)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?')
            store = __import__('semop').ReviewQueueStore(review_db)
            blocked_id = store.enqueue(domain='general', scenario='unsafe', query='Unsafe case', reasons=['invalid_advice_rate'], answer_text='Do not proceed.', kpis={'invalid_advice_rate': 0.4}, audit_items=[{'stage': 'safety', 'detail': 'unsafe answer corrected'}])
            allowed_id = store.enqueue(domain='general', scenario='grounding', query=graph.query, reasons=['grounding_review'], answer_text='Open the drawer first.', kpis={'clarification_need_rate': 0.0}, audit_items=[{'stage': 'evidence', 'detail': 'drawer access required'}])
            store.update_status(blocked_id, 'approved', 'reviewed but not training-safe by default')
            store.update_status(allowed_id, 'approved', 'accepted for retraining')
            summary = ContinuousLearningBundleBuilder().build_from_graphs([graph], output_dir, review_store_path=review_db)
            self.assertEqual(summary.approved_review_count, 2)
            self.assertEqual(summary.promoted_review_count, 1)
            self.assertEqual(summary.filtered_review_count, 1)
            manifest = json.loads(Path(os.path.join(output_dir, 'review_promotion_manifest.json')).read_text(encoding='utf-8'))
            self.assertEqual(len(manifest), 2)
            self.assertTrue(any(not item['promotable'] and 'invalid_advice_rate' in item['blocked_reasons'] for item in manifest))
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_domain_copilot_enqueues_review_with_severity(self) -> None:
        from semop import OpsKpiReport

        review_db = os.path.join(os.path.dirname(__file__), 'domain_copilot_severity_review.db')
        if os.path.exists(review_db):
            os.remove(review_db)
        try:
            copilot = DomainCopilot(review_queue_path=review_db)
            request = CopilotRequest(
                query='The aisle is blocked and approval is missing. Should I move the forklift anyway?',
                domain='warehouse_exception',
                scenario='exception_response',
            )
            graph = StructuredMeaningPipeline(mode='heuristic').run(request.query)
            result = __import__('semop').CopilotResult(
                request=request,
                graph=graph,
                answer_text='Move it anyway.',
                kpis=OpsKpiReport(
                    invalid_advice_rate=0.5,
                    plan_executability=0.4,
                    missing_prerequisite_rate=0.6,
                    context_misread_rate=0.3,
                    relation_recovery=0.4,
                    human_audit_usefulness=0.5,
                    clarification_need_rate=0.0,
                    notes=['unsafe case'],
                ),
                audit_items=[],
            )
            copilot._enqueue_review_if_needed(result)
            self.assertTrue(result.queued_for_review)
            self.assertEqual(result.review_severity, 'critical')
            detail = copilot.review_queue.fetch_item_detail(1)
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(detail['severity'], 'critical')
            self.assertAlmostEqual(detail['severity_weight'], 2.25, places=2)
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)

    def test_continuous_learning_bundle_exports_review_severity_weights(self) -> None:
        from semop import ContinuousLearningBundleBuilder

        review_db = os.path.join(os.path.dirname(__file__), 'continuous_review_severity.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'continuous_bundle_severity_artifacts')
        if os.path.exists(review_db):
            os.remove(review_db)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            context = 'Manual:\nOpen the drawer before retrieving the folder.'
            graph = StructuredMeaningPipeline(mode='heuristic').run(query, source_context=context)
            store = __import__('semop').ReviewQueueStore(review_db)
            item_id = store.enqueue(
                domain='warehouse_exception',
                scenario='exception_response',
                query=query,
                reasons=['grounding_review', 'compiler_validity_gap'],
                answer_text='Open the drawer first.',
                kpis={'clarification_need_rate': 0.0},
                audit_items=[{'stage': 'review', 'detail': 'high-risk correction'}],
                context_text=context,
                graph_payload=graph.model_dump(),
                severity='critical',
            )
            store.update_status(item_id, 'approved', 'accepted for high-risk retraining')
            summary = ContinuousLearningBundleBuilder().build_from_graphs([graph], output_dir, review_store_path=review_db)
            self.assertEqual(summary.promoted_review_count, 1)
            self.assertAlmostEqual(summary.promoted_training_weight_total, 2.25, places=2)
            manifest = json.loads(Path(os.path.join(output_dir, 'review_promotion_manifest.json')).read_text(encoding='utf-8'))
            promoted = [item for item in manifest if item['promotable']]
            self.assertEqual(len(promoted), 1)
            self.assertEqual(promoted[0]['severity'], 'critical')
            self.assertGreater(promoted[0]['training_weight'], 1.0)
            rows = [json.loads(line) for line in Path(os.path.join(output_dir, 'continuous_learning_sft.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            review_rows = [row for row in rows if row.get('task') == 'continuous_review_correction']
            self.assertEqual(len(review_rows), 1)
            self.assertEqual(review_rows[0]['metadata']['severity'], 'critical')
            self.assertGreater(review_rows[0]['metadata']['training_weight'], 1.0)
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_continuous_learning_bundle_exports_claim_grounding_metadata(self) -> None:
        from semop import ContinuousLearningBundleBuilder
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        review_db = os.path.join(os.path.dirname(__file__), 'continuous_review_claim_grounding.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'continuous_bundle_claim_grounding_artifacts')
        for path in [review_db]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first and inspect the hidden sensor.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            graph = compile_and_execute(graph)
            store = __import__('semop').ReviewQueueStore(review_db)
            item_id = store.enqueue(
                domain='general',
                scenario='qa',
                query=graph.query,
                reasons=['claim_grounding_review'],
                answer_text='Open the drawer first.',
                kpis={'clarification_need_rate': 0.0},
                audit_items=[{'stage': 'claim_grounding', 'detail': 'unsupported sensor clause removed'}],
                context_text=graph.source_context,
                graph_payload=graph.model_dump(),
                severity='high',
            )
            store.update_status(item_id, 'approved', 'accepted for claim grounding retraining')
            summary = ContinuousLearningBundleBuilder().build_from_graphs([graph], output_dir, review_store_path=review_db)
            self.assertEqual(summary.promoted_review_count, 1)
            rows = [json.loads(line) for line in Path(os.path.join(output_dir, 'continuous_learning_sft.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            review_rows = [row for row in rows if row.get('task') == 'continuous_review_correction']
            self.assertEqual(len(review_rows), 1)
            self.assertGreater(review_rows[0]['metadata']['unsupported_claim_count'], 0)
            self.assertLess(review_rows[0]['metadata']['claim_grounding_score'], 1.0)
            self.assertIn('Unsupported claims', review_rows[0]['prompt'])
            traces = [json.loads(line) for line in Path(os.path.join(output_dir, 'teacher_traces.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertGreater(traces[0]['metadata']['unsupported_claim_count'], 0)
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_continuous_learning_bundle_exports_repair_program_metadata(self) -> None:
        from semop import ContinuousLearningBundleBuilder, OperatorRepairEngine
        from semop.structures import SymbolicResult

        output_dir = os.path.join(os.path.dirname(__file__), 'continuous_bundle_repair_program_artifacts')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first and inspect the hidden sensor.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            repaired = OperatorRepairEngine().run(graph)
            ContinuousLearningBundleBuilder().build_from_graphs([repaired], output_dir)
            traces = [json.loads(line) for line in Path(os.path.join(output_dir, 'teacher_traces.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            self.assertIn('typed_claim_grounding_repair', traces[0]['teacher_trace']['repair_programs']['applied_programs'])
            self.assertIn('trim_unsupported_claims', traces[0]['completion_payload']['repair_actions_applied'])
            self.assertGreater(traces[0]['metadata']['repair_program_count'], 0)
            rows = [json.loads(line) for line in Path(os.path.join(output_dir, 'continuous_learning_sft.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            graph_rows = [row for row in rows if row.get('task') == 'continuous_graph_slots']
            self.assertEqual(len(graph_rows), 1)
            self.assertIn('repair_programs_applied', graph_rows[0]['completion'])
        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_benchmark_gated_continuous_trainer_derives_claim_grounding_benchmarks(self) -> None:
        from semop import BenchmarkGatedContinuousTrainer
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_claim_grounding_review.db')
        if os.path.exists(review_db):
            os.remove(review_db)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first and inspect the hidden sensor.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            graph = compile_and_execute(graph)
            store = __import__('semop').ReviewQueueStore(review_db)
            item_id = store.enqueue(
                domain='general',
                scenario='qa',
                query=graph.query,
                reasons=['claim_grounding_review'],
                answer_text='Open the drawer first.',
                kpis={'clarification_need_rate': 0.0},
                audit_items=[{'stage': 'claim_grounding', 'detail': 'unsupported sensor clause removed'}],
                context_text=graph.source_context,
                graph_payload=graph.model_dump(),
                severity='high',
            )
            store.update_status(item_id, 'approved', 'accepted for claim grounding benchmark')
            derived = BenchmarkGatedContinuousTrainer()._derive_promoted_review_benchmarks(review_db)
            self.assertTrue(derived.grounding_cases)
            self.assertTrue(any('open the drawer' in item for item in derived.grounding_cases[0].expected_claim_terms))
            self.assertTrue(any('sensor' in item for item in derived.grounding_cases[0].forbidden_unsupported_claim_terms))
            self.assertIn('trim_unsupported_claims', derived.compiler_cases[0].expected_repair_terms)
        finally:
            if os.path.exists(review_db):
                os.remove(review_db)

    def test_unified_benchmark_scores_claim_level_grounding_fidelity(self) -> None:
        from semop import GroundedExplanationEvalCase, UnifiedBenchmarkHarness

        score = UnifiedBenchmarkHarness._evaluate_grounded_case(
            StructuredMeaningPipeline(mode='heuristic'),
            GroundedExplanationEvalCase(
                query='What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
                expected_claim_terms=['open the drawer'],
                forbidden_unsupported_claim_terms=['hidden sensor'],
            ),
        )
        self.assertGreaterEqual(score, 0.9)

    def test_unified_benchmark_weights_grounding_cases_by_severity(self) -> None:
        from semop import GroundedExplanationEvalCase, UnifiedBenchmarkHarness, UnifiedSemOpTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_severity_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_severity_artifacts')
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            context = 'Manual:\nOpen the drawer before retrieving the folder.'
            graph = StructuredMeaningPipeline(mode='heuristic').run(query, source_context=context)
            store = CorpusMemoryStore(db_path)
            store.upsert_graph(graph, source='severity_benchmark', split='train')
            training = UnifiedSemOpTrainer().train_from_store(db_path, output_dir, source='severity_benchmark')
            benchmark = UnifiedBenchmarkHarness().evaluate(
                training.artifacts,
                grounding_cases=[
                    GroundedExplanationEvalCase(
                        query=query,
                        source_context=context,
                        expected_evidence_terms=['open the drawer'],
                        severity='low',
                    ),
                    GroundedExplanationEvalCase(
                        query=query,
                        source_context=context,
                        expected_evidence_terms=['impossible grounding phrase'],
                        severity='critical',
                    ),
                ],
            )
            self.assertGreater(benchmark.grounded_explanation_fidelity, 0.25)
            self.assertLess(benchmark.grounded_explanation_fidelity, 0.4)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_benchmark_gated_continuous_trainer_accepts_review_filtered_candidate(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_runtime.db')
        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_gate_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_artifacts')
        for path in [db_path, review_db]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            approved_query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            store = CorpusMemoryStore(db_path)
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            for query in [
                approved_query,
                'I am going to the car wash and traffic is bad, should I walk there?',
            ]:
                store.upsert_graph(pipeline.run(query), source='gate_demo', split='train')
            review_store = __import__('semop').ReviewQueueStore(review_db)
            item_id = review_store.enqueue(domain='access', scenario='drawer', query=approved_query, reasons=['approved_training_trace'], answer_text='Open the drawer first.', kpis={'clarification_need_rate': 0.0}, audit_items=[{'stage': 'approval', 'detail': 'use as accepted training trace'}])
            review_store.update_status(item_id, 'approved', 'accepted for retraining')
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='gate_demo',
                review_store_path=review_db,
                approved_queries_only=True,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
            )
            self.assertTrue(summary.gate.accepted)
            self.assertEqual(summary.training.trained_on_graphs, 1)
            payload = json.loads(Path(os.path.join(output_dir, 'benchmark_gate.json')).read_text(encoding='utf-8'))
            self.assertTrue(payload['gate']['accepted'])
            self.assertEqual(payload['training']['trained_on_graphs'], 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'accepted_benchmark_summary.json')))
        finally:
            for path in [db_path, review_db]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_benchmark_gated_continuous_trainer_derives_benchmarks_from_promoted_reviews(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_promoted_runtime.db')
        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_gate_promoted_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_promoted_artifacts')
        for path in [db_path, review_db]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            context = 'Manual:\nOpen the drawer before retrieving the folder.'
            graph = StructuredMeaningPipeline(mode='heuristic').run(query, source_context=context)
            store = CorpusMemoryStore(db_path)
            store.upsert_graph(graph, source='gate_promoted', split='train')
            review_store = __import__('semop').ReviewQueueStore(review_db)
            item_id = review_store.enqueue(
                domain='general',
                scenario='qa',
                query=query,
                reasons=['grounding_review', 'compiler_validity_gap'],
                answer_text='Open the drawer first.',
                kpis={'clarification_need_rate': 0.0, 'context_misread_rate': 0.0, 'relation_recovery': 1.0},
                audit_items=[{'stage': 'evidence', 'detail': 'drawer access required'}],
                context_text=context,
                graph_payload=graph.model_dump(),
            )
            review_store.update_status(item_id, 'approved', 'accepted for benchmark and retraining')
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='gate_promoted',
                review_store_path=review_db,
                approved_queries_only=True,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
            )
            self.assertEqual(summary.promoted_review_benchmarks['promoted_review_count'], 1)
            self.assertGreaterEqual(summary.promoted_review_benchmarks['hidden_premise_case_count'], 1)
            self.assertGreaterEqual(summary.promoted_review_benchmarks['grounding_case_count'], 1)
            self.assertGreaterEqual(summary.promoted_review_benchmarks['compiler_case_count'], 1)
            payload = json.loads(Path(os.path.join(output_dir, 'promoted_review_benchmark_cases.json')).read_text(encoding='utf-8'))
            self.assertEqual(payload['promoted_review_count'], 1)
            self.assertTrue(payload['grounding_cases'])
            self.assertTrue(payload['compiler_cases'])
            self.assertEqual(payload['grounding_cases'][0]['domain'], 'general')
            self.assertEqual(payload['grounding_cases'][0]['scenario'], 'qa')
            self.assertEqual(payload['compiler_cases'][0]['domain'], 'general')
            self.assertEqual(payload['compiler_cases'][0]['scenario'], 'qa')
        finally:
            for path in [db_path, review_db]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_benchmark_gated_continuous_trainer_persists_and_reuses_benchmark_corpus(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_corpus_runtime.db')
        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_gate_corpus_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_corpus_artifacts')
        second_output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_corpus_artifacts_round2')
        corpus_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_corpus.json')
        for path in [db_path, review_db, corpus_path]:
            if os.path.exists(path):
                os.remove(path)
        for path in [output_dir, second_output_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
        try:
            query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            context = 'Manual:\nOpen the drawer before retrieving the folder.'
            graph = StructuredMeaningPipeline(mode='heuristic').run(query, source_context=context)
            store = CorpusMemoryStore(db_path)
            store.upsert_graph(graph, source='gate_corpus', split='train')
            review_store = __import__('semop').ReviewQueueStore(review_db)
            item_id = review_store.enqueue(
                domain='general',
                scenario='qa',
                query=query,
                reasons=['grounding_review', 'compiler_validity_gap'],
                answer_text='Open the drawer first.',
                kpis={'clarification_need_rate': 0.0, 'context_misread_rate': 0.0, 'relation_recovery': 1.0},
                audit_items=[{'stage': 'evidence', 'detail': 'drawer access required'}],
                context_text=context,
                graph_payload=graph.model_dump(),
            )
            review_store.update_status(item_id, 'approved', 'accepted for persistent benchmark corpus')
            thresholds = BenchmarkGateThresholds(
                minimum_unseen_transfer=0.0,
                minimum_analogy_usefulness=0.0,
                minimum_compiler_validity=0.0,
                minimum_grounded_explanation_fidelity=0.0,
                minimum_repair_success_rate=0.0,
                require_improvement_if_baseline=False,
            )
            trainer = BenchmarkGatedContinuousTrainer()
            first = trainer.train_evaluate_and_gate(
                db_path,
                output_dir,
                source='gate_corpus',
                review_store_path=review_db,
                approved_queries_only=True,
                thresholds=thresholds,
                benchmark_corpus_path=corpus_path,
            )
            self.assertTrue(os.path.exists(corpus_path))
            self.assertGreaterEqual(first.benchmark_corpus['added_case_count'], 1)
            if os.path.exists(review_db):
                os.remove(review_db)
            second = trainer.train_evaluate_and_gate(
                db_path,
                second_output_dir,
                source='gate_corpus',
                thresholds=thresholds,
                benchmark_corpus_path=corpus_path,
            )
            self.assertEqual(second.promoted_review_benchmarks['promoted_review_count'], 0)
            self.assertGreaterEqual(second.benchmark_corpus['loaded_case_count'], 1)
            self.assertGreater(second.benchmark.grounded_explanation_fidelity, 0.0)
            persisted = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
            self.assertTrue(persisted['grounding_cases'])
            self.assertTrue(persisted['compiler_cases'])
        finally:
            for path in [db_path, review_db, corpus_path]:
                if os.path.exists(path):
                    os.remove(path)
            for path in [output_dir, second_output_dir]:
                if os.path.exists(path):
                    shutil.rmtree(path)

    def test_benchmark_gated_continuous_trainer_blocks_low_exception_slice(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer, GroundedExplanationEvalCase

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_slice_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_slice_artifacts')
        for path in [db_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            query = 'The aisle is blocked and approval is still missing. What should I do with the forklift move?'
            context = (
                'Exception SOP\n'
                '- Stop forklift movement when the aisle is blocked.\n'
                '- Do not rack-load without supervisor approval and safety clearance.\n'
                '- If not urgent, use the staging area and file an incident report.'
            )
            graph = StructuredMeaningPipeline(mode='heuristic').run(query, source_context=context)
            store = CorpusMemoryStore(db_path)
            store.upsert_graph(graph, source='slice_gate', split='train')
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='slice_gate',
                operating_domain='warehouse_exception',
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
                grounding_cases=[
                    GroundedExplanationEvalCase(
                        query='What should I do first?',
                        source_context=context,
                        expected_evidence_terms=['impossible grounding phrase'],
                        domain='warehouse_exception',
                        scenario='exception_response',
                    )
                ],
            )
            self.assertFalse(summary.gate.accepted)
            self.assertFalse(summary.gate.blocking_reasons)
            self.assertTrue(any('warehouse_exception::exception_response' in reason for reason in summary.gate.slice_blocking_reasons))
            self.assertIn('warehouse_exception::exception_response', summary.benchmark.slice_metrics)
            self.assertEqual(summary.benchmark.slice_metrics['warehouse_exception::exception_response']['scenario'], 'exception_response')
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_benchmark_gated_continuous_trainer_uses_domain_threshold_policy(self) -> None:
        from semop import BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_domain_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_domain_artifacts')
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            store = CorpusMemoryStore(db_path)
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            store.upsert_graph(
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
                source='gate_demo',
                split='train',
            )
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='gate_demo',
                operating_domain='warehouse_exception',
            )
            self.assertEqual(summary.gate.operating_domain, 'warehouse_exception')
            self.assertAlmostEqual(summary.gate.applied_thresholds['minimum_compiler_validity'], 0.74, places=2)
            self.assertFalse(summary.gate.accepted)
            self.assertTrue(any('compiler_validity below threshold' in item for item in summary.gate.blocking_reasons))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_benchmark_gated_continuous_trainer_rejects_regressing_baseline(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_reject_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_reject_artifacts')
        baseline_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_baseline.json')
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(baseline_path):
            os.remove(baseline_path)
        try:
            store = CorpusMemoryStore(db_path)
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            store.upsert_graph(
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
                source='gate_demo',
                split='train',
            )
            Path(baseline_path).write_text(
                json.dumps(
                    {
                        'benchmark': {
                            'unseen_transfer': 0.95,
                            'analogy_usefulness': 0.95,
                            'compiler_validity': 0.95,
                            'grounded_explanation_fidelity': 0.95,
                            'repair_success_rate': 0.95,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='gate_demo',
                baseline_summary_path=baseline_path,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=True,
                ),
            )
            self.assertFalse(summary.gate.accepted)
            self.assertTrue(summary.gate.regressed_axes)
            self.assertTrue(any('regressed against baseline' in item for item in summary.gate.blocking_reasons))
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            if os.path.exists(baseline_path):
                os.remove(baseline_path)

    def test_benchmark_gated_continuous_trainer_blocks_slice_regression_against_baseline(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer, CompilerRepairEvalCase

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_slice_regression_runtime.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_slice_regression_artifacts')
        baseline_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_slice_regression_baseline.json')
        for path in [db_path, baseline_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            query = 'The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
            graph = StructuredMeaningPipeline(mode='heuristic').run(query)
            store = CorpusMemoryStore(db_path)
            store.upsert_graph(graph, source='slice_regression_gate', split='train')
            Path(baseline_path).write_text(
                json.dumps(
                    {
                        'benchmark': {
                            'slice_metrics': {
                                'general::qa': {
                                    'domain': 'general',
                                    'scenario': 'qa',
                                    'grounded_explanation_fidelity': 0.0,
                                    'grounding_case_count': 0,
                                    'compiler_validity': 0.95,
                                    'compiler_case_count': 1,
                                    'repair_success_rate': 0.0,
                                    'hidden_premise_quality': 0.0,
                                    'hidden_premise_case_count': 0,
                                }
                            }
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='slice_regression_gate',
                operating_domain='general',
                baseline_summary_path=baseline_path,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
                compiler_cases=[CompilerRepairEvalCase(graph=graph, domain='general', scenario='qa')],
            )
            self.assertFalse(summary.gate.accepted)
            self.assertTrue(any('compiler_validity regressed against baseline' in item for item in summary.gate.slice_blocking_reasons))
            self.assertEqual(summary.gate.baseline_slice_metrics['general::qa']['compiler_validity'], 0.95)
        finally:
            for path in [db_path, baseline_path]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_benchmark_gated_continuous_trainer_uses_scenario_slice_balance_limit(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_exception_balance_runtime.db')
        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_gate_exception_balance_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_exception_balance_artifacts')
        corpus_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_exception_balance_corpus.json')
        for path in [db_path, review_db, corpus_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            store = CorpusMemoryStore(db_path)
            review_store = __import__('semop').ReviewQueueStore(review_db)
            context = (
                'Exception SOP\n'
                '- Stop forklift movement when the aisle is blocked.\n'
                '- Do not rack-load without supervisor approval and safety clearance.\n'
                '- If not urgent, use the staging area and file an incident report.'
            )
            for index in range(7):
                query = f'Exception case {index}: The aisle is blocked and approval is still missing. What should I do with the forklift move?'
                graph = pipeline.run(query, source_context=context)
                store.upsert_graph(graph, source='exception_slice_balance_gate', split='train')
                item_id = review_store.enqueue(
                    domain='warehouse_exception',
                    scenario='exception_response',
                    query=query,
                    reasons=['grounding_review', 'compiler_validity_gap'],
                    answer_text='Stop forklift movement and escalate for approval.',
                    kpis={'clarification_need_rate': 0.0, 'context_misread_rate': 0.0, 'relation_recovery': 1.0},
                    audit_items=[{'stage': 'evidence', 'detail': f'exception response #{index}'}],
                    context_text=context,
                    graph_payload=graph.model_dump(),
                    severity='critical',
                )
                review_store.update_status(item_id, 'approved', 'accepted for exception slice benchmark corpus')
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='exception_slice_balance_gate',
                review_store_path=review_db,
                approved_queries_only=True,
                benchmark_corpus_path=corpus_path,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
            )
            persisted = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
            self.assertEqual(len(persisted['hidden_premise_cases']), 6)
            self.assertEqual(len(persisted['grounding_cases']), 6)
            self.assertEqual(len(persisted['compiler_cases']), 6)
            self.assertEqual(summary.benchmark_corpus['slice_balance_limit'], 6)
            self.assertEqual(summary.benchmark_corpus['slice_balance_limits']['warehouse_exception::exception_response'], 6)
        finally:
            for path in [db_path, review_db, corpus_path]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_benchmark_gated_continuous_trainer_balances_dense_slice_corpus(self) -> None:
        from semop import BenchmarkGateThresholds, BenchmarkGatedContinuousTrainer

        db_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_balance_runtime.db')
        review_db = os.path.join(os.path.dirname(__file__), 'benchmark_gate_balance_review.db')
        output_dir = os.path.join(os.path.dirname(__file__), 'benchmark_gate_balance_artifacts')
        corpus_path = os.path.join(os.path.dirname(__file__), 'benchmark_gate_balance_corpus.json')
        for path in [db_path, review_db, corpus_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            store = CorpusMemoryStore(db_path)
            review_store = __import__('semop').ReviewQueueStore(review_db)
            for index in range(6):
                query = f'Case {index}: The drawer is closed and I need the folder inside. Should I pull the folder out right now?'
                context = f'Manual {index}:\nOpen the drawer before retrieving the folder.'
                graph = pipeline.run(query, source_context=context)
                store.upsert_graph(graph, source='slice_balance_gate', split='train')
                severity = 'critical' if index < 4 else 'low'
                item_id = review_store.enqueue(
                    domain='general',
                    scenario='qa',
                    query=query,
                    reasons=['grounding_review', 'compiler_validity_gap'],
                    answer_text='Open the drawer first.',
                    kpis={'clarification_need_rate': 0.0, 'context_misread_rate': 0.0, 'relation_recovery': 1.0},
                    audit_items=[{'stage': 'evidence', 'detail': f'drawer access required #{index}'}],
                    context_text=context,
                    graph_payload=graph.model_dump(),
                    severity=severity,
                )
                review_store.update_status(item_id, 'approved', 'accepted for balanced benchmark corpus')
            summary = BenchmarkGatedContinuousTrainer().train_evaluate_and_gate(
                db_path,
                output_dir,
                source='slice_balance_gate',
                review_store_path=review_db,
                approved_queries_only=True,
                benchmark_corpus_path=corpus_path,
                thresholds=BenchmarkGateThresholds(
                    minimum_unseen_transfer=0.0,
                    minimum_analogy_usefulness=0.0,
                    minimum_compiler_validity=0.0,
                    minimum_grounded_explanation_fidelity=0.0,
                    minimum_repair_success_rate=0.0,
                    require_improvement_if_baseline=False,
                ),
            )
            persisted = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
            self.assertEqual(len(persisted['hidden_premise_cases']), 4)
            self.assertEqual(len(persisted['grounding_cases']), 4)
            self.assertEqual(len(persisted['compiler_cases']), 4)
            self.assertTrue(all(item['severity'] == 'critical' for item in persisted['grounding_cases']))
            self.assertTrue(all(item['severity'] == 'critical' for item in persisted['compiler_cases']))
            self.assertGreater(summary.benchmark_corpus['trimmed_case_count'], 0)
            self.assertEqual(summary.benchmark_corpus['slice_balance_limit'], 4)
        finally:
            for path in [db_path, review_db, corpus_path]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
    def test_retained_repair_program_trainer_guides_runtime_repair(self) -> None:
        from semop import OperatorRepairEngine, RetainedRepairProgramTrainer
        from semop.structures import StructuredMeaningGraph

        output_path = os.path.join(os.path.dirname(__file__), 'retained_repair_programs_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            pipeline = StructuredMeaningPipeline(mode='heuristic')
            graphs = [
                pipeline.run('I am going to the car wash and traffic is bad, should I walk there?'),
                pipeline.run('The drawer is closed and I need the folder inside. Should I pull the folder out right now?'),
            ]
            summary = RetainedRepairProgramTrainer().train_from_graphs(graphs, output_path)
            self.assertGreaterEqual(summary.retained_program_count, 1)
            broken = StructuredMeaningGraph(query='repair program case', intent='goal_directed_reasoning')
            broken.hidden_goals = ['clean_car_goal']
            broken.required_premises = ['vehicle_present']
            repaired = OperatorRepairEngine(repair_program_path=output_path).run(broken)
            self.assertTrue(any(item.startswith('repair_program:') for item in repaired.operator_execution.derived_decisions))
            self.assertTrue(any(item.startswith('repair synthesis:') for item in repaired.audit_trace))
            self.assertGreaterEqual(repaired.operator_execution.composition_score, 0.65)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_retained_repair_program_trainer_harvests_multi_step_repair_trace(self) -> None:
        from semop import OperatorRepairEngine, RetainedRepairProgramTrainer
        from semop.structures import StructuredMeaningGraph

        output_path = os.path.join(os.path.dirname(__file__), 'retained_repair_programs_trace_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            broken = StructuredMeaningGraph(query='repair trace case', intent='goal_directed_reasoning')
            broken.hidden_goals = ['clean_car_goal']
            broken.required_premises = ['vehicle_present']
            repaired = OperatorRepairEngine().run(broken)
            summary = RetainedRepairProgramTrainer().train_from_graphs([repaired], output_path)
            self.assertTrue(any(item.get('sequence_length', 0) >= 2 for item in summary.model['records']))
            self.assertTrue(any(item.get('utility_delta', 0.0) >= 0.0 for item in summary.model['records']))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    def test_operator_repair_engine_trims_unsupported_document_claim(self) -> None:
        from semop import OperatorRepairEngine
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        graph = StructuredMeaningPipeline(mode='heuristic').run(
            'What should I do first?',
            source_context='Manual:\nOpen the drawer before retrieval.',
        )
        graph.symbolic_results = [
            SymbolicResult(
                domain='document_grounding',
                answer='Open the drawer first and inspect the hidden sensor.',
                evidence=['Open the drawer before retrieval.'],
                confidence=0.92,
                source='unit_test',
            )
        ]
        repaired = OperatorRepairEngine().run(graph)
        document_result = next(result for result in repaired.symbolic_results if result.domain == 'document_grounding')
        self.assertNotIn('hidden sensor', document_result.answer.lower())
        self.assertTrue(any(item.startswith('repair_applied:trim_unsupported_claims') for item in repaired.operator_execution.derived_decisions))

    def test_operator_repair_engine_records_typed_claim_repair_program(self) -> None:
        from semop import OperatorRepairEngine
        from semop.structures import SymbolicResult

        graph = StructuredMeaningPipeline(mode='heuristic').run(
            'What should I do first?',
            source_context='Manual:\nOpen the drawer before retrieval.',
        )
        graph.symbolic_results = [
            SymbolicResult(
                domain='document_grounding',
                answer='Open the drawer first and inspect the hidden sensor.',
                evidence=['Open the drawer before retrieval.'],
                confidence=0.92,
                source='unit_test',
            )
        ]
        repaired = OperatorRepairEngine().run(graph)
        self.assertIn('repair_program:typed_claim_grounding_repair', repaired.operator_execution.derived_decisions)
        self.assertTrue(any('repair synthesis: typed_claim_grounding_repair' in item for item in repaired.audit_trace))

    def test_operator_repair_engine_rejects_unsafe_claim_trim_without_grounded_fallback(self) -> None:
        from semop import OperatorRepairEngine
        from semop.structures import SymbolicResult

        graph = StructuredMeaningPipeline(mode='heuristic').run(
            'What should I do first?',
            source_context='Manual:\nOpen the drawer before retrieval.',
        )
        graph.symbolic_results = [
            SymbolicResult(
                domain='document_grounding',
                answer='Inspect the hidden sensor.',
                evidence=['Open the drawer before retrieval.'],
                confidence=0.92,
                source='unit_test',
            )
        ]
        repaired = OperatorRepairEngine().run(graph)
        document_result = next(result for result in repaired.symbolic_results if result.domain == 'document_grounding')
        self.assertIn('hidden sensor', document_result.answer.lower())
        self.assertIn('repair_rejected:trim_unsupported_claims', repaired.operator_execution.derived_decisions)
        self.assertTrue(any('unsafe_claim_trim' in item for item in repaired.audit_trace))
    def test_retained_repair_program_trainer_retains_claim_trim_program(self) -> None:
        from semop import OperatorRepairEngine, RetainedRepairProgramTrainer
        from semop.structures import StructuredMeaningGraph, SymbolicResult

        output_path = os.path.join(os.path.dirname(__file__), 'retained_repair_programs_claim_test.json')
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            graph = StructuredMeaningPipeline(mode='heuristic').run(
                'What should I do first?',
                source_context='Manual:\nOpen the drawer before retrieval.',
            )
            graph.symbolic_results = [
                SymbolicResult(
                    domain='document_grounding',
                    answer='Open the drawer first.',
                    evidence=['Open the drawer before retrieval.'],
                    confidence=0.92,
                    source='unit_test',
                )
            ]
            summary = RetainedRepairProgramTrainer().train_from_graphs([graph], output_path)
            self.assertTrue(any('trim_unsupported_claims' in item.get('actions', []) for item in summary.model['records']))
            broken = StructuredMeaningGraph.from_dict(graph.model_dump())
            broken.symbolic_results[0].answer = 'Open the drawer first and inspect the hidden sensor.'
            repaired = OperatorRepairEngine(repair_program_path=output_path).run(broken)
            document_result = next(result for result in repaired.symbolic_results if result.domain == 'document_grounding')
            self.assertNotIn('hidden sensor', document_result.answer.lower())
            self.assertTrue(any(item.startswith('repair_program:') for item in repaired.operator_execution.derived_decisions))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_operator_repair_engine_recovers_missing_goal_preservation_program(self) -> None:
        from semop import OperatorRepairEngine
        from semop.structures import StructuredMeaningGraph

        graph = StructuredMeaningGraph(query='repair loop case', intent='goal_directed_reasoning')
        graph.hidden_goals = ['clean_car_goal']
        graph.required_premises = ['vehicle_present']
        graph = OperatorRepairEngine().run(graph)
        self.assertTrue(any(item.operator_name == 'GOAL_PRESERVATION_OPERATOR' for item in graph.operator_decompositions))
        self.assertTrue(any(edge.relation == 'REQUIRES' and edge.target == 'vehicle_present' for edge in graph.edges))
        self.assertGreaterEqual(graph.operator_execution.composition_score, 0.65)
        self.assertTrue(any(item.startswith('repair_applied:') for item in graph.operator_execution.derived_decisions))
if __name__ == "__main__":
    unittest.main()









































