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
    CpParserEvaluator,
    CpParserPrediction,
    VlsoReviewImpactEvaluator,
    CpParserTrainConfig,
    CpParserTrainingScaffold,
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
    OperatorAlgebraEvaluator,
    OperatorHierarchyLearner,
    OlympiadReasoner,
    PlainRagBaseline,
    PublicCorpusIngestor,
    PublicDatasetAdapter,
    RemoteDatasetDownloader,
    ResponseSynthesizer,
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
        graph = pipeline.run("세차장에 가는데 차가 막혀, 걸어갈까?")
        self.assertIn("clean_car_goal", graph.hidden_goals)
        self.assertIn("vehicle_present", graph.required_premises)
        self.assertTrue(graph.clarification_needed)
        self.assertTrue(any(check.action == "walk_without_car" and check.status == "risk_high" for check in graph.goal_preservation_checks))
        self.assertTrue(any("hidden premise:" in warning for warning in graph.warnings))

    def test_hidden_premise_explorer_keeps_booking_interpretation_conditional(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("세차장 예약 취소하러 가는데 차가 막혀, 걸어갈까?")
        self.assertIn("booking_or_inquiry_goal", graph.hidden_goals)
        self.assertTrue(graph.clarification_needed)
        self.assertTrue(any(check.action == "walk_without_car" and check.status == "conditionally_valid" for check in graph.goal_preservation_checks))
        self.assertIn("clean_car_goal", graph.optional_interpretations)

    def test_hidden_premise_evaluator_scores_goal_and_premise_recall(self) -> None:
        evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        summary = evaluator.evaluate([
            HiddenPremiseEvalCase(
                query="세차장에 가는데 차가 막혀, 걸어갈까?",
                expected_hidden_goals=["clean_car_goal"],
                expected_required_premises=["vehicle_present"],
                expected_risky_actions=["walk_without_car"],
            )
        ])
        self.assertEqual(summary.num_cases, 1)
        self.assertEqual(summary.critical_premise_recall, 1.0)
        self.assertEqual(summary.hidden_goal_recall, 1.0)
        self.assertEqual(summary.goal_preservation_accuracy, 1.0)

    def test_operator_algebra_decomposes_hidden_goal_reasoning(self) -> None:
        graph = StructuredMeaningPipeline(mode="heuristic").run("세차장에 가는데 차가 막혀, 걸어갈까?")
        names = {item.operator_name for item in graph.operator_decompositions}
        self.assertIn("GOAL_PRESERVATION_OPERATOR", names)
        self.assertIn("SERVICE_GOAL_OPERATOR", names)
        self.assertTrue(any(item.name == "ServiceGoalToConstraintFunctor" for item in graph.functor_hypotheses))

    def test_vlso_language_parser_projects_hidden_premises_into_world_model(self) -> None:
        world, graph = VLSOReasoner().language_parser.parse("세차장에 가는데 차가 막혀, 걸어갈까?")
        self.assertIn("clean_car_goal", world.goals)
        self.assertIn("vehicle_present", world.constraints)
        self.assertTrue(world.metadata.get('hidden_premises'))
        self.assertTrue(world.metadata.get('functor_hypotheses'))

    def test_operator_algebra_evaluator_scores_decomposition_and_functor_recall(self) -> None:
        evaluator = OperatorAlgebraEvaluator(StructuredMeaningPipeline(mode="heuristic"))
        summary = evaluator.evaluate([
            __import__('semop').OperatorAlgebraEvalCase(
                query="세차장에 가는데 차가 막혀, 걸어갈까?",
                expected_decompositions=["GOAL_PRESERVATION_OPERATOR", "SERVICE_GOAL_OPERATOR"],
                expected_functors=["ServiceGoalToConstraintFunctor"],
            )
        ])
        self.assertEqual(summary.num_cases, 1)
        self.assertEqual(summary.decomposition_recall, 1.0)
        self.assertEqual(summary.functor_recall, 1.0)

    def test_vlso_aligner_uses_hidden_goal_checks_for_cross_modal_warning(self) -> None:
        visual_payload = {
            'objects': [
                {'id': 'container', 'label': 'polygon_10', 'kind': 'shape', 'bbox': [20, 20, 180, 180], 'polygon': [[20, 40], [30, 20], [170, 20], [180, 40], [180, 170], [170, 180], [30, 180], [20, 170]]},
                {'id': 'opening_band', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [50, 24, 150, 44], 'polygon': [[50, 24], [150, 24], [150, 44], [50, 44]]},
                {'id': 'handle', 'label': 'polygon_6', 'kind': 'shape', 'bbox': [18, 70, 36, 150], 'polygon': [[18, 70], [36, 70], [36, 150], [18, 150]]},
            ]
        }
        world = VLSOReasoner(mode='deep').run("닫힌 가방에 책을 바로 넣어도 될까?", visual_payload)
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
        self.assertTrue(any(relation in {"HAS", "AFFORDS", "BEFORE"} for _, relation, _ in relations))
        self.assertTrue(any("logical relation prior injected" in warning for warning in graph.warnings))



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
            summary = VisualConceptSelfTrainer().train_candidates_jsonl(candidates_path, store_path, summary_output=summary_path)
            self.assertEqual(summary.cluster_count, 1)
            cluster_rows = json.loads(summary_path.read_text(encoding='utf-8'))['clusters']
            self.assertGreater(cluster_rows[0]['review_priority'], 0.0)
            self.assertIn('low margin', cluster_rows[0]['review_reason'])
        finally:
            shutil.rmtree(base_dir)

    def test_competitive_programming_reasoner_records_search_trace_for_geometry(self) -> None:
        result = CompetitiveProgrammingReasoner().solve('Given coordinates of three points, compute the area of the triangle they form.')
        self.assertIsNotNone(result)
        self.assertEqual(result.category, 'computational_geometry_analysis')
        self.assertEqual(result.selection_strategy, 'verifier_rerank_top3')
        self.assertTrue(result.search_trace)
        self.assertEqual(result.search_trace[0]['category'], 'computational_geometry_analysis')

    def test_vlso_question_answerer_prioritizes_geometry_queries(self) -> None:
        world = __import__('semop').SharedWorldModel(query='geometry')
        world.add_relation(__import__('semop').VLSORelation(source='line_ab', relation='PARALLEL', target='line_cd', modality='vision', confidence=0.8))
        world.add_relation(__import__('semop').VLSORelation(source='line_ab', relation='PERPENDICULAR', target='line_ef', modality='vision', confidence=0.8))
        answer = VLSOQuestionAnswerer().answer('What geometric structure is visible here?', world, answer_mode='structured')
        self.assertIn('parallel', answer.answer_text.lower())
        self.assertIn('perpendicular', answer.answer_text.lower())


    def test_vlso_question_answerer_summarizes_containers_and_parts(self) -> None:
        world = __import__('semop').SharedWorldModel(query='inventory')
        world.add_entity(__import__('semop').VLSOEntity(id='shape_1', label='main box', modality='vision', entity_type='container', attributes={'concept_labels': ['BOX_LIKE_CONTAINER', 'HAS_INTERIOR']}))
        world.add_entity(__import__('semop').VLSOEntity(id='shape_2', label='front handle', modality='vision', entity_type='part', attributes={'concept_labels': ['HANDLE_LIKE_PART', 'GRASPABLE_PART']}))
        answer = VLSOQuestionAnswerer().answer('What objects are visible here?', world, answer_mode='structured')
        self.assertIn('containers:', answer.answer_text)
        self.assertIn('parts/openings:', answer.answer_text)

    def test_build_object_family_manifest_expands_multiple_families(self) -> None:
        manifest = __import__('semop').build_object_family_manifest(['bag', 'door', 'tool'], limit_per_source=7)
        self.assertIn('sources', manifest)
        self.assertGreaterEqual(len(manifest['sources']), 6)
        families = {row['metadata']['family'] for row in manifest['sources']}
        self.assertEqual(families, {'bag', 'door', 'tool'})
        self.assertTrue(all(int(row['limit']) == 7 for row in manifest['sources']))


    def test_visual_data_collector_build_download_manifest_preserves_extension(self) -> None:
        import importlib
        data_collection = importlib.import_module('semop.vlso.data_collection')
        records = [data_collection.VisualCollectionRecord(provider='openverse', query='bag', title='bag', page_url='', media_url='https://example.com/file.png', source_id='abc')]
        manifest = VisualDataCollector().build_download_manifest(records, 'data/downloads')
        self.assertEqual(len(manifest), 1)
        self.assertTrue(manifest[0].target_path.endswith('.png'))

    def test_visual_download_summary_reports_download_counts(self) -> None:
        summary = VisualDataCollector().prepare_downloads(
            records_path='examples/cp_labeled_public_sample.jsonl',
            approved_output='tests/tmp_approved.jsonl',
            manifest_output='tests/tmp_manifest.jsonl',
            download_root='tests/tmp_downloads',
            allow_providers=['openverse'],
            accept_all=False,
            execute=False,
        )
        self.assertTrue(hasattr(summary, 'downloaded_count'))
        self.assertTrue(hasattr(summary, 'failed_downloads'))

    def test_visual_data_collector_retries_http_429_once(self) -> None:
        import importlib
        from unittest.mock import patch
        from urllib.error import HTTPError
        data_collection = importlib.import_module('semop.vlso.data_collection')

        class _FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
            def read(self) -> bytes:
                return self._payload
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        plan = VisualDataCollector().build_plan(VisualCollectionSource(provider='openverse', query='bag', limit=2))
        throttled = HTTPError(plan.request_url, 429, 'Too many requests', hdrs={}, fp=None)
        with patch.object(data_collection, 'urlopen', side_effect=[throttled, _FakeResponse(b'{"results": []}')]), patch.object(data_collection.time, 'sleep') as sleep_mock:
            rows = VisualDataCollector().fetch_and_normalize(plan)
        self.assertEqual(rows, [])
        self.assertTrue(sleep_mock.called)

    def test_visual_data_collector_family_batch_dry_run_builds_workspace_outputs(self) -> None:
        base_dir = Path(os.path.dirname(__file__)) / 'vlso_family_batch_test'
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        try:
            summary = VisualDataCollector().run_family_batch({'bag': 12, 'door': 8}, base_dir, execute_collect=False)
            self.assertTrue(Path(summary.manifest_path).exists())
            self.assertTrue(Path(summary.records_path).exists())
            self.assertEqual(summary.families, {'bag': 12, 'door': 8})
            self.assertGreaterEqual(summary.manifest_sources, 4)
            self.assertEqual(summary.approved_count, 0)
        finally:
            shutil.rmtree(base_dir)


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


if __name__ == "__main__":
    unittest.main()










