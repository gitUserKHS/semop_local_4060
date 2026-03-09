from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semop import (
    AffordanceLabelDataset,
    AffordanceWeightTrainer,
    BASELINE_SPECS,
    CompetitiveProgrammingReasoner,
    CpCorpusBuilder,
    CpDslDatasetBuilder,
    CpEpisodeStore,
    CpKnowledgeLoader,
    CpParserEvaluator,
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
    GeometryTopologyExtractor,
    OpenImagesAnnotationAdapter,
    VisualCollectionSource,
    VisualDataCollector,
    VisualDownloadEntry,
    ImageMaskPreprocessor,
    VisualConceptLabelRecommender,
    VisualConceptLearningSummary,
    VisualConceptMemory,
    VisualConceptPrototypeTrainer,
    VisualConceptRecord,
    VisualAffordanceFeatureExtractor,
    VisualEmbeddingRecord,
    VisualEmbeddingStore,
    VisionEmbeddingExtractor,
    VisualGeometryReasoner,
    WeakAffordanceClassifier,
    DetectorOutputAdapter,
    HardProblemEngine,
    PatternOutcomeTrainer,
    LabeledOpsEvaluator,
    LogicalGrammarInducer,
    MemoryPriorEvaluator,
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


if __name__ == "__main__":
    unittest.main()









