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
    BASELINE_SPECS,
    CopilotRequest,
    CorpusBuilder,
    CorpusMemoryStore,
    CorpusReasoningLearner,
    CURATED_PRESETS,
    CURATED_PUBLIC_DATASETS,
    DomainCopilot,
    EmbeddingModelCache,
    LabeledOpsEvaluator,
    MemoryPriorEvaluator,
    OperatorHierarchyLearner,
    PlainRagBaseline,
    PublicCorpusIngestor,
    PublicDatasetAdapter,
    RemoteDatasetDownloader,
    ResponseSynthesizer,
    ReviewQueueStore,
    StructuredMeaningPipeline,
    TransferEvaluator,
    curated_manifest,
    load_labeled_ops_cases,
    preset_manifest,
    resolve_embedding_model_id,
)
from semop.llm_client import LocalLLMConfig, LocalTransformersExtractor


class StructuredMeaningPipelineTests(unittest.TestCase):
    def test_bag_book_case_has_access_precondition(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("가방에 책을 넣으려면 어떻게 해야 하나요?")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("insert_book", "REQUIRES", "open_access"), relations)
        self.assertTrue(any("열" in step.action for step in graph.plan))

    def test_carwash_case_rejects_walk_only_advice(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("car_wash", "REQUIRES", "vehicle_present"), relations)
        self.assertTrue(any("걸어서" in text for text in graph.invalid_advice))

    def test_generic_case_produces_constraint_aware_plan(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("교통체증 때문에 세차장 이동이 어려워요.")
        self.assertTrue(any(edge.relation == "ALTERNATIVE" for edge in graph.edges))
        self.assertTrue(any(("대안" in step.rationale) or ("우회" in step.action) for step in graph.plan))

    def test_synthesizer_produces_human_readable_answer(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?")
        response = ResponseSynthesizer().synthesize(graph).to_text()
        self.assertIn("핵심 판단:", response)
        self.assertIn("논리적 근거:", response)
        self.assertIn("실행 계획:", response)
        self.assertIn("창의적 대안:", response)
        self.assertIn("유도된 연산자:", response)
        self.assertIn("유도된 문법 가설:", response)

    def test_induced_operators_and_grammar_exist(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?")
        self.assertTrue(graph.induced_operators)
        self.assertTrue(graph.grammar_hypotheses)
        self.assertTrue(any(candidate.family.startswith(("ALTERNATIVE_SEARCH", "REDIRECT", "PRECONDITION")) for candidate in graph.induced_operators))
        self.assertTrue(any(candidate.family.startswith(("ALTERNATIVE_SEARCH_", "ACTION_REWRITE_", "PRECONDITION_")) for candidate in graph.induced_operators if not candidate.family.startswith("SYMBOLIC_")))

    def test_llm_json_self_repair_handles_common_breakage(self) -> None:
        extractor = LocalTransformersExtractor(LocalLLMConfig())
        repaired = extractor._parse_with_repair("""```json
        {intent: 'test', entities: [], relations: [], constraints: [], scripts: [], candidate_actions: [], missing_knowledge: [],}
        ```""")
        self.assertEqual(repaired["intent"], "test")
        self.assertEqual(repaired["entities"], [])

    def test_corpus_learning_builds_reusable_families(self) -> None:
        learner = CorpusReasoningLearner(mode="heuristic")
        result = learner.learn_from_queries([
            "가방에 책을 넣으려면 어떻게 해야 하나요?",
            "가방이 꽉 차 있는데 책을 넣으려면 먼저 무엇을 봐야 하죠?",
            "세차장이 멀고 길이 막히는데 어떻게 가야 할까요?",
            "교통체증 때문에 세차장 가기가 어려운데 대안이 뭐가 있나요?",
        ])
        self.assertEqual(result.corpus_size, 4)
        self.assertTrue(result.learned_families)
        self.assertTrue(any(family.support >= 2 for family in result.learned_families))
        self.assertTrue(all(0.0 <= family.purity <= 1.0 for family in result.learned_families))

    def test_memory_store_round_trip(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("가방에 책을 넣으려면 어떻게 해야 하나요?")
        db_path = os.path.join(os.path.dirname(__file__), "memory_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(graph, source="test", split="train")
        loaded = store.fetch_graphs(split="train", source="test")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].query, graph.query)
        self.assertEqual(store.count_examples(split="train", source="test"), 1)

    def test_memory_retrieval_augments_pipeline(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "memory_retrieval_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

        base_pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        stored_graph = base_pipeline.run("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?")
        store.upsert_graph(stored_graph, source="demo", split="train")

        pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="demo")
        graph = pipeline.run("세차장이 멀고 교통체증도 심한데 다른 방법이 있을까요?")
        self.assertTrue(any("memory hint from similar query" in warning for warning in graph.warnings))
        self.assertTrue(any("memory prior promoted families" in warning for warning in graph.warnings))
        self.assertTrue(any(any(tag.startswith("memory_prior:") for tag in candidate.provenance) for candidate in graph.induced_operators))

    def test_transfer_evaluation_reports_reuse(self) -> None:
        evaluator = TransferEvaluator(mode="heuristic")
        result = evaluator.evaluate_queries([
            "가방에 책을 넣으려면 어떻게 해야 하나요?",
            "가방이 꽉 차 있는데 책을 넣으려면 먼저 무엇을 봐야 하죠?",
            "세차장이 멀고 길이 막히는데 어떻게 가야 할까요?",
            "차가 많이 막히는데 굳이 세차를 해야 하면 어떤 선택지가 있죠?",
            "교통체증 때문에 세차장 가기가 어려운데 대안이 뭐가 있나요?",
            "책이 큰데 가방에 넣으려면 어떤 순서로 해야 하나요?",
        ], train_ratio=0.67)
        self.assertGreaterEqual(result.train_size, 1)
        self.assertGreaterEqual(result.test_size, 1)
        self.assertTrue(0.0 <= result.family_reuse_rate <= 1.0)
        self.assertTrue(0.0 <= result.grammar_transfer_rate <= 1.0)
        self.assertTrue(result.per_family)

    def test_corpus_builder_expands_and_splits_queries(self) -> None:
        builder = CorpusBuilder(train_ratio=0.75)
        records = builder.build_from_queries([
            ("가방에 책을 넣으려면 어떻게 해야 하나요?", "seed"),
            ("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?", "seed"),
        ], augment=True)
        self.assertGreater(len(records), 2)
        self.assertTrue(any(record.augmented for record in records))
        self.assertTrue(all(record.split in {"train", "test"} for record in records))
        self.assertEqual(len({record.normalized_query for record in records}), len(records))

    def test_memory_store_fetch_queries_by_split(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "memory_query_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(pipeline.run("가방에 책을 넣으려면 어떻게 해야 하나요?"), source="demo", split="train")
        store.upsert_graph(pipeline.run("세차장이 멀고 길이 막히는데 어떻게 가야 할까요?"), source="demo", split="test")
        self.assertEqual(len(store.fetch_queries(split="train", source="demo")), 1)
        self.assertEqual(len(store.fetch_queries(split="test", source="demo")), 1)

    def test_public_dataset_adapter_loads_csv_and_json(self) -> None:
        adapter = PublicDatasetAdapter()
        csv_examples = adapter.load_path(os.path.join(os.path.dirname(__file__), "..", "examples", "public_reasoning_sample.csv"))
        json_examples = adapter.load_path(os.path.join(os.path.dirname(__file__), "..", "examples", "public_reasoning_sample.json"))
        self.assertEqual(len(csv_examples), 2)
        self.assertEqual(len(json_examples), 2)
        self.assertTrue(any("세차장" in example.query for example in csv_examples))
        self.assertTrue(any("옷장" in example.query for example in json_examples))

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
            self.assertTrue(artifact.dataset_paths)
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
                {
                    "name": "csv_demo",
                    "source": "csv_demo",
                    "urls": [csv_path.as_uri()],
                    "query_fields": ["question"],
                    "context_fields": ["context"],
                    "answer_fields": ["answer"],
                    "augment": False,
                },
                {
                    "name": "json_demo",
                    "source": "json_demo",
                    "urls": [json_path.as_uri()],
                    "query_fields": ["question", "instruction"],
                    "context_fields": ["context"],
                    "answer_fields": ["answer", "output"],
                    "augment": False,
                },
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
            store = CorpusMemoryStore(store_path)
            self.assertGreaterEqual(store.count_examples(source="csv_demo"), 2)
            self.assertGreaterEqual(store.count_examples(source="json_demo"), 2)
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
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        train_query = "세차장이 멀고 길이 막히는데 어떻게 가야 할까요?"
        test_query = "세차장이 멀고 교통체증도 심한데 다른 방법이 있을까요?"
        store.upsert_graph(pipeline.run(train_query), source="demo", split="train")
        store.upsert_graph(pipeline.run(test_query), source="demo", split="test")

        evaluator = MemoryPriorEvaluator(mode="heuristic")
        result = evaluator.evaluate_store(db_path, source="demo", max_test_queries=10, sample_queries=3)
        self.assertEqual(result.train_size, 1)
        self.assertEqual(result.test_size, 1)
        self.assertTrue(0.0 <= result.baseline_family_reuse_rate <= 1.0)
        self.assertTrue(0.0 <= result.memory_family_reuse_rate <= 1.0)
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
        response = ResponseSynthesizer().synthesize(graph).to_text()
        self.assertIn("Symbolic math answer:", response)

    def test_symbolic_document_grounding_extracts_evidence(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = "The office closes at 6 PM. The support desk closes at 5 PM. Which desk closes earlier?"
        graph = pipeline.run(query)
        self.assertTrue(any(result.domain == "document_grounding" for result in graph.symbolic_results))
        self.assertTrue(any(candidate.family == "SYMBOLIC_EVIDENCE" for candidate in graph.induced_operators))
        self.assertTrue(any("support desk closes at 5 PM" in result.answer for result in graph.symbolic_results))

    def test_symbolic_fraction_weighted_total_solver(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = (
            "A Statistics student wants to find out the average daily allowance of the middle school students. "
            "According to his survey, 2/3 of the students receive an average of $6 allowance per day while the rest gets an average of $4 a day. "
            "If he surveyed 60 students, what is the total amount of money those 60 students get in a day?"
        )
        graph = pipeline.run(query)
        self.assertTrue(any("320" in result.answer for result in graph.symbolic_results))

    def test_symbolic_pack_round_up_solver(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = (
            "It is Roger's turn to provide a snack for the baseball team after the game and he has decided to bring trail mix. "
            "The trail mix comes in packs of 6 individual pouches. Roger has 13 members on his baseball team, plus 3 coaches and 2 helpers. "
            "How many packs of trail mix does he need to buy?"
        )
        graph = pipeline.run(query)
        self.assertTrue(any("3 packs" in result.answer for result in graph.symbolic_results))

    def test_symbolic_finance_document_grounding_prefers_relevant_sentences(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = (
            "Revenue was $10 million in fiscal 2024. Operating income was $2 million. "
            "Cash flow from operations was $3 million. What was the operating income?"
        )
        graph = pipeline.run(query)
        self.assertTrue(any(result.domain == "document_grounding" for result in graph.symbolic_results))
        self.assertTrue(any("Operating income was $2 million" in result.answer for result in graph.symbolic_results))

    def test_symbolic_table_document_grounding_uses_table_row(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = """INCOME STATEMENT
Metric | 2023 | 2024
Revenue | 8 | 10
Operating income | 1 | 2
What was operating income in 2024?"""
        graph = pipeline.run(query)
        self.assertTrue(any(result.domain == "document_grounding" for result in graph.symbolic_results))
        self.assertTrue(any("Operating income" in result.answer and "2024=2" in result.answer for result in graph.symbolic_results))

    def test_symbolic_arithmetic_parser_handles_nested_expressions(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        query = (
            "A store sells 3 notebooks that cost $4 each and 2 pens that cost $1.5 each. "
            "How much will each person pay if 3 people split the bill equally?"
        )
        graph = pipeline.run(query)
        self.assertTrue(any("5" in result.answer for result in graph.symbolic_results))

    def test_operator_hierarchy_learner_builds_l1_l2_l3_nodes(self) -> None:
        learner = OperatorHierarchyLearner(mode="heuristic")
        result = learner.learn_from_queries([
            "What should I check before putting a book into a bag?",
            "How do I put a book into a bag?",
            "Five friends eat at a fast-food chain and order the following: 5 pieces of hamburger that cost $3 each; 4 sets of French fries that cost $1.20; 5 cups of soda that cost $0.5 each; and 1 platter of spaghetti that cost $2.7. How much will each of them pay if they will split the bill equally?",
            "Revenue was $10 million in fiscal 2024. Operating income was $2 million. Cash flow from operations was $3 million. What was the operating income?",
        ])
        self.assertEqual(result.corpus_size, 4)
        self.assertTrue(result.micro_nodes)
        self.assertTrue(result.family_nodes)
        self.assertTrue(result.abstract_nodes)
        self.assertTrue(result.composition_patterns)
        self.assertTrue(any(node.level == "L3" for node in result.abstract_nodes))

    def test_operator_hierarchy_result_persists_to_memory_store(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "hierarchy_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        store.upsert_graph(pipeline.run("How do I put a book into a bag?"), source="hierarchy_demo", split="train")
        store.upsert_graph(pipeline.run("Traffic is heavy and the car wash is far away. What should I do?"), source="hierarchy_demo", split="train")
        store.upsert_graph(pipeline.run("What should I check before putting a book into a bag?"), source="hierarchy_demo", split="train")

        learner = OperatorHierarchyLearner(mode="heuristic")
        result = learner.learn_from_graphs(store.fetch_graphs(split="train", source="hierarchy_demo"))
        run_id = store.store_hierarchy_result(result, source="hierarchy_demo", split="train")
        summary = store.fetch_latest_hierarchy_summary(source="hierarchy_demo", split="train")

        self.assertGreater(run_id, 0)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["corpus_size"], 3)
        self.assertTrue(summary["family_nodes"])

    def test_registry_attaches_abstract_parents_and_promotes_them(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "registry_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        base_pipeline = StructuredMeaningPipeline(mode="heuristic")
        store = CorpusMemoryStore(db_path)
        for query in [
            "How do I put a book into a bag?",
            "Traffic is heavy and the car wash is far away. What should I do?",
            "What should I check before putting a book into a bag?",
        ]:
            store.upsert_graph(base_pipeline.run(query), source="registry_demo", split="train")

        learner = OperatorHierarchyLearner(mode="heuristic")
        result = learner.learn_from_graphs(store.fetch_graphs(split="train", source="registry_demo"))
        store.store_hierarchy_result(result, source="registry_demo", split="train")

        pipeline = StructuredMeaningPipeline(mode="heuristic", memory_store_path=db_path, memory_source="registry_demo")
        graph = pipeline.run("Traffic is heavy and the car wash is far away. Is there another option?")
        self.assertTrue(any(candidate.abstract_parents for candidate in graph.induced_operators if not candidate.family.startswith("SYMBOLIC_")))
        self.assertTrue(any(("registry attached abstract parents" in warning) or ("memory prior promoted abstract operators" in warning) for warning in graph.warnings))


    def test_ops_warehouse_exception_case_recovers_blockers_and_prerequisites(self) -> None:
        pipeline = StructuredMeaningPipeline(mode="heuristic")
        graph = pipeline.run("지게차로 팔레트를 랙에 올리려는데 통로가 막혀 있고 승인도 없습니다. 어떻게 해야 하나요?")
        relations = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
        self.assertIn(("move_pallet_to_rack", "BLOCKED_BY", "blocked_aisle"), relations)
        self.assertIn(("move_pallet_to_rack", "REQUIRES", "supervisor_approval"), relations)
        self.assertTrue(any("승인 없이" in text for text in graph.invalid_advice))

    def test_domain_copilot_scores_ops_kpis_and_audit_trace(self) -> None:
        copilot = DomainCopilot(mode="heuristic")
        request = CopilotRequest(
            query="신입 작업자인데 바코드와 주문 라벨이 다르면 바로 포장해서 보내도 되나요?",
            context=(
                "창고 온보딩 SOP\n"
                "1. 피킹 후 바코드를 스캔한다.\n"
                "2. 주문 라벨과 피킹 티켓이 일치해야 포장을 진행한다.\n"
                "3. 라벨 불일치가 나오면 출고 보류 후 관리자 승인을 받는다."
            ),
            domain="warehouse_onboarding",
            scenario="onboarding",
        )
        result = copilot.run(request)
        self.assertEqual(result.graph.domain, "warehouse_onboarding")
        self.assertTrue(result.graph.audit_trace)
        self.assertGreaterEqual(result.kpis.relation_recovery, 0.5)
        self.assertGreater(result.kpis.human_audit_usefulness, 0.5)
        self.assertIn("운영 KPI:", result.to_text())

    def test_plain_rag_baseline_retrieves_relevant_context_chunk(self) -> None:
        baseline = PlainRagBaseline()
        context = (
            "출고 품질 SOP\n"
            "라벨 불일치가 발생하면 출고를 보류한다.\n"
            "재스캔으로 해결되지 않으면 supervisor approval을 받기 전까지 포장을 닫지 않는다."
        )
        result = baseline.answer("라벨이 다르면 그냥 보내도 되나요?", context, domain="warehouse_exception")
        self.assertTrue(result.retrieved_chunks)
        self.assertIn("출고를 보류", result.answer_text)

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
            try:
                os.remove(db_path)
            except PermissionError:
                pass
        copilot = DomainCopilot(mode="heuristic", review_queue_path=db_path)
        request = CopilotRequest(
            query="지게차로 팔레트를 랙에 올리려는데 통로가 막혀 있고 아직 승인도 안 났습니다. 어떻게 해야 하나요?",
            context=(
                "예외 대응 SOP\n"
                "- 막힌 통로에서는 지게차 이동을 즉시 중단한다.\n"
                "- 관리자 승인과 안전 확인 없이 랙 적재를 진행하지 않는다.\n"
                "- 긴급하지 않으면 스테이징 구역에 임시 보관하고 incident report를 남긴다."
            ),
            domain="warehouse_exception",
            scenario="exception_response",
        )
        result = copilot.run(request)
        queue = ReviewQueueStore(db_path)
        pending = queue.fetch_pending(limit=5)
        self.assertTrue(result.queued_for_review)
        self.assertTrue(pending)
        self.assertIn("invalid_advice_rate", pending[0].reasons)

    def test_compare_baseline_schema_supports_multiple_baselines(self) -> None:
        self.assertIn("lexical_rag", BASELINE_SPECS)
        self.assertIn("first_chunk", BASELINE_SPECS)

    def test_review_queue_can_update_status_and_fetch_stats(self) -> None:
        db_path = os.path.join(os.path.dirname(__file__), "review_queue_status_test.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
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
        self.assertEqual(detail["resolution_note"], "validated by supervisor")
        self.assertEqual(stats["approved"], 1)

    def test_labeled_ops_case_supports_clarification_field(self) -> None:
        cases = load_labeled_ops_cases(os.path.join(os.path.dirname(__file__), "..", "examples", "ops_labeled_eval_ko.jsonl"))
        self.assertTrue(any(hasattr(case, "expected_clarification") for case in cases))

    def test_configurable_keyword_baseline_can_be_used_in_labeled_eval(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "baseline_config_test.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({
                "name": "test_config",
                "top_k": 1,
                "required_any": ["??"],
                "preferred_order": ["??", "??"],
                "keyword_bonus": {"??": 2.0, "??": 1.5},
            }, handle, ensure_ascii=False)
        cases = load_labeled_ops_cases(os.path.join(os.path.dirname(__file__), "..", "examples", "ops_labeled_eval_ko.jsonl"))
        evaluator = LabeledOpsEvaluator(
            copilot=DomainCopilot(mode="heuristic"),
            baseline_name="configurable_keyword",
            baseline_config_path=config_path,
        )
        comparison = evaluator.evaluate_case(cases[1])
        self.assertIn("configurable_keyword", comparison)
        self.assertTrue(comparison["configurable_keyword"].answer_text)

    def test_feedback_rules_are_applied_back_into_copilot(self) -> None:
        rules_path = os.path.join(os.path.dirname(__file__), "feedback_rules_test.json")
        with open(rules_path, "w", encoding="utf-8") as handle:
            json.dump({
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
            }, handle, ensure_ascii=False)
        copilot = DomainCopilot(mode="heuristic", feedback_rules_path=rules_path)
        result = copilot.run(CopilotRequest(
            query="The aisle is blocked and approval is still missing. What should I do?",
            context="Exception SOP\nIf the aisle is blocked, stop the move and verify approval and safety first.",
            domain="warehouse_exception",
            scenario="exception_response",
        ))
        self.assertTrue(any("feedback rules applied" in warning for warning in result.graph.warnings))
        self.assertTrue(any("Check approval and safety conditions" in item for item in result.graph.creative_alternatives))


if __name__ == "__main__":
    unittest.main()





