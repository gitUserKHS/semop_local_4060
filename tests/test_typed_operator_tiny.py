from __future__ import annotations

from math import sqrt
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRAIN = ROOT / "tools" / "train"
for path in (SRC, TRAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semop.kernel import (
    Goal,
    MdlMacroLibrary,
    OperatorKernel,
    TraceCorpus,
    build_decision_training_cases,
    generate_symbolic_curriculum,
    hard_negative_records,
    symbolic_curriculum_domains,
)
from semop.kernel.domains import make_hidden_premise_instance, parse_geometry_dsl
from semop.tiny_controller import (
    NumpyTinyController,
    TinyControllerConfig,
    canonicalize_problem,
)


class TinyControllerTests(unittest.TestCase):
    def test_controller_rejects_non_positive_greedy_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "score margin"):
            TinyControllerConfig(action_limit_score_margin=0.0)

    def test_multidomain_curriculum_is_balanced_and_replayable(self) -> None:
        curriculum = generate_symbolic_curriculum(
            2,
            seed=7,
            curriculum="language-math-vision",
        )

        self.assertEqual(len(curriculum), 6)
        counts = {
            domain: sum(item.domain == domain for item in curriculum)
            for domain in ("language", "math", "vision")
        }
        self.assertEqual(
            counts,
            {"language": 2, "math": 2, "vision": 2},
        )
        for synthetic in curriculum:
            result = OperatorKernel(synthetic.instance.registry).solve(
                synthetic.instance.state,
                synthetic.instance.goals,
            )
            self.assertTrue(result.success, synthetic.problem_id)
            self.assertTrue(result.verified, synthetic.problem_id)
            self.assertLessEqual(len(result.proof), 6)
            cases = build_decision_training_cases(
                OperatorKernel(synthetic.instance.registry), result
            )
            self.assertTrue(cases)
            self.assertTrue(all(len(case.actions) == 5 for case in cases))
            for case in cases:
                goal_predicates = {
                    goal.atom.predicate.name for goal in case.goals
                }
                for action_index, action in enumerate(case.actions):
                    if action_index == case.target_action:
                        continue
                    self.assertTrue(
                        any(
                            effect.predicate.name in goal_predicates
                            for effect in action.effects
                        )
                    )

    def test_curriculum_domain_filter_is_balanced_and_seed_invariant(self) -> None:
        complete = generate_symbolic_curriculum(
            2,
            seed=11,
            curriculum="language-math-vision",
        )
        math_only = generate_symbolic_curriculum(
            2,
            seed=11,
            curriculum="language-math-vision",
            domains=("math",),
        )

        complete_math = tuple(item for item in complete if item.domain == "math")
        self.assertEqual(
            [item.problem_id for item in complete_math],
            [item.problem_id for item in math_only],
        )
        self.assertEqual(
            [item.instance.metadata["answer"] for item in complete_math],
            [item.instance.metadata["answer"] for item in math_only],
        )
        self.assertEqual(
            symbolic_curriculum_domains("language-math-vision"),
            ("language", "math", "vision"),
        )
        with self.assertRaisesRegex(ValueError, "not in language-math-vision"):
            generate_symbolic_curriculum(
                1,
                curriculum="language-math-vision",
                domains=("grid",),
            )

    def test_composed_curriculum_generates_replayable_deep_programs(self) -> None:
        curriculum = generate_symbolic_curriculum(
            3,
            seed=19,
            curriculum="language-math-vision-composed",
            domains=("composed",),
        )

        self.assertEqual(len(curriculum), 3)
        self.assertEqual(
            symbolic_curriculum_domains("language-math-vision-composed"),
            ("language", "math", "vision", "composed"),
        )
        proof_lengths = []
        for synthetic in curriculum:
            kernel = OperatorKernel(synthetic.instance.registry)
            result = kernel.solve(
                synthetic.instance.state,
                synthetic.instance.goals,
            )
            cases = build_decision_training_cases(kernel, result)

            self.assertTrue(result.success and result.verified)
            self.assertTrue(all(len(case.actions) == 5 for case in cases))
            proof_lengths.append(len(result.proof))

        self.assertEqual(proof_lengths, [5, 5, 4])

    def test_composed_curriculum_trains_and_exports_numpy_policy(self) -> None:
        try:
            import torch  # noqa: F401
            from train_tiny_controller import train
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "composed_debug.npz"
            summary = train(
                output,
                examples_per_domain=1,
                epochs=1,
                seed=23,
                debug_small=True,
                curriculum="language-math-vision-composed",
                domains=("composed",),
            )

            self.assertTrue(output.exists())
            self.assertEqual(summary["trained_domains"], ["composed"])
            self.assertEqual(summary["verified_synthetic_traces"], 1)
            self.assertEqual(summary["decision_examples"], 5)
            self.assertEqual(summary["hard_negative_records"], 20)
            policy = NumpyTinyController.load(output)
            self.assertEqual(policy.parameter_count, summary["parameter_count"])

    def test_future_gold_actions_are_not_labeled_as_hard_negatives(self) -> None:
        problem = make_hidden_premise_instance(
            "deploy",
            ["tests", "approval"],
            satisfied=["tests", "approval"],
        )
        kernel = OperatorKernel(problem.registry)
        result = kernel.solve(problem.state, problem.goals)

        cases = build_decision_training_cases(kernel, result)

        self.assertGreater(len(result.proof), 1)
        self.assertTrue(all(len(case.actions) == 1 for case in cases))
        self.assertEqual(hard_negative_records(cases), ())
        self.assertTrue(
            any(
                goal.label == "operator_frontier"
                for goal in cases[0].goals
            )
        )
        self.assertEqual(
            sum(
                goal.label != "operator_frontier"
                for goal in cases[0].goals
            ),
            len(problem.goals),
        )

    def test_default_parameter_and_artifact_caps(self) -> None:
        config = TinyControllerConfig()
        count = config.estimated_parameter_count()
        self.assertGreaterEqual(count, 5_000_000)
        self.assertLessEqual(count, 8_000_000)
        self.assertLessEqual(count, 15_000_000)
        self.assertLessEqual(count * 4, 64 * 1024 * 1024)

    def test_canonical_features_do_not_depend_on_point_names(self) -> None:
        first = parse_geometry_dsl(
            "point A, B, M\nassume midpoint(M, A, B)\nprove collinear(A, M, B)\n"
        )
        second = parse_geometry_dsl(
            "point P, Q, X\nassume midpoint(X, P, Q)\nprove collinear(P, X, Q)\n"
        )
        first_graph = canonicalize_problem(first.state, first.goals)
        second_graph = canonicalize_problem(second.state, second.goals)

        self.assertEqual(first_graph.node_tokens, second_graph.node_tokens)
        self.assertEqual(first_graph.relations, second_graph.relations)
        self.assertEqual(first_graph.goals, second_graph.goals)

    def test_action_features_expose_shared_family_and_typed_effect_shape(self) -> None:
        problem = make_hidden_premise_instance(
            "deploy", ["approval"], satisfied=["approval"]
        )
        actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
        graph = canonicalize_problem(problem.state, problem.goals, actions)

        verify_action = next(
            action
            for action in graph.actions
            if action.operator == "verify_required_premise"
        )
        self.assertIn("family:verify", verify_action.operator_features)
        self.assertTrue(
            any(
                feature.startswith("effect:REQUIREMENT_MET")
                for feature in verify_action.operator_features
            )
        )
        self.assertTrue(
            any(
                feature.startswith("structure:effect_goal_predicate:")
                for feature in verify_action.operator_features
            )
        )

    def test_frontier_is_encoded_separately_from_verifier_goal(self) -> None:
        problem = parse_geometry_dsl(
            "point A, B, M\n"
            "assume midpoint(M, A, B)\n"
            "prove collinear(A, M, B)\n"
            "prove equal_length(segment(A, M), segment(M, B))\n"
        )
        goals = (
            problem.goals[1],
            Goal(problem.goals[0].atom, label="operator_frontier"),
        )
        actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)

        graph = canonicalize_problem(problem.state, goals, actions)
        by_operator = {action.operator: action for action in graph.actions}

        self.assertEqual({goal.role for goal in graph.goals}, {"goal", "frontier"})
        self.assertIn(
            "structure:effect_verifier_exact:true",
            by_operator["midpoint_implies_equal_lengths"].operator_features,
        )
        self.assertIn(
            "structure:effect_frontier_exact:true",
            by_operator["midpoint_implies_collinear"].operator_features,
        )
    def test_controller_features_hide_training_labels_and_expose_goal_match(self) -> None:
        synthetic = generate_symbolic_curriculum(
            1,
            seed=3,
            curriculum="language-math-vision",
        )[0]
        instance = synthetic.instance
        actions = OperatorKernel(instance.registry).enumerate_actions(instance.state)
        graph = canonicalize_problem(instance.state, instance.goals, actions)

        all_features = {
            feature for action in graph.actions for feature in action.operator_features
        }
        self.assertNotIn("tag:hard_negative", all_features)
        self.assertNotIn("tag:synthetic", all_features)

        geometry = parse_geometry_dsl(
            "point A, B, M\n"
            "assume midpoint(M, A, B)\n"
            "prove collinear(A, M, B)\n"
        )
        geometry_actions = OperatorKernel(geometry.registry).enumerate_actions(
            geometry.state
        )
        geometry_graph = canonicalize_problem(
            geometry.state, geometry.goals, geometry_actions
        )
        exact = next(
            action
            for action in geometry_graph.actions
            if action.operator == "midpoint_implies_collinear"
        )
        self.assertIn(
            "structure:effect_goal_exact:true",
            exact.operator_features,
        )
        self.assertEqual(len(exact.structural_values), 8)
        self.assertEqual(exact.structural_values[0], 1.0)
        self.assertEqual(exact.structural_values[3], 1.0)

    def test_numpy_runtime_round_trip_and_kernel_policy(self) -> None:
        config = TinyControllerConfig(
            d_model=32,
            token_buckets=128,
            relation_buckets=32,
            operator_buckets=32,
        )
        model = NumpyTinyController.random(config, seed=7)
        problem = parse_geometry_dsl(
            "point A, B, M\nassume midpoint(M, A, B)\nprove collinear(A, M, B)\n"
        )
        result = OperatorKernel(problem.registry).solve(
            problem.state, problem.goals, policy=model
        )
        self.assertTrue(result.success)
        self.assertTrue(result.verified)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.npz"
            model.save(path)
            loaded = NumpyTinyController.load(path)
            self.assertEqual(model.parameter_count, loaded.parameter_count)
            actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
            self.assertEqual(
                model.score_actions(problem.state, problem.goals, actions).action_scores,
                loaded.score_actions(problem.state, problem.goals, actions).action_scores,
            )

    def test_torch_training_and_numpy_inference_action_scores_match(self) -> None:
        import numpy as np
        import torch

        from semop.tiny_controller.training import (
            TorchTinyController,
            encode_training_example,
        )

        config = TinyControllerConfig(
            d_model=16,
            token_buckets=64,
            relation_buckets=16,
            operator_buckets=16,
        )
        runtime = NumpyTinyController.random(config, seed=11)
        problem = make_hidden_premise_instance(
            "deploy",
            ["tests", "approval"],
            satisfied=["tests", "approval"],
        )
        actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
        example = encode_training_example(
            problem.state,
            problem.goals,
            actions,
            target_action=0,
            target_halt=False,
            target_value=1.0,
            config=config,
        )
        torch_model = TorchTinyController.from_numpy(runtime)
        torch_model.eval()
        with torch.no_grad():
            torch_scores = torch_model(example)["action_logits"].cpu().numpy()
        numpy_scores = np.asarray(
            runtime.score_actions(
                problem.state,
                problem.goals,
                actions,
            ).action_scores
        )

        np.testing.assert_allclose(torch_scores, numpy_scores, rtol=1e-5, atol=1e-6)

    def test_v3_structure_head_artifact_is_score_preserving_migrated(self) -> None:
        import numpy as np

        config = TinyControllerConfig(
            d_model=16,
            token_buckets=64,
            relation_buckets=16,
            operator_buckets=16,
        )
        model = NumpyTinyController.random(config, seed=2)
        old_weight = np.full((8,), 0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.npz"
            legacy = Path(directory) / "legacy-v3.npz"
            model.save(current)
            with np.load(current, allow_pickle=False) as payload:
                migrated_payload = {
                    name: payload[name] for name in payload.files
                }
            migrated_payload["format_version"] = np.asarray(3, dtype=np.int32)
            migrated_payload["action_structure_weight"] = old_weight
            np.savez_compressed(legacy, **migrated_payload)
            loaded = NumpyTinyController.load(legacy)

        np.testing.assert_allclose(
            loaded.weights["action_structure_weight"],
            old_weight / sqrt(config.d_model),
        )

    def test_verified_trace_caps_and_hard_negatives(self) -> None:
        problem = make_hidden_premise_instance(
            "deploy", ["tests", "approval"], satisfied=["tests", "approval"]
        )
        kernel = OperatorKernel(problem.registry)
        result = kernel.solve(problem.state, problem.goals)
        cases = build_decision_training_cases(kernel, result)
        negatives = hard_negative_records(cases)
        corpus = TraceCorpus(max_synthetic_per_domain=1)
        record = corpus.add_result(
            "trace-1",
            "hidden_premise",
            result,
            source="synthetic",
            hard_negatives=negatives,
        )

        self.assertLessEqual(len(record.hard_negatives), 4 * len(record.actions))
        with self.assertRaises(ValueError):
            corpus.add_result(
                "trace-2", "hidden_premise", result, source="synthetic"
            )

    def test_mdl_macro_requires_support_compression_and_heldout_check(self) -> None:
        traces = {}
        for index, name in enumerate(("deploy", "publish", "release")):
            problem = make_hidden_premise_instance(
                name, ["approval"], satisfied=["approval"]
            )
            traces[str(index)] = OperatorKernel(problem.registry).solve(
                problem.state, problem.goals
            )
        library = MdlMacroLibrary.induce(
            traces, heldout_validator=lambda candidate: bool(candidate.effect_signature)
        )

        self.assertTrue(library.records)
        self.assertTrue(all(record.retained for record in library.records))
        self.assertGreaterEqual(library.records[0].reduction_ratio, 0.10)

    def test_torch_training_loss_is_available_only_in_training_module(self) -> None:
        try:
            from semop.tiny_controller.training import (
                TorchTinyController,
                controller_loss,
                encode_training_example,
            )
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")
        config = TinyControllerConfig(
            d_model=16,
            token_buckets=64,
            relation_buckets=16,
            operator_buckets=16,
        )
        problem = parse_geometry_dsl(
            "point A, B, M\nassume midpoint(M, A, B)\nprove collinear(A, M, B)\n"
        )
        actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
        example = encode_training_example(
            problem.state,
            problem.goals,
            actions,
            target_action=0,
            target_halt=False,
            target_value=1.0,
            config=config,
        )
        model = TorchTinyController(config)
        loss = controller_loss(model(example), example)
        loss.total.backward()
        self.assertGreater(float(loss.total.detach()), 0.0)

    def test_structural_head_receives_goal_binding_supervision(self) -> None:
        try:
            from semop.tiny_controller.training import (
                TorchTinyController,
                controller_loss,
                encode_training_example,
            )
        except ImportError:
            self.skipTest("PyTorch training profile is not installed")
        synthetic = generate_symbolic_curriculum(
            1,
            seed=5,
            curriculum="language-math-vision",
            domains=("language",),
        )[0]
        kernel = OperatorKernel(synthetic.instance.registry)
        result = kernel.solve(synthetic.instance.state, synthetic.instance.goals)
        case = build_decision_training_cases(kernel, result)[-1]
        config = TinyControllerConfig(
            d_model=16,
            token_buckets=64,
            relation_buckets=16,
            operator_buckets=16,
        )
        example = encode_training_example(
            case.state,
            case.goals,
            case.actions,
            target_action=case.target_action,
            target_halt=False,
            target_value=1.0,
            config=config,
        )
        model = TorchTinyController(config)
        controller_loss(model(example), example).total.backward()

        gradient = model.action_structure_weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
