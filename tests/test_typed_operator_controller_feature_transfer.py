from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from evaluate_controller_feature_transfer import (  # noqa: E402
    evaluate_controller_feature_transfer,
)
from semop.kernel import OperatorKernel  # noqa: E402
from semop.kernel.domains import (  # noqa: E402
    make_hidden_premise_instance,
    parse_geometry_dsl,
)
from semop.tiny_controller import (  # noqa: E402
    ControllerFeatureProfile,
    NumpyTinyController,
    StructuralLinearPolicy,
    TinyControllerConfig,
    canonicalize_problem,
)


def test_typed_structure_profile_removes_names_but_keeps_typed_goal_shape() -> None:
    problem = make_hidden_premise_instance(
        "deploy",
        ["tests", "approval"],
        satisfied=["tests", "approval"],
    )
    actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)

    full = canonicalize_problem(problem.state, problem.goals, actions)
    typed = canonicalize_problem(
        problem.state,
        problem.goals,
        actions,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )

    assert full.feature_profile is ControllerFeatureProfile.FULL
    assert typed.feature_profile is ControllerFeatureProfile.TYPED_STRUCTURE
    assert any(
        feature.startswith("schema:")
        for action in full.actions
        for feature in action.operator_features
    )
    assert not any(
        feature.startswith(("schema:", "tag:"))
        for action in typed.actions
        for feature in action.operator_features
    )
    assert all(
        relation.name.startswith("typed:")
        for relation in typed.relations + typed.goals
    )
    assert any(
        feature.startswith("effect:typed:arity:")
        for action in typed.actions
        for feature in action.operator_features
    )
    assert tuple(action.structural_values for action in typed.actions) == tuple(
        action.structural_values for action in full.actions
    )


def test_numpy_artifact_roundtrip_preserves_typed_structure_contract() -> None:
    config = TinyControllerConfig(
        d_model=8,
        token_buckets=32,
        relation_buckets=8,
        operator_buckets=8,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )
    model = NumpyTinyController.random(config, seed=19)

    restored = NumpyTinyController.from_artifact(model.to_artifact())

    assert restored.config.feature_profile is ControllerFeatureProfile.TYPED_STRUCTURE
    assert restored.config == config


def test_typed_structure_is_invariant_to_names_and_symmetric_argument_order() -> None:
    first = parse_geometry_dsl(
        "point A, B, M\n"
        "assume midpoint(M, A, B)\n"
        "prove collinear(A, M, B)\n"
    )
    second = parse_geometry_dsl(
        "point Z, X, Q\n"
        "assume midpoint(Q, Z, X)\n"
        "prove collinear(Z, Q, X)\n"
    )
    first_actions = OperatorKernel(first.registry).enumerate_actions(first.state)
    second_actions = OperatorKernel(second.registry).enumerate_actions(second.state)
    first_graph = canonicalize_problem(
        first.state,
        first.goals,
        first_actions,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )
    second_graph = canonicalize_problem(
        second.state,
        second.goals,
        second_actions,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )
    config = TinyControllerConfig(
        d_model=8,
        token_buckets=32,
        relation_buckets=8,
        operator_buckets=8,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )
    model = NumpyTinyController.random(config, seed=29)

    assert first_graph.node_tokens == second_graph.node_tokens
    assert first_graph.relations == second_graph.relations
    assert first_graph.goals == second_graph.goals
    assert first_graph.actions == second_graph.actions
    assert model.score_actions(
        first.state,
        first.goals,
        first_actions,
    ).action_scores == model.score_actions(
        second.state,
        second.goals,
        second_actions,
    ).action_scores


def test_typed_structure_uses_edges_instead_of_node_id_tokens_for_terms() -> None:
    problem = parse_geometry_dsl(
        "point A, B, M\n"
        "assume midpoint(M, A, B)\n"
        "prove equal_length(segment(A, M), segment(M, B))\n"
    )
    actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)

    graph = canonicalize_problem(
        problem.state,
        problem.goals,
        actions,
        feature_profile=ControllerFeatureProfile.TYPED_STRUCTURE,
    )

    assert not any(
        token.startswith("anon:") or ":node:" in token
        for tokens in graph.node_tokens
        for token in tokens
    )
    assert any("term:application" in tokens for tokens in graph.node_tokens)
    assert any(relation.role == "function" for relation in graph.relations)


def test_v5_full_artifact_migrates_without_changing_action_scores() -> None:
    import numpy as np

    problem = make_hidden_premise_instance(
        "deploy",
        ["approval"],
        satisfied=["approval"],
    )
    actions = OperatorKernel(problem.registry).enumerate_actions(problem.state)
    config = TinyControllerConfig(
        d_model=8,
        token_buckets=32,
        relation_buckets=8,
        operator_buckets=8,
    )
    model = NumpyTinyController.random(config, seed=23)
    with np.load(BytesIO(model.to_artifact()), allow_pickle=False) as payload:
        legacy = {name: payload[name] for name in payload.files}
    legacy_config = json.loads(str(legacy["config_json"].item()))
    legacy_config.pop("feature_profile")
    legacy["config_json"] = np.asarray(json.dumps(legacy_config, sort_keys=True))
    legacy["format_version"] = np.asarray(5, dtype=np.int32)
    buffer = BytesIO()
    np.savez_compressed(buffer, **legacy)

    restored = NumpyTinyController.from_artifact(buffer.getvalue())

    assert restored.config.feature_profile is ControllerFeatureProfile.FULL
    assert restored.score_actions(
        problem.state,
        problem.goals,
        actions,
    ).action_scores == model.score_actions(
        problem.state,
        problem.goals,
        actions,
    ).action_scores


def test_structural_policy_loads_v1_artifact_as_full_profile() -> None:
    legacy = json.dumps(
        {
            "format_version": 1,
            "kind": "structural-linear-v1",
            "action_limit_score_margin": 0.5,
            "weights": [["dense:effect_goal_exact", 1.0]],
        }
    ).encode("utf-8")

    restored = StructuralLinearPolicy.from_artifact(legacy)

    assert restored.feature_profile is ControllerFeatureProfile.FULL
    assert restored.weights == (("dense:effect_goal_exact", 1.0),)


def test_sparse_lodo_gate_proves_identity_ablation_and_replay_integrity() -> None:
    report = evaluate_controller_feature_transfer(
        training_per_domain=3,
        test_per_domain=4,
        seed=317,
        max_expansions=2_000,
    )

    assert report["gates"]["all_passed"]
    full = report["profiles"]["full"]
    typed = report["profiles"]["typed_structure"]
    assert full["feature_audit"]["identity_token_count"] > 0
    assert typed["feature_audit"]["identity_token_count"] == 0
    assert (
        typed["feature_audit"]["union_feature_count"]
        < full["feature_audit"]["union_feature_count"]
    )
    for held_out, run in typed["leave_one_domain_out"].items():
        assert held_out not in run["trained_domains"]
        assert run["action_top1_accuracy"] == 1.0
        assert run["guided_verified_solve_rate"] == 1.0
        assert run["replay_integrity"] == 1.0
