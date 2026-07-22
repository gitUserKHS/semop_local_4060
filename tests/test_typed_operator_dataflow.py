from __future__ import annotations

from fractions import Fraction
import unittest

from semop.kernel import (
    Fact,
    FactStatus,
    Goal,
    KernelRegistry,
    NumericConditionSpec,
    NumericDataflowRuleSpec,
    NumericMeasurementRef,
    OperatorKernel,
    TypeValidationError,
    TypedDataflowCompiler,
    WorldState,
)


def _dataflow_registry(*, measurement_verified: bool = True):
    registry = KernelRegistry()
    entity = registry.types.register("Entity")
    number = registry.types.register("Number", entity)
    sensor_type = registry.types.register("Sensor", entity)
    concept = registry.types.register("Concept", entity)
    measurement = registry.register_predicate(
        "MEASURED_VALUE",
        (sensor_type, number),
        verified=measurement_verified,
    )
    classified = registry.register_predicate(
        "INSTANCE_OF",
        (entity, concept),
    )
    blocked = registry.register_predicate(
        "NOT_INSTANCE_OF",
        (entity, concept),
    )
    sensor = registry.symbol("sensor_a", sensor_type)
    subject = registry.symbol("sample", entity)
    property_term = registry.symbol("in_band", concept)
    return registry, measurement, classified, blocked, sensor, subject, property_term


def _compile_band(*, lower: int = 2, upper: int = 4, blocked: bool = False):
    (
        registry,
        measurement,
        classified,
        negative,
        sensor,
        subject,
        property_term,
    ) = _dataflow_registry()
    conclusion = registry.atom(classified, subject, property_term)
    blocker = registry.atom(negative, subject, property_term)
    reference = NumericMeasurementRef(measurement, (sensor,), value_position=1)
    compiled = TypedDataflowCompiler(registry).compile(
        NumericDataflowRuleSpec(
            name="band",
            conditions=(
                NumericConditionSpec(
                    "lower",
                    reference,
                    ">=",
                    Fraction(lower),
                ),
                NumericConditionSpec(
                    "upper",
                    reference,
                    "<=",
                    Fraction(upper),
                ),
            ),
            conclusion=conclusion,
            blocked_by=(blocker,) if blocked else (),
        )
    )
    return registry, compiled, measurement, sensor, conclusion, blocker


class TypedNumericDataflowTests(unittest.TestCase):
    def test_generic_measurement_conditions_compile_to_a_verified_program(self) -> None:
        registry, compiled, measurement, sensor, conclusion, _blocker = _compile_band()
        measured = registry.symbol("3", "Number")
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(measurement, sensor, measured),
                    FactStatus.OBSERVED,
                    "deterministic_sensor",
                ),
            )
        )

        result = OperatorKernel(registry).solve(state, (Goal(conclusion),))

        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            tuple(step.action.operator.name for step in result.proof),
            ("verify_band_lower", "verify_band_upper", "conclude_band"),
        )
        self.assertEqual(len(compiled.rule_facts), 2)
        self.assertEqual(
            tuple(operator.family for operator in compiled.comparison_operators),
            ("compare", "compare"),
        )
        self.assertEqual(compiled.conclusion_operator.family, "compose")

    def test_one_false_condition_prevents_the_conclusion(self) -> None:
        registry, compiled, measurement, sensor, conclusion, _blocker = _compile_band(
            lower=4,
            upper=6,
        )
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(
                        measurement,
                        sensor,
                        registry.symbol("3", "Number"),
                    ),
                    FactStatus.OBSERVED,
                    "deterministic_sensor",
                ),
            )
        )

        result = OperatorKernel(registry).solve(state, (Goal(conclusion),))

        self.assertFalse(result.success)
        self.assertFalse(result.final_state.contains(conclusion))

    def test_explicit_blocker_is_conservative_even_when_not_proof_eligible(self) -> None:
        registry, compiled, measurement, sensor, conclusion, blocker = _compile_band(
            blocked=True
        )
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(
                        measurement,
                        sensor,
                        registry.symbol("3", "Number"),
                    ),
                    FactStatus.OBSERVED,
                    "deterministic_sensor",
                ),
                Fact(blocker, FactStatus.CONTRADICTED, "explicit_negative"),
            )
        )

        result = OperatorKernel(registry).solve(state, (Goal(conclusion),))

        self.assertFalse(result.success)
        self.assertFalse(result.final_state.contains(conclusion))

    def test_replay_rechecks_the_numeric_guard(self) -> None:
        registry, compiled, measurement, sensor, conclusion, _blocker = _compile_band()
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(
                        measurement,
                        sensor,
                        registry.symbol("3", "Number"),
                    ),
                    FactStatus.OBSERVED,
                    "deterministic_sensor",
                ),
            )
        )
        kernel = OperatorKernel(registry)
        result = kernel.solve(state, (Goal(conclusion),))
        registry.guards["guard_band_lower"] = lambda _binding, _state: False

        replay = kernel.replay(state, (Goal(conclusion),), result.proof)

        self.assertTrue(result.success and result.verified)
        self.assertFalse(replay.verified)
        self.assertIn("guard rejected", replay.diagnostics[0])

    def test_value_position_can_precede_fixed_arguments(self) -> None:
        registry, _measurement, classified, _negative, sensor, subject, concept = (
            _dataflow_registry()
        )
        reversed_measurement = registry.register_predicate(
            "REVERSED_MEASUREMENT",
            ("Number", "Sensor"),
        )
        reference = NumericMeasurementRef(
            reversed_measurement,
            (sensor,),
            value_position=0,
        )
        conclusion = registry.atom(classified, subject, concept)
        compiled = TypedDataflowCompiler(registry).compile(
            NumericDataflowRuleSpec(
                name="reversed",
                conditions=(
                    NumericConditionSpec(
                        "positive",
                        reference,
                        ">",
                        Fraction(0),
                    ),
                ),
                conclusion=conclusion,
            )
        )
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(
                        reversed_measurement,
                        registry.symbol("7/2", "Number"),
                        sensor,
                    )
                ),
            )
        )

        result = OperatorKernel(registry).solve(state, (Goal(conclusion),))

        self.assertTrue(result.success and result.verified)

    def test_numeric_subtype_is_preserved_in_the_measurement_variable(self) -> None:
        registry, _measurement, classified, _negative, sensor, subject, concept = (
            _dataflow_registry()
        )
        probability = registry.types.register("Probability", "Number")
        confidence = registry.register_predicate(
            "CONFIDENCE",
            ("Sensor", probability),
        )
        conclusion = registry.atom(classified, subject, concept)
        compiled = TypedDataflowCompiler(registry).compile(
            NumericDataflowRuleSpec(
                name="confidence",
                conditions=(
                    NumericConditionSpec(
                        "accepted",
                        NumericMeasurementRef(
                            confidence,
                            (sensor,),
                            value_position=1,
                        ),
                        ">=",
                        Fraction(1, 2),
                    ),
                ),
                conclusion=conclusion,
            )
        )
        state = WorldState(
            compiled.rule_facts
            + (
                Fact(
                    registry.atom(
                        confidence,
                        sensor,
                        registry.symbol("3/4", probability),
                    )
                ),
            )
        )

        result = OperatorKernel(registry).solve(state, (Goal(conclusion),))

        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            compiled.comparison_operators[0].parameters[0].type,
            probability,
        )

    def test_non_numeric_and_unverified_measurements_are_rejected(self) -> None:
        registry, _measurement, classified, _negative, sensor, subject, concept = (
            _dataflow_registry()
        )
        with self.assertRaisesRegex(TypeError, "must be Terms"):
            NumericMeasurementRef(
                registry.predicates["MEASURED_VALUE"],
                ("not_a_term",),  # type: ignore[arg-type]
                value_position=1,
            )
        label = registry.register_predicate("LABEL", ("Sensor", "Concept"))
        bad_reference = NumericMeasurementRef(
            label,
            (sensor,),
            value_position=1,
        )
        bad_spec = NumericDataflowRuleSpec(
            name="bad_type",
            conditions=(
                NumericConditionSpec(
                    "condition",
                    bad_reference,
                    "==",
                    Fraction(1),
                ),
            ),
            conclusion=registry.atom(classified, subject, concept),
        )

        with self.assertRaisesRegex(TypeValidationError, "not Number"):
            TypedDataflowCompiler(registry).compile(bad_spec)

        (
            unverified_registry,
            unverified_measurement,
            unverified_classified,
            _unverified_negative,
            unverified_sensor,
            unverified_subject,
            unverified_concept,
        ) = _dataflow_registry(measurement_verified=False)
        unverified_spec = NumericDataflowRuleSpec(
            name="unverified",
            conditions=(
                NumericConditionSpec(
                    "condition",
                    NumericMeasurementRef(
                        unverified_measurement,
                        (unverified_sensor,),
                        value_position=1,
                    ),
                    ">",
                    Fraction(0),
                ),
            ),
            conclusion=unverified_registry.atom(
                unverified_classified,
                unverified_subject,
                unverified_concept,
            ),
        )

        with self.assertRaisesRegex(TypeValidationError, "must be verified"):
            TypedDataflowCompiler(unverified_registry).compile(unverified_spec)


if __name__ == "__main__":
    unittest.main()
