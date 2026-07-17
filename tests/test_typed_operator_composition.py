from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel import (
    CompositionComponent,
    DomainInstance,
    Fact,
    Goal,
    KernelRegistry,
    OperatorKernel,
    RegistryCompositionError,
    Rule,
    WorldState,
    compose_domain_instances,
    parse_arithmetic_expression,
)


class DomainCompositionTests(unittest.TestCase):
    def test_two_arithmetic_programs_share_types_but_namespace_expression_nodes(self) -> None:
        left = parse_arithmetic_expression("2 + 3")
        right = parse_arithmetic_expression("4 * 5")
        composition = compose_domain_instances(
            (
                CompositionComponent(
                    left,
                    alias="left",
                    namespace_symbol_types=frozenset({"Expression"}),
                ),
                CompositionComponent(
                    right,
                    alias="right",
                    namespace_symbol_types=frozenset({"Expression"}),
                ),
            )
        )

        result = OperatorKernel(composition.instance.registry).solve(
            composition.instance.state,
            composition.instance.goals,
        )
        left_root = composition.import_for("left").map_term(
            left.goals[0].atom.arguments[0]
        )
        right_root = composition.import_for("right").map_term(
            right.goals[0].atom.arguments[0]
        )

        self.assertTrue(result.success and result.verified)
        self.assertNotEqual(left_root, right_root)
        self.assertTrue(str(left_root).startswith("left::"))
        self.assertTrue(str(right_root).startswith("right::"))
        self.assertIn(
            "right__evaluate_literal_000",
            composition.instance.registry.operators,
        )

    def test_cross_component_rule_is_executed_and_replayed(self) -> None:
        first_registry = KernelRegistry()
        entity = first_registry.types.register("Entity")
        first_registry.register_predicate("FIRST", (entity,))
        first_symbol = first_registry.symbol("shared", entity)
        first = DomainInstance(
            first_registry,
            WorldState((Fact(first_registry.atom("FIRST", first_symbol)),)),
            (),
            "first",
        )

        second_registry = KernelRegistry()
        entity = second_registry.types.register("Entity")
        second_registry.register_predicate("SECOND", (entity,))
        second_symbol = second_registry.symbol("shared", entity)
        second = DomainInstance(
            second_registry,
            WorldState((Fact(second_registry.atom("SECOND", second_symbol)),)),
            (),
            "second",
        )

        composition = compose_domain_instances((first, second))
        registry = composition.instance.registry
        registry.register_predicate("COMBINED", ("Entity",))
        item = registry.variable("item", "Entity")
        registry.register_operator(
            Rule(
                name="combine_component_evidence",
                parameters=(item,),
                preconditions=(
                    registry.atom("FIRST", item),
                    registry.atom("SECOND", item),
                ),
                effects=(registry.atom("COMBINED", item),),
            ),
            family="compose",
            tags=("cross_domain",),
        )
        goal = Goal(registry.atom("COMBINED", registry.symbol("shared", "Entity")))
        instance = replace(composition.instance, goals=(goal,))

        result = OperatorKernel(registry).solve(instance.state, instance.goals)

        self.assertTrue(result.success and result.verified)
        self.assertEqual(
            [step.action.operator.name for step in result.proof],
            ["combine_component_evidence"],
        )

    def test_incompatible_type_parent_is_rejected(self) -> None:
        first_registry = KernelRegistry()
        entity = first_registry.types.register("Entity")
        first_registry.types.register("Thing", entity)
        first = DomainInstance(first_registry, WorldState(), (), "first")

        second_registry = KernelRegistry()
        second_registry.types.register("Entity")
        second_registry.types.register("Thing")
        second = DomainInstance(second_registry, WorldState(), (), "second")

        with self.assertRaisesRegex(RegistryCompositionError, "incompatible parents"):
            compose_domain_instances((first, second))


if __name__ == "__main__":
    unittest.main()
