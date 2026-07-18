from __future__ import annotations

from dataclasses import replace
import unittest

from semop.kernel import (
    DomainCatalog,
    DomainKind,
    FunctionSemanticCodec,
    LanguageTextProblem,
    TypedDomainRequest,
    UnifiedTypedReasoner,
    create_default_domain_catalog,
    decode_semantic_request,
    encode_semantic_request,
    evaluate_lmv_core_gate,
)


class DomainCatalogTests(unittest.TestCase):
    def test_default_catalog_describes_one_shared_lmv_contract(self) -> None:
        catalog = create_default_domain_catalog()

        self.assertEqual(
            catalog.kinds,
            (
                DomainKind.COMPOSED,
                DomainKind.LANGUAGE,
                DomainKind.MATH,
                DomainKind.VISION,
            ),
        )
        self.assertEqual(
            set(catalog.semantic_kinds),
            {DomainKind.LANGUAGE, DomainKind.MATH, DomainKind.VISION},
        )
        for domain in catalog.semantic_kinds:
            with self.subTest(domain=domain.value):
                spec = catalog.require(domain)
                self.assertTrue(spec.supports("typed_grounding"))
                self.assertTrue(spec.supports("proof_replay"))
                self.assertTrue(spec.supports("semantic_codec"))
                self.assertTrue(spec.input_contract)
                self.assertTrue(spec.description_ko)

    def test_catalog_rejects_duplicates_and_exposes_read_only_adapters(self) -> None:
        catalog = create_default_domain_catalog()
        language = catalog.require(DomainKind.LANGUAGE)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            DomainCatalog((language, language))
        with self.assertRaises(TypeError):
            catalog.adapters[DomainKind.LANGUAGE] = language.adapter  # type: ignore[index]

    def test_adapter_override_returns_a_new_catalog(self) -> None:
        catalog = create_default_domain_catalog()
        original = catalog.require(DomainKind.LANGUAGE).adapter
        replacement = _DelegatingLanguageAdapter(original)

        changed = catalog.with_adapter(DomainKind.LANGUAGE, replacement)

        self.assertIs(catalog.require(DomainKind.LANGUAGE).adapter, original)
        self.assertIs(changed.require(DomainKind.LANGUAGE).adapter, replacement)

    def test_semantic_codec_dispatches_through_catalog_not_domain_branch(self) -> None:
        catalog = create_default_domain_catalog()
        language = catalog.require(DomainKind.LANGUAGE)
        custom_codec = FunctionSemanticCodec(
            "test-language-wrapper",
            lambda value: {
                "wrapped": (
                    value.text if isinstance(value, LanguageTextProblem) else str(value)
                )
            },
            lambda payload: LanguageTextProblem(
                str(payload["wrapped"]),
                use_legacy_heuristics=False,
            ),
        )
        custom = catalog.with_spec(
            replace(language, semantic_codec=custom_codec),
            replace_existing=True,
        )
        request = TypedDomainRequest(
            DomainKind.LANGUAGE,
            LanguageTextProblem(
                "Goal: deploy\nRequires: tests\nSatisfied: tests",
                use_legacy_heuristics=False,
            ),
        )

        encoded = encode_semantic_request(request, catalog=custom)
        decoded = decode_semantic_request(
            DomainKind.LANGUAGE,
            encoded["payload"],
            catalog=custom,
        )

        self.assertEqual(encoded["payload"].keys(), {"wrapped"})
        self.assertIsInstance(decoded.payload, LanguageTextProblem)
        self.assertEqual(decoded.payload.text, request.payload.text)

    def test_runtime_rejects_cross_domain_instance_mismatch(self) -> None:
        reasoner = UnifiedTypedReasoner()
        math_instance = reasoner.ground(
            TypedDomainRequest(DomainKind.MATH, "2 + 2 == 4")
        )

        with self.assertRaisesRegex(ValueError, "produced instance domain"):
            reasoner.run(TypedDomainRequest(DomainKind.LANGUAGE, math_instance))

    def test_runtime_result_carries_catalog_contract(self) -> None:
        reasoner = UnifiedTypedReasoner()
        result = reasoner.run(TypedDomainRequest(DomainKind.MATH, "2 + 2 == 4"))
        spec = reasoner.catalog.require(DomainKind.MATH)

        self.assertTrue(result.success)
        self.assertEqual(result.domain_capabilities, tuple(sorted(spec.capabilities)))
        self.assertEqual(result.input_contract, spec.input_contract)
        self.assertEqual(
            result.to_dict()["domain_capabilities"],
            sorted(spec.capabilities),
        )


class LMVCoreGateTests(unittest.TestCase):
    def test_all_three_domains_pass_positive_negative_codec_and_replay_gate(self) -> None:
        report = evaluate_lmv_core_gate()

        self.assertTrue(report.passed)
        self.assertEqual(
            tuple(result.domain for result in report.results),
            (DomainKind.LANGUAGE, DomainKind.MATH, DomainKind.VISION),
        )
        for result in report.results:
            with self.subTest(domain=result.domain.value):
                self.assertTrue(result.positive_verified)
                self.assertTrue(result.negative_fail_closed)
                self.assertTrue(result.proof_replay_verified)
                self.assertTrue(result.semantic_roundtrip_stable)
                self.assertTrue(result.semantic_digest_stable)
                self.assertTrue(result.catalog_contract_valid)
                self.assertEqual(result.diagnostics, ())

    def test_gate_report_keeps_its_narrow_claim_scope(self) -> None:
        payload = evaluate_lmv_core_gate().to_dict()

        self.assertIn("not open-domain", payload["claim_scope"])
        self.assertEqual(payload["gate"], "lmv_core_contract")


class _DelegatingLanguageAdapter:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def adapt(self, value):
        return self.delegate.adapt(value)

    def project(self, value, result):
        return self.delegate.project(value, result)


if __name__ == "__main__":
    unittest.main()
