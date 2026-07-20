from __future__ import annotations

import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from semop.beginner import (
    BeginnerInputError,
    BeginnerReasoner,
    VISION_PRESETS,
)
from semop.beginner_web import create_server, render_home_page
from semop.kernel import PolicyDecision, TypedExperienceStore


class _CountingPolicy:
    name = "counting-test-policy"

    def __init__(self) -> None:
        self.calls = 0

    def score_actions(self, _state, _goals, actions):
        self.calls += 1
        return PolicyDecision(tuple(0.0 for _action in actions))


class BeginnerReasonerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reasoner = BeginnerReasoner()

    def test_language_ready_form_hides_typed_dsl(self) -> None:
        result = self.reasoner.solve_language(
            goal="배포",
            required="테스트 통과, 관리자 승인",
            satisfied="테스트 통과, 관리자 승인",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.conclusion, "ready")
        self.assertIn("필요한 조건이 모두 충족", result.summary)
        self.assertIn("READY(배포)", result.proof)
        self.assertIn("외부 증거로 확인하지 않았어", result.trust_notice)

    def test_language_blocked_form_proves_not_ready(self) -> None:
        result = self.reasoner.solve_language(
            goal="배포",
            required="테스트 통과, 관리자 승인",
            satisfied="테스트 통과",
            blocked="관리자 승인",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.conclusion, "not_ready")
        self.assertIn("관리자 승인", result.summary)
        self.assertIn("NOT_READY(배포)", result.proof)

    def test_language_missing_status_is_honestly_unproven(self) -> None:
        result = self.reasoner.solve_language(
            goal="배포",
            required="테스트 통과, 관리자 승인",
            satisfied="테스트 통과",
        )

        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertEqual(result.conclusion, "not_proven")
        self.assertIn("관리자 승인", result.summary)

    def test_language_rejects_conflicting_or_unrelated_status(self) -> None:
        with self.assertRaisesRegex(BeginnerInputError, "동시에"):
            self.reasoner.solve_language(
                goal="배포",
                required="승인",
                satisfied="승인",
                blocked="승인",
            )
        with self.assertRaisesRegex(BeginnerInputError, "필요 조건 목록에 없어"):
            self.reasoner.solve_language(
                goal="배포",
                required="승인",
                satisfied="테스트",
            )

    def test_language_matches_english_conditions_case_insensitively(self) -> None:
        result = self.reasoner.solve_language(
            goal="Deploy",
            required="Approval",
            satisfied="approval",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.conclusion, "ready")

    def test_math_reports_proven_and_unproven_comparisons(self) -> None:
        proven = self.reasoner.solve_math("(2 + 3) * 4 == 20")
        unproven = self.reasoner.solve_math("2 + 2 == 5")

        self.assertTrue(proven.success)
        self.assertTrue(proven.verified)
        self.assertEqual(proven.conclusion, "proven")
        self.assertFalse(unproven.success)
        self.assertEqual(unproven.conclusion, "not_proven")

    def test_every_beginner_vision_preset_is_replay_verified(self) -> None:
        for key in VISION_PRESETS:
            with self.subTest(preset=key):
                result = self.reasoner.solve_vision(key)
                self.assertTrue(result.success)
                self.assertTrue(result.verified)
                self.assertEqual(result.conclusion, "proven")
                self.assertIn("일반 사진 인식이 아니라", result.trust_notice)

    def test_public_result_is_json_serializable(self) -> None:
        result = self.reasoner.solve_math("7 < 10")
        encoded = json.dumps(result.to_dict(), ensure_ascii=False, default=str)
        self.assertIn("검증", encoded)

    def test_one_controller_guides_all_three_domains_without_owning_facts(self) -> None:
        policy = _CountingPolicy()
        reasoner = BeginnerReasoner(policy=policy)

        language = reasoner.solve_language(
            goal="배포",
            required="테스트, 승인",
            satisfied="테스트, 승인",
        )
        after_language = policy.calls
        math = reasoner.solve_math("(2 + 3) * 4 == 20")
        after_math = policy.calls
        vision = reasoner.solve_vision("red_square")

        self.assertTrue(language.verified and math.verified and vision.verified)
        self.assertGreater(after_language, 0)
        self.assertGreater(after_math, after_language)
        self.assertGreater(policy.calls, after_math)
        self.assertTrue(reasoner.controller_summary["active"])
        self.assertEqual(
            reasoner.controller_summary["kind"],
            "counting-test-policy",
        )

    def test_local_experience_store_collects_only_noteworthy_runs(self) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "beginner.db")
            reasoner = BeginnerReasoner(experience_store=store)
            ready = reasoner.solve_language(
                goal="배포",
                required="테스트",
                satisfied="테스트",
            )
            missing = reasoner.solve_language(
                goal="출시",
                required="검토, 승인",
                satisfied="검토",
            )
            with self.assertRaisesRegex(BeginnerInputError, "수학식을 읽지 못했어"):
                reasoner.solve_math("2 +")
            stats = store.stats()
            items = store.list_items(limit=10)

        self.assertTrue(reasoner.experience_enabled)
        self.assertTrue(ready.success)
        self.assertFalse(missing.success)
        self.assertEqual(stats.requests, 2)
        self.assertEqual(dict(stats.by_domain), {"language": 1, "math": 1})
        self.assertEqual(
            {trigger for item in items for trigger in item.triggers},
            {"runtime_exception", "unsolved"},
        )

    def test_human_can_review_a_pending_item_and_capture_a_normal_success(self) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "beginner.db")
            reasoner = BeginnerReasoner(experience_store=store)
            reasoner.solve_language(
                goal="출시",
                required="검토, 승인",
                satisfied="검토",
            )
            pending = reasoner.experience_snapshot()["items"][0]

            with self.assertRaisesRegex(BeginnerInputError, "직접 확인"):
                reasoner.review_experience(
                    pending["request_digest"],
                    expected_solved=False,
                    attest_human_review=False,
                )
            reviewed = reasoner.review_experience(
                pending["request_digest"],
                expected_solved=False,
                attest_human_review=True,
            )
            captured = reasoner.review_example(
                "math",
                {"expression": "3 * (4 + 1) == 15"},
                expected_solved=True,
                attest_human_review=True,
            )
            snapshot = reasoner.experience_snapshot()

        self.assertEqual(reviewed["review"]["review"]["decision"], "approved")
        self.assertTrue(captured["result"]["verified"])
        self.assertTrue(captured["result"]["experience_digest"])
        self.assertEqual(snapshot["stats"]["requests"], 2)
        self.assertEqual(snapshot["stats"]["approved"], 2)
        self.assertEqual(snapshot["stats"]["by_domain"], {"language": 1, "math": 1})
        self.assertTrue(
            any("3 * (4 + 1)" in item["summary"] for item in snapshot["items"])
        )


class BeginnerWebTests(unittest.TestCase):
    def test_server_rejects_an_empty_port_search(self) -> None:
        with self.assertRaisesRegex(ValueError, "attempts"):
            create_server(8765, attempts=0)

    def test_home_page_contains_three_plain_language_paths(self) -> None:
        page = render_home_page()
        collecting_page = render_home_page(experience_enabled=True)
        guided_page = render_home_page(
            controller_summary={"active": True}
        )

        self.assertIn("SemOp 쉬운 시작", page)
        self.assertIn("언어 조건", page)
        self.assertIn("수학식", page)
        self.assertIn("색상 비전", page)
        self.assertIn("red_square", page)
        self.assertIn("입력을 학습 후보로 저장하지 않아", page)
        self.assertIn("로컬 검토 큐에 저장", collecting_page)
        self.assertIn("외부 전송이나 자동 학습은 하지 않아", collecting_page)
        self.assertIn("로컬 학습 후보 검토", collecting_page)
        self.assertIn("나는 위 입력과 기대 결과를 직접 확인했어", collecting_page)
        self.assertIn("/api/experience/review-example", collecting_page)
        self.assertIn("검증 학습 시도", collecting_page)
        self.assertIn("/api/experience/learn", collecting_page)
        self.assertIn("결정론적 탐색", page)
        self.assertIn("탐색 컨트롤러가 연산자 순서를 안내", guided_page)
        self.assertNotIn("__VISION_PRESETS__", page)
        self.assertNotIn("__EXPERIENCE_NOTICE__", collecting_page)
        self.assertNotIn("__EXPERIENCE_ENABLED__", collecting_page)
        self.assertNotIn("__CONTROLLER_NOTICE__", guided_page)

    def test_health_and_solve_http_endpoints(self) -> None:
        server = create_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base}/health", timeout=5) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["ok"])
            self.assertFalse(health["experience_collection"])
            self.assertEqual(health["active_learned_rules"], 0)
            self.assertFalse(health["controller"]["active"])

            request = Request(
                f"{base}/api/solve",
                data=json.dumps(
                    {
                        "domain": "language",
                        "values": {
                            "goal": "배포",
                            "required": "테스트, 승인",
                            "satisfied": "테스트, 승인",
                            "blocked": "",
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["result"]["verified"])
            self.assertEqual(payload["result"]["conclusion"], "ready")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_local_experience_http_review_requires_human_confirmation(self) -> None:
        with TemporaryDirectory() as directory:
            store = TypedExperienceStore(Path(directory) / "beginner.db")
            service = BeginnerReasoner(experience_store=store)
            service.solve_language(
                goal="출시",
                required="검토, 승인",
                satisfied="검토",
            )
            server = create_server(0, service=service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{base}/api/experience", timeout=5) as response:
                    queue = json.loads(response.read().decode("utf-8"))
                digest = queue["items"][0]["request_digest"]
                self.assertTrue(queue["enabled"])
                self.assertEqual(queue["stats"]["pending"], 1)

                unconfirmed = Request(
                    f"{base}/api/experience/review",
                    data=json.dumps(
                        {
                            "request_digest": digest,
                            "expected_solved": False,
                            "attest_human_review": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as raised:
                    urlopen(unconfirmed, timeout=5)
                self.assertEqual(raised.exception.code, 400)

                unconfirmed_learning = Request(
                    f"{base}/api/experience/learn",
                    data=json.dumps({"confirm_learning": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as raised_learning:
                    urlopen(unconfirmed_learning, timeout=5)
                self.assertEqual(raised_learning.exception.code, 400)

                confirmed = Request(
                    f"{base}/api/experience/review",
                    data=json.dumps(
                        {
                            "request_digest": digest,
                            "expected_solved": False,
                            "attest_human_review": True,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(confirmed, timeout=5) as response:
                    reviewed = json.loads(response.read().decode("utf-8"))

                reviewed_example = Request(
                    f"{base}/api/experience/review-example",
                    data=json.dumps(
                        {
                            "domain": "math",
                            "values": {"expression": "7 < 10"},
                            "expected_solved": True,
                            "attest_human_review": True,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(reviewed_example, timeout=5) as response:
                    captured = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertTrue(reviewed["ok"])
        self.assertEqual(reviewed["queue"]["stats"]["approved"], 1)
        self.assertTrue(captured["result"]["verified"])
        self.assertEqual(captured["queue"]["stats"]["approved"], 2)

    def test_invalid_http_input_returns_beginner_message(self) -> None:
        server = create_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            request = Request(
                f"{base}/api/solve",
                data=json.dumps(
                    {"domain": "math", "values": {"expression": ""}}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)
            payload = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(raised.exception.code, 400)
            self.assertIn("수학식", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
