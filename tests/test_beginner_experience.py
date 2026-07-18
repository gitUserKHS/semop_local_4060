from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from semop.beginner import (
    BeginnerInputError,
    BeginnerReasoner,
    VISION_PRESETS,
)
from semop.beginner_web import create_server, render_home_page


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


class BeginnerWebTests(unittest.TestCase):
    def test_server_rejects_an_empty_port_search(self) -> None:
        with self.assertRaisesRegex(ValueError, "attempts"):
            create_server(8765, attempts=0)

    def test_home_page_contains_three_plain_language_paths(self) -> None:
        page = render_home_page()

        self.assertIn("SemOp 쉬운 시작", page)
        self.assertIn("언어 조건", page)
        self.assertIn("수학식", page)
        self.assertIn("색상 비전", page)
        self.assertIn("red_square", page)
        self.assertNotIn("__VISION_PRESETS__", page)

    def test_health_and_solve_http_endpoints(self) -> None:
        server = create_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base}/health", timeout=5) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["ok"])

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
