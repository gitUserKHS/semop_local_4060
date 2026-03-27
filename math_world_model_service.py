from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import ProductionMathServiceConfig, WorldModelMathProductionService


class _Handler(BaseHTTPRequestHandler):
    service: WorldModelMathProductionService

    def do_GET(self) -> None:
        if self.path == '/healthz':
            self._send_json(self.service.health())
            return
        if self.path == '/readyz':
            readiness = self.service.readiness().model_dump()
            self._send_json(readiness, status=200 if readiness.get('ready') else 503)
            return
        if self.path == '/metrics':
            self._send_json(self.service.metrics.model_dump())
            return
        self._send_json({'error': 'not_found'}, status=404)

    def do_POST(self) -> None:
        if self.path == '/self_test':
            summary = self.service.self_test().model_dump()
            self._send_json(summary, status=200 if summary.get('ready') else 503)
            return
        if self.path == '/solve':
            payload = self._read_json()
            response = self.service.solve_request(
                payload.get('query', ''),
                source_context=payload.get('source_context', ''),
                visual_input=payload.get('visual_input'),
                task_mode=payload.get('task_mode', 'auto'),
                metadata=payload.get('metadata') if isinstance(payload.get('metadata'), dict) else None,
            )
            status = 200 if response.accepted else 202 if response.status == 'review' else 400
            self._send_json(response.model_dump(), status=status)
            return
        self._send_json({'error': 'not_found'}, status=404)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(length).decode('utf-8', errors='replace') if length else '{}'
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: dict[str, object], *, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description='Production service for world-model math reasoning')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8792)
    parser.add_argument('--concept-store', default='data/vlso_visual_prototypes.db')
    parser.add_argument('--operator-store', default='data/vlso_visual_operators.db')
    parser.add_argument('--affordance-weights', default='data/vlso_samples/trained_affordance_weights.json')
    parser.add_argument('--logical-weights', default='data/math_world_model_gui_run/logical_pattern_weights.json')
    parser.add_argument('--strategy-memory', default='data/math_world_model_gui_run/math_strategy_memory.json')
    parser.add_argument('--audit-log', default='data/math_world_model_service/audit_log.jsonl')
    parser.add_argument('--hardware-profile', default='auto')
    args = parser.parse_args()

    service = WorldModelMathProductionService(
        ProductionMathServiceConfig(
            concept_store_path=args.concept_store,
            operator_store_path=args.operator_store,
            affordance_weights_path=args.affordance_weights,
            logical_weight_path=args.logical_weights,
            strategy_memory_path=args.strategy_memory,
            audit_log_path=args.audit_log,
            hardware_profile=args.hardware_profile,
        )
    )
    handler = type('MathWorldModelServiceHandler', (_Handler,), {'service': service})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'World Model Math Service running at http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
