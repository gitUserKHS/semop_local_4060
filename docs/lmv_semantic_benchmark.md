# Language, Math, Vision Semantic Benchmark

## 목적

이 벤치마크는 typed kernel의 다음 세 경계를 따로 측정한다.

1. 입력이 언어, 수학, 비전 어댑터에서 typed instance로 변환되는가
2. 기대한 성공 또는 실패와 verifier 실행 결과가 일치하는가
3. 그 기대 결과를 독립적으로 사람이 검토했는가

기본 seed corpus는 20개 case다.

- 언어 6개: 결합 전제, 누락 전제, 차단, 모순, 문장 순서, entity binding
- 수학 6개: 일차방정식, 연산 우선순위, 유리수, 경계값, 수치 near-miss
- 비전 8개: 정사각형, 픽셀 누락, 방향 관계, count, area 비교

seed case는 구현자가 현재 어댑터 범위에 맞춰 작성한 회귀 자료다. 따라서
모두 통과해도 의미 이해의 독립 증거가 아니다. 리뷰 전 권한은 반드시
`curated_unreviewed`이며 `semantic_correctness`는 `null`이다.

## 파일

- `data/semantic_benchmark/v1/cases.jsonl`: 불변 raw payload와 기대 결과
- `data/semantic_benchmark/v1/reviews.jsonl`: 독립 리뷰 sidecar
- `src/semop/kernel/semantic_benchmark.py`: schema, codec, audit, evaluator
- `tools/eval/evaluate_semantic_benchmark.py`: 실행 CLI
- `tools/eval/review_semantic_benchmark.py`: 사람 리뷰 기록 CLI

case에는 언어 문자열, 수학 표현식 또는 작은 RGB raster와 명시적 goal이
들어간다. review는 case 전체의 canonical SHA-256 digest에 결속된다. payload,
기대 결과, rationale 등이 하나라도 바뀌면 기존 review는 `stale`이 되고 더는
`HUMAN_REVIEWED`로 인정되지 않는다.

## 평가 실행

```powershell
python tools/eval/evaluate_semantic_benchmark.py
python tools/eval/evaluate_semantic_benchmark.py --format json
```

초기 결과에서 다음 두 줄은 함께 읽어야 한다.

```text
labeled outcome accuracy: 1.000
semantic correctness: n/a
```

첫 줄은 curated 기대 결과와 구현의 일치 여부다. 두 번째 줄은 아직 독립
사람 gold가 없음을 뜻한다.

CI나 논문 수치에서 모든 case의 사람 검토를 요구하려면 fail-closed gate를
사용한다.

```powershell
python tools/eval/evaluate_semantic_benchmark.py --require-all-reviewed
```

부분 review를 사용하는 0/5/20/100-shot 실험이나 정책 승격에서는 core와 같은
semantic promotion gate를 CLI에서 실행한다.

```powershell
python tools/eval/evaluate_semantic_benchmark.py `
  --gate-semantic-correctness 1.0 `
  --gate-min-gold 3 `
  --gate-min-gold-per-domain 1 `
  --gate-domains language,math,vision
```

현재 review가 0개이므로 이 명령은 각 도메인의 gold 부족과 의미 정확도 부재를
모두 출력하고 종료 코드 `2`로 실패한다. `--gate-domains`에 들어간 도메인은
`--gate-min-gold-per-domain`이 0이어도 최소 한 개의 유효한 review를 요구한다.

자가 학습 API에서는 같은 규칙을 `SelfLearningBudget`에 넣는다.

```python
SelfLearningBudget(
    required_semantic_correctness=1.0,
    min_semantic_gold_tasks=20,
    min_semantic_gold_tasks_per_domain=5,
    required_semantic_domains=("language", "math", "vision"),
)
```

후보 policy의 held-out metric이 이 조건을 만족하지 못하면
`SelfLearningLoop`는 checkpoint 승격 전에 거절한다. 기본값은 기존 합성 실험의
호환성을 위해 비활성이며, semantic transfer를 주장하는 실행은 명시적으로 켜야
한다.

## 사람 리뷰

대기 목록과 각 digest를 확인한다.

```powershell
python tools/eval/review_semantic_benchmark.py --list-pending
```

리뷰어는 raw payload와 `expected_solved`를 직접 확인한다. 라벨이 틀렸으면
case를 먼저 수정하고 다시 평가한다. 맞다고 판단한 뒤에만 다음 명령으로
승인한다.

```powershell
python tools/eval/review_semantic_benchmark.py `
  --case-id language-ready-two-requirements `
  --reviewer "reviewer-name" `
  --decision approved `
  --notes "입력과 기대 결과를 독립 검토함" `
  --attest-human-review
```

`--attest-human-review`가 없으면 도구는 쓰기를 거부한다. 자동 agent나 학습
스크립트는 이 플래그를 대신 실행해서는 안 된다. 프로젝트가 실제 신원을
암호학적으로 증명하는 것은 아니므로 reviewer identity와 이해상충 관리는
연구 운영 절차에서 별도로 보존해야 한다.

## JSONL case 형식

공통 필드는 다음과 같다.

```json
{
  "schema_version": 1,
  "case_id": "math-off-by-one-equality",
  "domain": "math",
  "payload": {"expression": "5 * (2 + 1) == 14"},
  "expected_solved": false,
  "phenomenon": "numeric_near_miss",
  "rationale": "The left side is 15.",
  "split": "heldout",
  "difficulty": 2,
  "author": "semop-seed-v1",
  "tags": ["negative", "near_miss"]
}
```

비전 payload는 색상 이름, `#RRGGBB`, 또는 `[r, g, b]` 픽셀을 지원하며 goal
종류는 `relation`, `property`, `count`, `area`다. 모든 비전 seed는 작은
결정론적 raster다. 자연 이미지 의미 인식 성능을 나타내지 않는다.

## 지표 해석

- `labeled_outcome_accuracy`: 선택된 모든 라벨과 실행 결과의 일치율
- `curated_unreviewed_accuracy`: 독립 승인 전 회귀 라벨의 일치율
- `semantic_correctness`: digest-bound 승인 case만의 일치율
- `semantic_review_coverage`: 선택된 task 중 digest-bound 승인 case의 비율
- `primitive_replay_integrity`: 성공 proof를 primitive operator로 재생한 무결성
- `review_coverage`: 전체 case 중 유효한 승인 review가 있는 비율

자가 학습 promotion에서는 최종 untouched test에 대해 review coverage와
semantic correctness를 함께 기록해야 한다. replay integrity만으로 parsing,
grounding 또는 현실 증거의 정확성을 주장하면 안 된다.
