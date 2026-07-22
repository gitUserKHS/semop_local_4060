# Trust Provenance And Honest Metrics

## 목적

SemOp의 verifier는 typed operator program이 등록된 규칙대로 실행됐는지는
강하게 검증할 수 있다. 하지만 입력 문장이 현실에서 참인지, 픽셀이 실제 사물을
뜻하는지, 등록된 규칙이 현실 법칙을 완전하게 표현하는지까지 자동으로 보장하지는
않는다.

이 문서는 다음 세 신뢰 축을 분리하는 공통 계약을 설명한다.

1. 입력이 어떤 방식으로 assertion이 되었는가
2. 그 assertion을 어떤 evidence가 뒷받침하는가
3. proof engine에서 어떤 logical status를 갖는가

입력에서 `Fact`가 만들어지기 전의 후보, 결정 권한, verifier와 hard negative lineage는
`typed_grounding_boundary.md`의 `GroundingTrace`가 담당한다.

## Fact의 세 축

`Fact`는 이제 서로 독립적인 다음 필드를 가진다.

| 축 | 타입 | 질문 |
| --- | --- | --- |
| assertion | `AssertionStatus` | 이 사실은 명시, 측정, 추론, import 중 어디서 왔는가? |
| evidence | `EvidenceStatus` | 외부 검증, 어댑터 검증, 미검증, 가정 중 무엇인가? |
| logical | `FactStatus` | 증명 전제로 사용할 수 있는 observed/assumed/derived인가? |

예를 들어 `Satisfied: supervisor approval`은 통제 문법으로 정확히 파싱됐더라도
다음과 같이 기록된다.

```text
assertion_status = explicit
evidence_status = unverified
logical_status = observed
```

이는 문장에 해당 주장이 명시되어 있어 조건부 문서 추론에는 사용할 수 있지만,
실제 승인 시스템에서 승인됐다는 외부 증거까지 확인한 것은 아니라는 뜻이다.

반대로 정확한 수학 문자열의 AST 구조와 RGB 배열에서 결정론적으로 계산한 픽셀
면적은 `adapter_verified`다. 이 상태도 자연 이미지의 고수준 의미가 사람 gold와
일치한다는 뜻은 아니다. 어댑터가 주어진 입력에 대해 해당 구조를 결정론적으로
계산했다는 좁은 보장이다.

## Proof dependency

성공한 `SolveResult`는 실제 supporting proof가 사용한 초기 사실만 추적한다.

```python
result.conditional
result.observed_dependencies
result.assumption_dependencies
result.derived_input_dependencies
result.unverified_dependencies
result.dependencies.evidence_complete
```

`conditional=True`이면 논리 전개는 replay 검증을 통과했지만 적어도 한 개의
`ASSUMED` 사실에 의존한다. `unverified_dependencies`가 비어 있지 않으면 결론은
문서 assertion에는 충실해도 외부 의미 증거가 확인되지 않은 상태다.

state digest에는 atom뿐 아니라 logical status, source, confidence, assertion status,
evidence status가 모두 포함된다. 따라서 proof trace의 사실 provenance를 바꾸면
replay digest가 달라진다.

## 평가 지표

이전 이름은 호환 별칭으로만 유지한다.

| 공식 지표 | 의미 | 이전 별칭 |
| --- | --- | --- |
| `replay_verified_goal_completion` | positive 목표 중 replay까지 통과한 비율 | `verified_solve_rate` |
| `primitive_replay_integrity` | 성공으로 보고된 program 중 primitive replay가 통과한 비율 | `proof_soundness` |
| `labeled_outcome_accuracy` | 권한과 무관하게 모든 기대 결과와 성공 여부가 일치한 비율 | `expected_outcome_accuracy` |
| `programmatic_outcome_accuracy` | 합성 또는 프로그램 라벨만 따로 계산한 일치율. 해당 과제가 없으면 `None` | 없음 |
| `curated_unreviewed_accuracy` | 독립 리뷰 전 curated 라벨의 일치율. 의미 gold로 해석하지 않음 | 없음 |
| `semantic_correctness` | 사람 검토 gold label과 결과가 일치한 비율 | 없음 |

`semantic_correctness`는 `SemanticLabelAuthority.HUMAN_REVIEWED` task가 있을 때만
계산한다. 합성 task만 있는 평가에서는 `None`이며 `semantic_gold_tasks=0`이다.
따라서 `primitive_replay_integrity=1.0`을 현실 의미 정확도 100%라고 해석하면 안 된다.

라벨 권한은 `programmatic`, `curated_unreviewed`, `human_reviewed`, `unknown`으로
분리한다. `expected_outcome_accuracy`는 호환성 별칭이며 이제 모든 라벨을 합친
`labeled_outcome_accuracy`를 가리킨다. 권한별 연구 주장을 할 때는 이 별칭을
사용하지 않는다.

`human_reviewed` enum만 직접 지정하는 것으로는 gold task를 만들 수 없다.
`SemanticLabelEvidence`에 case SHA-256, reviewer, timezone이 있는 timestamp와
고정 attestation이 모두 있어야 하며, raw grounding과 trace metadata까지 이
증거를 전달한다. LMV benchmark는 별도 review sidecar의 digest가 현재 case와
정확히 일치할 때만 이 evidence를 생성한다.

`SelfLearningBudget`은 `required_semantic_correctness`,
`min_semantic_gold_tasks`, `min_semantic_gold_tasks_per_domain`,
`required_semantic_domains`를 선택적으로 받는다. 하나라도 활성화한 연구 실행은
후보 정책의 held-out 사람 gold가 부족하거나 정확도 기준에 못 미치면 승격을
거절한다. 필수 도메인은 각각 최소 한 개의 digest-bound review를 가져야 한다.

## 더 어려운 negative control

`generate_semantic_near_miss_controls()`는 먼저 positive proof를 replay한 뒤 실제로
사용된 초기 전제 하나를 제거한다. 목표와 심볼은 그대로 보존하고, 생성된 문제가
정말 풀리지 않는지 verifier로 다시 확인한다.

현재 우선순위는 다음과 같다.

1. missing proof premise
2. 타입 안전한 관계 방향 반전
3. 기존 unreachable-symbol fallback

향후에는 숫자 경계 차이, 명시적 모순, 같은 속성을 가진 다른 객체, 강한 단정 표현과
증거 부재를 별도 사람 작성 holdout으로 추가한다.

## 자가 학습 주장의 범위

현재 loop가 학습하는 것은 이미 등록된 primitive operator로 풀 수 있는 문제에서
어떤 action을 먼저 실행할지에 대한 policy다. 정확한 표현은 다음과 같다.

> verifier-guided search-policy distillation with replay-gated promotion

macro 학습은 검증된 primitive sub-program을 재사용하지만 새 primitive의 의미 법칙을
스스로 발명하지는 않는다. 새 primitive 획득은 별도 discovery, semantic validation,
human gold gate를 통과해야 하는 다음 연구 단계다.

## CI 계약

`.github/workflows/typed-core.yml`은 다음을 자동 검사한다.

- Ubuntu와 Windows
- Python 3.11과 3.12
- `pip install -e . --no-deps` 무의존 core
- 언어, 수학, raster vision, proof replay, provenance 테스트
- Ubuntu Python 3.12 NumPy controller와 전체 typed-operator suite

로컬에서 같은 핵심 검증을 실행하려면 다음을 사용한다.

```powershell
python -m pytest tests/test_typed_operator_trust.py
python -m pytest -k typed_operator
python -m pytest
```

## 새 어댑터 체크리스트

1. 입력 assertion의 생성 방식을 `AssertionStatus`로 지정한다.
2. 검증 범위를 과장하지 않는 `EvidenceStatus`를 지정한다.
3. 미확인 후보는 `PROPOSED`로 두어 proof premise 사용을 막는다.
4. assumption과 observed dependency가 결과에 나타나는지 테스트한다.
5. 사람 gold가 없다면 `semantic_correctness`를 보고하지 않는다.
6. positive와 표현이 가까운 verifier-confirmed negative를 함께 추가한다.
7. 새 adapter의 모든 입력 fact를 `GroundingTrace`에 연결하고 state audit을 통과한다.
