# 언어 텍스트 Typed Adapter

## 목적과 범위

`LanguageTextAdapter`는 한국어·영어 텍스트에서 명시적으로 표현된 목표, 필수 전제,
충족 상태, 차단 상태를 typed fact로 바꾼다. 이 계층의 목적은 작은 언어 모델이 답을
직접 생성하게 하는 것이 아니라, 통제 문법으로 명시된 문장을 operator kernel의
논리 입력으로 전달하고 그 외 후보를 격리하는 것이다.

현재 구현은 범용 자연어 이해기가 아니다. 지원 문법과 일치하는 명시적 문장은
`observed`, 기존 휴리스틱이 추측한 후보는 `proposed`, 서로 충돌하는 상태는
`contradicted`가 된다. `observed`, `assumed`, `derived`만 proof 전제로 사용할 수
있으므로 휴리스틱 추측만으로는 목표가 증명되지 않는다.

여기서 `observed`는 문서 안에 명시된 assertion이라는 논리 상태다. 현실의 승인,
완료 여부를 외부 시스템에서 확인했다는 뜻은 아니다. 명시 문장은
`assertion_status=explicit`, `evidence_status=unverified`로 남으며, 성공 결과의
`unverified_dependencies`에서 확인할 수 있다. 자세한 계약은
`trust_provenance_and_metrics.md`를 참고한다.

## 입력 형태

라벨 형식은 가장 결정적이고 권장되는 입력이다.

```text
Goal: deploy
Requires: tests, approval
Satisfied: tests
Satisfied: approval
```

한국어 라벨 `목표`, `필요`, `전제`, `충족`, `완료`, `차단`, `누락`도 사용할 수
있다. 줄바꿈이나 세미콜론으로 문장을 구분한다.

현재 고정밀 문장 패턴의 예는 다음과 같다.

```text
Deploy requires tests and approval.
Security review is not satisfied.
Can we deploy?

배포하려면 테스트 통과와 승인이 필요하다.
테스트 통과가 충족되었다.
승인이 충족되었다.
```

문장 표면형은 소문자화하고 공백을 `_`로 정규화한다. 한국어 `와/과`는 뒤에 공백이
있는 목록 접속 조사일 때만 분리하므로 `통과` 같은 단어 내부를 자르지 않는다.

## Typed 변환

```text
text
  -> LanguageTextParser
  -> LanguageClaim(GOAL | REQUIRES | SATISFIED | BLOCKED)
  -> AssertionStatus + EvidenceStatus + FactStatus trust boundary
  -> hidden-premise operators
  -> OperatorKernel.solve
  -> proof replay
```

준비 가능 목표에는 각 `REQUIRES + SATISFIED` 쌍을 확인하는 operator와 모든 유한
전제를 결합하는 ground operator가 등록된다. 하나의 필수 전제가 명시적으로
`BLOCKED`이면 `BLOCKED_BY`를 거쳐 `NOT_READY`를 증명한다. 상태가 빠져 있으면
`READY`를 추측하지 않고 미증명으로 끝난다.

동일한 전제가 `SATISFIED`와 `BLOCKED`로 동시에 주어지면 두 fact 모두
`contradicted`가 되어 어느 proof에도 사용되지 않는다.

## 공개 API

```python
from semop.kernel import TypedDomainRequest, UnifiedTypedReasoner

runtime = UnifiedTypedReasoner()
result = runtime.run(
    TypedDomainRequest(
        "language",
        "Goal: deploy; Requires: tests; Satisfied: tests",
        "shadow",
    )
)

assert result.success
assert result.verified  # operator program replay가 검증됨
assert result.unverified_dependencies  # 현실 증거는 별도 확인 대상
print(result.proof_ko)
```

fallback을 끄고 결정적 문법만 평가하려면 `LanguageTextProblem`을 사용한다.

```python
from semop.kernel import LanguageTextAdapter, LanguageTextProblem

instance = LanguageTextAdapter().adapt(
    LanguageTextProblem(
        "Can we deploy? Deploy requires tests.",
        use_legacy_heuristics=False,
    )
)
```

문자열과 `LanguageTextProblem`은 불변 입력이므로 `typed` 모드에서도 원본에 결과를
투영하지 않는다. 기존 mutable `StructuredMeaningGraph` 입력은 같은
`LanguageInputAdapter`가 이전 projection 계약을 유지한다.

## 개념 포함과 개체 분류

`Every programmer is a person`, `모든 개발자는 사람이다`, `Ada is a programmer`처럼
분류 논리를 명시한 입력은 `LanguageLogicAdapter`가 처리한다. 보편 명제는
`SUBCLASS_OF`, 개체 분류는 `INSTANCE_OF`가 되며, operator kernel이 개념 포함의
추이성과 개체 분류 상속을 조합한다.

명시적 부정은 `NOT_INSTANCE_OF`로 보존한다. 같은 분류의 긍정과 부정이 충돌하면 두
fact 모두 `contradicted`가 되어 proof 전제에서 제외되고, 부정에서 contraposition을
추측하지 않는다. 지원 문법과 예제는 `docs/typed_compositional_extensions.md`에 있다.

## 휴리스틱 경계

명시적 목표나 전제를 찾지 못하고 `use_legacy_heuristics=True`이면 기존
`StructuredMeaningPipeline`을 지연 호출한다. 여기서 얻은 모든 claim은 confidence와
무관하게 `verified=False`, 즉 `proposed`다. 후보는 review와 후속 parser 개선에는
쓸 수 있지만 proof goal이나 증거를 만들 수 없다.

이 fallback은 기존 파이프라인 비용을 포함하므로 benchmark와 재현 가능한 테스트에서는
`use_legacy_heuristics=False`를 권장한다.

## 메타데이터와 디버깅

`DomainInstance.metadata`에는 다음 항목이 남는다.

- `claims`: relation, 정규화된 인자, 검증 여부, confidence, source, 원문
- `explicit_claim_count`, `proposed_claim_count`
- `contradictions`, `unparsed_statements`
- `heuristic_used`, `parser_warnings`

따라서 어떤 문장이 proof 증거가 되었고 어떤 문장이 보류되었는지 모델 출력과 별개로
감사할 수 있다.

## 확장 규칙

1. 새 표면형은 의미가 단일한 경우에만 deterministic parser에 추가한다.
2. 부정형을 긍정형보다 먼저 검사해 `not satisfied` 오분류를 막는다.
3. 정규화 규칙은 일반 단어를 훼손하지 않는 예제로 회귀 테스트한다.
4. 통계 모델 parser는 `proposed`만 출력하고 별도 verifier나 사람 review가 승격한다.
5. 새 relation은 typed predicate와 operator replay 테스트를 함께 추가한다.

현재 회귀 테스트는 라벨, 한국어·영어 문장, 부정, 미충족 전제, 모순, 애매한 문장,
휴리스틱 격리, 기존 graph projection을 포함한다.
