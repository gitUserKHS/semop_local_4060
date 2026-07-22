# Typed Numeric Dataflow

## 목적

`semop.kernel.dataflow`는 한 도메인이 검증한 숫자 측정값을 다른 도메인의
결론으로 안전하게 연결한다. 핵심은 결론을 직접 추가하는 것이 아니라 다음 operator
program을 선언적으로 만드는 것이다.

```text
verified measurement
  -> guarded numeric comparison
  -> verified condition fact
  -> typed conclusion operator
  -> proof replay
```

현재 대표 경로는 다음과 같다.

```text
RGB pixels
  -> OBJECT_COUNT(red, 3)                 [vision]
  -> 3 >= 2                              [math guard]
  -> COUNT_CONDITION_MET(condition_001)   [typed bridge]
  -> INSTANCE_OF(scene, occupied)         [language]
```

`SceneThresholdAdapter`는 문장과 이미지를 준비하지만 비교 operator를 직접 구현하지
않는다. 공통 `TypedDataflowCompiler`가 같은 계약을 맡기 때문에 앞으로 pixel area,
거리, 확률처럼 `Number`로 타입화된 측정에도 재사용할 수 있다.

## 공개 타입

| 타입 | 역할 |
|---|---|
| `NumericMeasurementRef` | verified predicate에서 숫자 슬롯과 나머지 ground 인자를 지정한다. |
| `NumericConditionSpec` | comparator와 exact `Fraction` threshold를 선언한다. |
| `NumericDataflowRuleSpec` | 여러 조건, 단일 결론, 선택적 blocker를 하나의 프로그램으로 묶는다. |
| `TypedDataflowCompiler` | typed guard, comparison operator, conclusion operator를 registry에 등록한다. |
| `CompiledDataflowRule` | 초기 rule fact와 생성된 condition/operator를 돌려준다. |

숫자 슬롯은 정확히 `Number`일 필요는 없다. `Probability <: Number`처럼 `Number`의
subtype도 허용하며, 생성되는 operator 변수는 측정 predicate가 요구한 정확한 subtype을
보존한다.

## Python 예제

```python
from fractions import Fraction

from semop.kernel import (
    Fact,
    Goal,
    KernelRegistry,
    NumericConditionSpec,
    NumericDataflowRuleSpec,
    NumericMeasurementRef,
    OperatorKernel,
    TypedDataflowCompiler,
    WorldState,
)

registry = KernelRegistry()
entity = registry.types.register("Entity")
number = registry.types.register("Number", entity)
sensor_type = registry.types.register("Sensor", entity)
concept = registry.types.register("Concept", entity)

measured = registry.register_predicate(
    "MEASURED_VALUE", (sensor_type, number)
)
instance_of = registry.register_predicate(
    "INSTANCE_OF", (entity, concept)
)

sensor = registry.symbol("sensor_a", sensor_type)
sample = registry.symbol("sample", entity)
accepted = registry.symbol("accepted", concept)
conclusion = registry.atom(instance_of, sample, accepted)

compiled = TypedDataflowCompiler(registry).compile(
    NumericDataflowRuleSpec(
        name="acceptance",
        conditions=(
            NumericConditionSpec(
                name="minimum",
                measurement=NumericMeasurementRef(
                    measured, (sensor,), value_position=1
                ),
                comparator=">=",
                threshold=Fraction(3, 4),
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
                measured,
                sensor,
                registry.symbol("4/5", number),
            ),
            source="deterministic_sensor",
        ),
    )
)
result = OperatorKernel(registry).solve(state, (Goal(conclusion),))
assert result.success and result.verified
```

## 검증 경계

컴파일러는 다음을 강제한다.

1. 측정 predicate와 결론 predicate가 현재 registry에 등록되어 있어야 한다.
2. 측정 predicate는 `verified=True`여야 한다.
3. 숫자 슬롯은 `Number`에 assignable해야 하고 나머지 인자는 ground typed term이어야 한다.
4. threshold는 부동소수점 근사가 아닌 `Fraction`이어야 한다.
5. 모든 성공 proof는 `OperatorKernel.replay`에서 guard를 다시 계산해야 한다.

컴파일러는 측정 사실이나 결론 사실을 직접 만들지 않는다. adapter가 제공한 rule
declaration만 `observed` fact로 넣고, condition과 conclusion은 operator 실행으로만
`derived`된다. `blocked_by`에 명시된 사실이 state에 있으면 최종 operator는 열리지
않는다.

## 자가 학습 평가

`generate_semantic_flow_transfer_split`은 문장이나 이름의 무작위 분할이 아니라 operator
program 형태를 분리한다.

| split | 프로그램 |
|---|---|
| train | 단일 threshold, 서로 다른 selector의 2조건 conjunction |
| held-out | 같은 측정값을 재사용하는 상·하한, 3조건 mixed comparison |

held-out에는 더 큰 이미지, 새로운 숫자, `number of` 문구와 한국어 문구가 들어간다.
양성 trace는 실제 proof에서 `vision`, `math`, `language` tag를 모두 지나야 하고, 음성
control은 verifier가 끝까지 미증명으로 남겨야 한다.

```powershell
python tools/eval/evaluate_semantic_flow_self_learning.py
```

2026-07-18 CPU 기준 기본 평가 결과는 positive expansion `44 -> 30`, 31.8% 감소다.
verified solve rate와 proof soundness는 100%, false positive는 0이며, 승격된 sparse
policy는 21 parameters, artifact 1,085 bytes였다.

## 현재 한계

- 현재 compiler는 하나의 ground numeric measurement predicate를 조건 하나의 입력으로 쓴다.
- 합계, 평균, 객체 집합 join은 별도 typed aggregation operator가 필요하다.
- raster vision은 작은 분리 단색 component만 deterministic observation으로 인정한다.
- 자연어는 controlled 한국어·영어 규칙 parser이며 자유 문장 의미 획득은 아직 아니다.
- 학습되는 것은 action ranking이다. 새 사실이나 정답을 신경망이 직접 쓰지는 못한다.
