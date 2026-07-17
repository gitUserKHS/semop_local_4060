# Composed Operator Runtime v4

## 목적

이 단계의 핵심은 언어, 수학, 비전을 각각 실행하는 데서 멈추지 않고, 세 도메인의
operator가 하나의 검증 가능한 프로그램을 이루게 하는 것이다. 작은 controller는
정답 사실을 직접 쓰지 않는다. 현재 상태에서 실행 가능한 typed action의 순서를
고를 뿐이며, executor와 proof replay가 모든 성공을 다시 검사한다.

현재 구현된 대표 프로그램은 다음과 같다.

```text
RGB pixels
  -> count_raster_objects
  -> compare_scene_object_count
  -> classify_scene_from_verified_count
  -> INSTANCE_OF(scene, crowded)
```

예를 들어 물체가 세 개인 작은 raster와 다음 문장을 함께 입력할 수 있다.

```text
If the number of all objects is greater than 2, the scene is crowded.
Prove: the scene is crowded.
```

한국어의 같은 제약 문장도 동일한 typed operator program으로 정규화된다.

v4는 한 조건에서 멈추지 않고 최대 8개의 count 조건을 하나의 불변 IR로 묶는다.

```text
If the count of red objects is at least 1 and
the count of blue objects is at least 1, the scene is colorful.
```

각 조건은 별도의 `Condition` identity, selector, comparator, exact `Fraction`
threshold를 가진다. 비전 count는 selector별로 한 번만 측정하며, 같은 count에 상한과
하한을 함께 적용하면 관측 결과를 재사용한다. 마지막 분류 operator는 모든
`COUNT_CONDITION_MET` 전제가 있을 때만 열린다. 한 조건이라도 거짓이면 결론을 만들지
않고, replay도 각 guard를 다시 계산한다.

## 공통 계약 리팩터링

도메인 어댑터의 공통 계약은 `semop.kernel.contracts`가 소유한다.

- `DomainInstance`: registry, 초기 `WorldState`, `Goal`, domain, metadata
- `TypedDomainAdapter`: `adapt`와 선택적 legacy projection 계약
- `domains.base`: 기존 import를 위한 얇은 호환 re-export

`compose_domain_instances`는 여러 `DomainInstance`를 한 registry와 state로 합친다.
조합 과정은 다음 조건을 강제한다.

- 동일 명목 타입은 부모가 호환될 때만 재사용한다.
- 동일 function/predicate 이름은 typed schema가 같을 때만 재사용한다.
- 충돌하는 operator 이름은 component alias로 범위를 나눈다.
- guard 이름도 component alias로 분리한다.
- 이름 공간을 분리해야 하는 타입은 `namespace_symbol_types`로 지정한다.
- source term, fact, goal은 문자열 치환이 아니라 typed term 재구성으로 옮긴다.

타입 부모가 충돌하거나 같은 predicate가 다른 인자 타입을 요구하면 조합은 즉시
`RegistryCompositionError`를 낸다. 조용히 의미를 섞지 않는다.

## 세 도메인의 v4 확장

### 언어

`LanguageLogicAdapter`는 단항 Horn chain뿐 아니라 2개 이상 전제를 요구하는
conjunctive rule을 지원한다.

```text
Rule: red & square -> marker
Fact: tile is red
Fact: tile is square
Prove: tile is marker
```

전제 하나가 없거나 `NOT_INSTANCE_OF`가 명시된 경우에는 긍정 결론을 만들지 않는다.
이는 자유 자연어 이해기가 아니라 높은 정밀도의 통제 언어 adapter다.

### 수학

`NumericComparisonAdapter`는 `<`, `<=`, `>`, `>=`, `==`, `!=`를 양쪽 exact
산술식과 조합한다. 모든 숫자는 `Fraction`으로 계산하므로
`0.1 + 0.2 == 0.3`도 부동소수점 근사 없이 증명된다. 거짓 비교는 goal이
도출되지 않으며 replay에서 guard를 다시 계산한다.

### 비전

`RasterVisionAdapter`는 goal이 없어도 component, color, pixel area와 요청된 count
관측 연산자를 준비할 수 있다. 따라서 뒤쪽 언어 goal이 앞쪽 비전 answer operator를
몰래 결정하지 않는다. component 수는 기본 256개로 제한해 O(n^2) 관계 생성을
제어한다.

현재 비전 입력은 작은, 분리된 단색 RGB component에 한정된다. 실사 객체 인식이나
학습 detector의 출력을 곧바로 observed fact로 승격하지 않는다.

## Operator Frontier

최종 goal의 operator가 아직 실행 불가능하면 커널은 operator schema를 뒤로 따라가
현재 실행 가능한 grounded precondition을 controller용 frontier로 만든다.

```text
verifier goal: INSTANCE_OF(scene, crowded)
frontier 1:    COUNT_CONDITION_MET(all, crowded)
frontier 2:    OBJECT_COUNT(all, 3)
```

frontier는 fact도 아니고 성공 조건도 아니다. 다음 action을 점수화할 때만 전달된다.
`GoalDirectedPolicy`는 실제 verifier goal을 frontier보다 우선하며, tiny controller의
canonical graph에는 각각 `goal`, `frontier` 역할로 따로 인코딩된다. controller가
halt를 요청해도 원래 goal이 state에 없으면 executor가 거절한다.
재귀 term을 만드는 확장 operator가 잘못 등록되어도 frontier 역추적은
`SolveBudget.max_steps`와 fact/expansion 상한 안에서 중단된다.
verifier trace를 decision 학습 자료로 바꿀 때도 `OperatorKernel.policy_goals`를 호출해
학습과 CPU 추론이 같은 frontier 입력 계약을 사용한다.

## 검증 경계

성공 결과는 항상 다음 순서를 통과한다.

1. 현재 eligible fact만으로 ground action을 열거한다.
2. policy가 action 순서를 제안한다.
3. executor가 타입, 전제, guard, effect를 검사해 action을 실행한다.
4. supporting proof DAG만 추출한다.
5. 초기 state부터 operator program을 다시 replay한다.
6. 원래 verifier goal이 모두 존재할 때만 `verified=True`를 반환한다.

`proposed`, `contradicted`, unverified predicate는 proof 전제로 사용할 수 없다.

## 실행과 평가

```powershell
python examples/typed_cross_domain_scene_demo.py
python tools/eval/evaluate_low_resource_transfer.py --suite language-math-vision
python tools/eval/evaluate_low_resource_transfer.py --suite composed-v4
```

2026-07-17 deterministic 측정에서 language/math/vision suite 36건은 양성 25건을
모두 풀고 음성 대조군 11건을 모두 거절했다. proof soundness는 100%, false positive는
0건이며 `GoalDirectedPolicy`는 총 expansion을 `156 -> 81`로 줄였다. 별도의
`composed-v4` 8건도 양성 5건과 음성 3건을 정확히 처리했다. 양성 proof 길이는
`3, 3, 5, 5, 4`였고, 전체 expansion은 `47 -> 34`였다. 여기에는 영어·한국어
다중 조건, 동일 관측 재사용, 거짓 conjunct 대조군이 포함된다.

이 결과는 통제 입력에서의 symbolic baseline 증거다. 자유 자연어, 일반 수학,
자연 이미지의 범용 지능이나 학습된 controller의 도메인 외 일반화를 증명하지 않는다.

기존 frontier-aware 29,834-parameter LODO 진단 artifact 세 개를 v4에 다시 적용했을
때도 soundness 100%, false positive 0을 유지하며 expansion이 `47 -> 40`이었다.
별도의 composed synthetic trace 20개, 5 epoch 진단도 같은 `47 -> 40`을 기록했다.
이는 학습/NumPy 추론 파이프라인과 새 schema의 제한된 호환성 증거다. 비학습
`GoalDirectedPolicy`의 34 expansion보다 아직 나쁘며, 5.84M 전체 모델이나 큰 분포
이동에서의 일반화 증거는 아니다.

## 확장 순서

새 교차 도메인 기능은 다음 순서로 추가한다.

1. 각 도메인 adapter가 goal과 독립적인 observed fact를 만들게 한다.
2. 공통 타입과 predicate schema를 명시한다.
3. `compose_domain_instances`로 registry 충돌을 검사한다.
4. 도메인 사이를 잇는 최소 typed operator와 guard를 등록한다.
5. 성공, 잘못된 binding, 누락 전제, 명시적 모순 테스트를 함께 작성한다.
6. proof replay 변조 테스트와 expansion A/B를 통과시킨다.

이 절차를 지키면 작은 신경망은 계속 조합 선택에 집중하고, 새 능력의 진실성은
도메인 verifier가 책임진다.
