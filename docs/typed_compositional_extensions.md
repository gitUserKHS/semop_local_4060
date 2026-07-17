# Typed Compositional Extensions v2

## 왜 이 리팩터링을 했는가

초기 typed MVP는 세 입력 경계를 연결했지만 능력 폭이 좁았다.

- 언어: 목표의 필수 전제와 차단 상태
- 수학: 변수 없는 정확 산술식
- 비전: component 사이의 공간 관계

v2는 모델 크기를 늘리지 않고 각 영역에서 새로운 operator 조합을 추가한다. 답을
생성하는 신경망을 키운 것이 아니라, 기존 5.84M controller가 선택할 수 있는 검증 가능
program vocabulary를 확장했다.

```text
언어: INSTANCE_OF + SUBCLASS_OF -> inherited INSTANCE_OF
수학: LEFT/RIGHT_LINEAR_FORM -> NORMALIZED_LINEAR -> SOLUTION
비전: pixel facts -> shape / count / area comparison
```

세 경로는 모두 `DomainInstance -> OperatorKernel -> proof replay` 계약을 공유한다.

## 언어: 개념 포함 Horn 추론

`LanguageLogicAdapter`는 제한된 한국어·영어 분류 문장을 읽는다.

```text
Every programmer is a person.
Every person is mortal.
Ada is a programmer.
Prove: Ada is mortal.
```

```text
모든 개발자는 사람이다.
모든 사람은 생명체이다.
아다는 개발자이다.
증명: 아다는 생명체이다.
```

보편 문장은 `SUBCLASS_OF`, 개체 분류는 `INSTANCE_OF`가 된다. 상위 개념의 추이성과
개체 분류 상속은 서로 다른 typed operator다. 명시적 `NOT_INSTANCE_OF`가 있으면
긍정 분류 operator guard가 도출을 거부한다. 긍정·부정이 직접 충돌하면 두 입력 모두
`contradicted`여서 proof에 사용할 수 없다. 부정으로부터 contraposition을 수행하지
않는다.

plain `str` 입력은 `Prove:`/`증명:` 또는 보편문과 typed `Goal:` 조합이 있을 때만 이
parser로 자동 분기한다. 더 애매한 문장은 기존 숨은 전제 parser의 trust boundary를
따른다. `LanguageLogicProblem`으로 명시적인 분기도 가능하다.

## 수학: 정확한 일차방정식

`MathInputAdapter`는 `=` 유무로 정확 산술과 일차방정식을 분기한다.

```python
result = runtime.run(TypedDomainRequest("math", "3*(x - 2) = x + 4"))
```

전용 recursive-descent parser는 `Fraction`만 사용하며 다음을 지원한다.

- 한 개의 변수와 양변의 변수항
- 괄호, 단항 부호, `+ - * /`
- `2*x`와 `2x` 표기
- 소수와 분수의 exact rational 해석

parser가 만든 양변의 선형형은 첫 operator에서 `a*x=b`로 정규화되고, 두 번째
operator가 0이 아닌 `a`로 나누어 해를 만든다. 두 단계 모두 guard가 재계산한다.
비선형 곱셈, 다변수, 변수식으로 나누기, 0으로 나누기, 무한해와 무해는 위치 포함
`LinearEquationError`로 거절한다.

legacy `ArithmeticReasoner`의 직접 방정식 경로도 이 exact adapter와 proof replay에
위임하므로 float 풀이가 별도로 진화하지 않는다.

## 리팩터링과 코드 표면

한국어·영어 statement 분리와 identifier 정규화는 `language_common.py`, exact rational
표현은 `math_common.py`로 모아 adapter 사이의 중복 구현을 제거했다. runtime은 언어와
수학에서 각각 `LanguageInputAdapter`, `MathInputAdapter` 하나만 진입점으로 사용한다.

`audit_code_surface.py`의 v3 정적 감사에서는 활성 표면 190개가 확인됐고
`deletion_ready` 후보는 없었다. 기존 GUI, 동적 import, 저장 artifact 호환성은 정적
검색만으로 안전한 삭제를 증명할 수 없기 때문에 남겼다. 이 코드는 shadow telemetry와
typed A/B gate가 실제 대체를 입증한 기능군부터 별도 변경으로 제거한다.

## 비전: 도형·개수·면적

`RasterVisionProblem`은 기존 `VisionRelationGoal`과 함께 다음 goal을 받는다.

- `VisionPropertyGoal("SQUARE", subject)`
- `VisionCountGoal("all" | color, expected)`
- `VisionAreaGoal(larger, smaller)`

`FILLS_BOUNDING_BOX`는 최소 2x2이고 component의 모든 bbox 픽셀이 실제로 채워졌을
때만 observed가 된다. 여기에 `EQUAL_EXTENT`를 합성해야 `SQUARE`를 증명할 수 있다.
따라서 중심점이나 단순 bbox 비율만으로 정사각형을 선언하지 않는다.

개수는 segmentation 설정을 통과한 component 집합에 대한 closed-world 연산이다.
모든 component의 `VISUAL_ENTITY`와 `HAS_COLOR` fact를 전제로 ground count operator가
상태를 다시 센다. 작은 noise로 무시된 component는 개수에 포함되지 않으며 metadata의
`ignored_component_count`에 남는다. 면적 비교는 각 component의 정확한 pixel count를
guard에서 다시 비교한다.

## Controller 점수 계약 v5

감사 과정에서 PyTorch 학습과 NumPy 추론의 action logit 구성이 다르다는 문제를
찾았다. v5는 양쪽 모두 다음 joint score를 사용한다.

```text
tanh(operator + state compatibility)
+ tanh(typed argument pointer)
+ learned structural score
```

latent와 pointer를 bounded해 out-of-domain hash/message logit이 typed goal-overlap
head를 무제한으로 덮지 못하게 했다. 1등과 2등 점수 차가 0.1 이상일 때만 한 action
beam을 요청하고, 불확실하면 기본 beam 4를 유지한다. 이 결정은 성공 판정이 아니며
모든 effect는 여전히 executor와 replay를 통과해야 한다.

## 실행

```powershell
python examples/typed_compositional_v2_demo.py
```

이 예제는 언어 2-hop 분류, 양변 일차방정식, raster 정사각형·개수·면적 비교를 같은
runtime과 policy로 실행한다.

## 2026-07-17 평가 스냅샷

v2 기능에 v3의 conjunctive rule, exact numeric comparison, goal-independent raster
관측을 더한 suite는 36개 사례다. 실제 도메인 간 조합은
`docs/composed_operator_runtime.md`의 별도 8-case suite로 측정한다.

- 기대 성공 25건: 전부 성공하고 replay 검증
- 음성 대조군 11건: 전부 거절
- reported proof soundness: 100%
- non-learned goal-directed expansion: `156 -> 81`
- domain median: 언어 `33.3%`, 수학 `37.5%` 절감
- composed-v4: 양성 5건을 3~5-step proof로 검증, 음성 3건 거절, `47 -> 34`

5,837,578-parameter controller의 LODO는 held-out마다 나머지 두 도메인의 synthetic
trace 20개씩, 총 40개를 5 epoch 학습했다. 세 holdout 모두 solve rate와 soundness
100%를 유지했고 언어 median `3 -> 2`, 수학 `6.5 -> 4`, 비전 전용 distractor
`13 -> 1`을 기록했다. artifact는 약 21.64MB, 최대 held-out p95는 약 0.083초였다.

이는 synthetic typed 구조 전이 결과다. 새 Horn distractor 자체의 fresh neural 절감은
`15 -> 13`으로 아직 약하다. 사람 검토 20/100-shot, 자유문장 분포 이동, 다변수 대수,
자연사진 의미 인식은 검증되지 않았으므로 promotion과 일반지능 달성 주장은 하지 않는다.
