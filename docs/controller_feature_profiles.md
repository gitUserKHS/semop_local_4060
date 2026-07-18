# Controller Feature Profiles

## 왜 필요한가

SemOp의 shared controller는 언어, 수학, 비전의 답을 직접 생성하지 않는다. 현재 상태에서
적용 가능한 typed operator action을 정렬하고, 실제 상태 변경과 성공 판정은
`OperatorKernel`과 proof replay가 담당한다.

하지만 symbol 이름만 익명화한다고 곧바로 도메인 전이가 보장되지는 않는다. 기존
canonical graph에는 다음과 같은 identity shortcut이 남아 있었다.

- `schema:verify_required_premise` 같은 operator 고유 이름
- `effect:READY`, `precondition:VALUE` 같은 predicate 고유 이름
- `tag:vision`, `tag:spatial` 같은 도메인 tag
- relation과 term function의 고유 이름

이 토큰으로도 현재 synthetic benchmark를 잘 풀 수 있다. 따라서 높은 점수가 operator
구조의 학습인지, 학습 세트에서 반복된 이름의 암기인지 분리해서 측정해야 한다.

## 두 입력 프로필

`ControllerFeatureProfile`은 같은 typed state를 두 방식으로 canonicalize한다.

| 프로필 | 용도 | controller가 보는 것 |
|---|---|---|
| `full` | 이전 artifact 호환과 대조군 | 구조 특징과 operator/predicate/function 이름 |
| `typed_structure` | 새 학습 기본값과 전이 실험 | 타입 서명, arity, 검증 여부, symmetry, operator family, goal 정합성 |

두 프로필 모두 entity와 symbol의 실제 이름은 익명화한다. `typed_structure`는 추가로
schema, predicate, function, tag identity를 숨긴다. predicate와 term의 symmetry group은
anonymous node index로 다시 정렬하므로, 대칭 인자의 원래 이름순 정렬도 입력 신호로
남지 않는다. `anon:Type:index`와 `arg:node_id` 토큰도 제거하고 function application은
typed relation edge로 표현한다. 다음 정보는 보존한다.

- nominal `TypeRef`와 argument 위치
- domain-neutral `OperatorFamily`
- 전제, 효과, guard, binding 개수와 비용
- effect와 goal의 exact/predicate/type 정합성
- goal argument overlap과 새로운 effect 비율
- observed/assumed/derived fact role과 verifier/frontier goal role

타입 이름은 의도적으로 남긴다. SemOp의 가설은 untyped 구조 하나가 모든 입력을 처리한다는
것이 아니라, 다른 감각 adapter가 만든 typed graph를 같은 operator processor가 다룬다는
것이기 때문이다.

## 실행 경로

```text
Domain adapter
  -> typed WorldState + Goal + applicable GroundAction
  -> canonicalize_problem(feature_profile)
  -> sparse or recurrent controller scores
  -> OperatorKernel executes registered action only
  -> proof replay verifies every successful result
```

`TinyControllerConfig.feature_profile`은 PyTorch 학습 encoder와 NumPy CPU runtime에 함께
저장된다. 따라서 학습과 추론에서 서로 다른 표현을 실수로 사용할 수 없다.

- 새 `train_tiny_controller.py` 실행 기본값: `typed_structure`
- 새 LODO 실험 기본값: `typed_structure`
- `TinyControllerConfig()` 기본값: `full`

마지막 기본값은 기존 코드를 호출하거나 v2-v5 artifact를 읽는 경우의 의미를 보존하기
위한 것이다. 새 NumPy artifact format은 v6이며, 이전 v2-v5를 계속 읽는다.
`StructuralLinearPolicy` v2도 v1 artifact를 `full` 프로필로 migration한다.

## 빠른 identity-ablation LODO 게이트

다음 명령은 PyTorch 없이 실행된다.

```powershell
python tools/eval/evaluate_controller_feature_transfer.py --require-pass
```

기본 실행은 도메인마다 8개 verified synthetic problem을 학습 pool에 넣고, 다른 seed의
12개를 holdout으로 둔다. 언어, 수학, 비전을 한 번씩 통째로 학습에서 제외한다. 각 run은
나머지 두 도메인의 verifier trace만으로 작은 `StructuralLinearPolicy`를 학습한다.

현재 고정 seed의 기본 결과는 다음과 같다.

| 측정 | `full` | `typed_structure` |
|---|---:|---:|
| 노출된 identity token | 224 | 0 |
| 전체 operator feature vocabulary | 105 | 41 |
| 언어 holdout action top-1 / replay | 100% / 100% | 100% / 100% |
| 수학 holdout action top-1 / replay | 100% / 100% | 100% / 100% |
| 비전 holdout action top-1 / replay | 100% / 100% | 100% / 100% |

`typed_structure` sparse artifact는 holdout별 21-26개 파라미터다. median expansion은
언어 `7 -> 5`, 수학 `9 -> 9`, 비전 `5 -> 1`이었다. 성공으로 보고된 모든 결과는 별도의
primitive proof replay를 다시 통과했다.

pairwise feature Jaccard는 보고서에 진단값으로 남기지만 통과 기준으로 사용하지 않는다.
기본 크기에서 `full`의 같은 이름 synthetic distractor가 겉보기 교집합을 부풀리는 반례가
발견됐기 때문이다. 전이 주장은 이름 중복률이 아니라 실제 held-out action 선택과 replay로
판정한다.

이 결과는 통제된 symbolic LMV curriculum의 operator-policy 전이다. 자유 자연어 이해,
고등수학 전반, 자연 사진 인식의 증거가 아니다.

## 연구 근거

[A Generalist Neural Algorithmic Learner](https://arxiv.org/abs/2209.11142)는 task별
입출력 경계와 하나의 graph processor로 여러 알고리즘을 학습하고 OOD 일반화를 평가한다.
SemOp도 adapter는 분리하지만 action processor는 공유한다. 차이는 processor가 직접 상태를
만들 수 없고 typed executor가 모든 행동을 재검증한다는 점이다.

[GOAL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/826aea2253363fe04e8c4991b2a8869e-Abstract-Conference.html)은
여러 조합최적화 문제에 공통 backbone과 가벼운 문제별 adapter를 사용한다. SemOp의
`DomainCatalog`와 shared controller도 같은 방향이지만, v1에서는 mixed-attention을
복제하지 않고 작은 recurrent/sparse policy와 명시적 typed action space를 사용한다.

[Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z)는
표준 benchmark에서 잘 작동하지만 어려운 조건으로 전이되지 않는 쉬운 decision rule을
구분해야 한다고 지적한다. `full` 대조군, identity audit, domain holdout은 이 위험을 직접
검사하기 위한 것이다.

[The CLRS Algorithmic Reasoning Benchmark](https://proceedings.mlr.press/v162/velickovic22a.html)는
서로 다른 알고리즘과 문제 크기에 대한 통일된 평가의 중요성을 보여 준다. SemOp의 현재
LMV fixture는 CLRS 규모가 아니므로, 구조·크기·operator 조합 holdout을 계속 넓혀야 한다.

## 다음 단계

1. full recurrent 5.84M 모델도 두 feature profile로 같은 LODO seed 집합에서 비교한다.
2. 현재 점 이름·대칭 인자 score invariance를 더 복잡한 isomorphic graph pair로 확장한다.
3. grammar-constrained language producer와 object-centric vision producer가 같은 typed graph
   계약을 출력하게 한다.
4. 검증된 실제 사용자 문제와 near-miss를 추가하되 training, validation, final test를
   digest로 고정한다.
5. 구조 프로필에서 실패한 case만 human/frontier review queue로 보내 능동 학습한다.
