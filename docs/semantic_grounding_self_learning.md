# Candidate-Level Semantic Grounding Self-Learning

## 목적

이 단계는 작은 shared brain이 언어, 수학, 비전의 입력을 직접 답으로 외우는 대신
typed candidate가 입력에 의해 지지되는지를 선택적으로 판단하도록 학습한다.

이번 구현이 이전 synthetic sensor 실험과 다른 점은 모든 예제가 실제 production
adapter를 지난다는 것이다.

- 언어: 결합 전제, 누락 전제, 차단, 모순, entity binding 원문
- 수학: 정확한 유리수 식, 경계 비교, off-by-one, 관계 극성 반례
- 비전: RGB raster의 정사각형, 누락 픽셀, 방향, count, area 반례

신경 정책의 `ACCEPT`는 사실 확정이 아니다. 후보는 계속 `PROPOSED`이며 독립 verifier
또는 사람이 승인해야만 observed fact가 될 수 있다.

## 왜 별도 review가 필요한가

`SemanticReviewRecord`는 문제 전체의 `expected_solved`가 맞는지를 검토한다. 이것을
문제 안의 모든 atom에 복사하면 case-level 정답이 candidate label로 누수된다.

새 `SemanticGroundingReview`는 다음 항목을 모두 SHA-256 digest로 결속한다.

- raw semantic case 전체
- production adapter가 만든 정확한 typed candidate
- 검토 대상 label인 `ACCEPT` 또는 `REJECT`
- `human:` namespace reviewer, timezone timestamp, 명시적 attestation

case, parser, sensor feature, candidate atom 중 하나라도 바뀌면 review는 `stale`이 되고
학습 예제로 변환되지 않는다. 리뷰어가 기존 curated label이 틀렸다고 판단하면 다른
`reviewed_label`을 승인할 수 있으며 audit에 `corrected_label_targets`로 남는다.

```text
raw LMV input
  -> production adapter
  -> typed goal-support candidate + input-only operator features
  -> model ACCEPT / REJECT / ABSTAIN
  -> PROPOSED only
  -> independent verifier or exact human review
  -> GroundingLearningExample
  -> validation-gated sparse policy update
```

## 공통 operator feature

센서 특징에는 `expected_solved`, proof success, verifier decision, gold label을 넣을 수
없다. `GroundingCandidate`가 이름 수준에서 이 조각들을 거부한다.

대신 각 adapter가 이미 계산할 수 있는 primitive 결과를 사용한다.

| 도메인 | primitive 입력 특징 | 주요 near-miss |
| --- | --- | --- |
| 언어 | requirement 수, 충족/차단/모순 비율 | 한 전제 누락, 엉뚱한 entity, 동시 모순 |
| 수학 | exact rational delta, 관계 종류, 경계 여부 | off-by-one, `!=` 극성, `<=` 경계 |
| 비전 | fill ratio, extent, centroid 방향, count gap, area margin | 픽셀 하나 누락, 방향 반전, count 1 차이 |

서로 다른 primitive는 공통 `operator.support_margin`으로 정규화된다. 이는 solver의
성공 label이 아니라 입력에서 결정론적으로 재계산한 관계 margin이다. 작은 sparse
head는 이 공통 비교 구조와 typed predicate를 조합한다. 복잡한 exact reasoning은
여전히 kernel operator와 proof replay가 담당한다.

## 공개 API

구현 책임은 세 파일로 나뉜다.

- `semantic_grounding.py`: candidate/review 계약, audit, 학습 예제 bridge
- `semantic_grounding_features.py`: label-free domain primitive와 shared margin
- `semantic_grounding_cases.py`: raw controlled LMV case와 programmatic oracle corpus

```python
from semop.kernel import (
    SemanticGroundingCorpus,
    create_semantic_grounding_review,
    load_semantic_benchmark,
)

benchmark = load_semantic_benchmark("data/semantic_benchmark/v1/cases.jsonl")
corpus = SemanticGroundingCorpus.compile(benchmark)
target = corpus.targets[0]

review = create_semantic_grounding_review(
    target,
    reviewed_label="accept",
    reviewer="human:reviewer-name",
)

# review가 exact target과 일치할 때만 한 개의 HUMAN_REVIEW 예제가 생긴다.
reviewed = SemanticGroundingCorpus(corpus.targets, (review,))
examples = reviewed.learning_examples()
```

통제된 raw LMV corpus는 다음 API로 만든다.

```python
from semop.kernel import (
    LearningSplit,
    generate_controlled_semantic_benchmark,
    programmatic_semantic_learning_examples,
)

raw = generate_controlled_semantic_benchmark(
    per_domain=20,
    split=LearningSplit.TRAIN,
    seed=73,
    namespace="experiment-a",
)
examples = programmatic_semantic_learning_examples(raw)
```

이 programmatic label은 controlled generator의 독립 oracle에 한정된다. 자유 문장이나
자연 이미지의 사람 의미 gold로 세지 않는다.

## 사람 review 절차

전체 후보와 audit를 본다.

```powershell
python tools/eval/review_semantic_grounding.py
```

후보 하나의 원문, atom, sensor feature, digest를 확인한다.

```powershell
python tools/eval/review_semantic_grounding.py `
  --case-id language-ready-two-requirements
```

직접 확인한 뒤에만 기록한다.

```powershell
python tools/eval/review_semantic_grounding.py `
  --case-id language-ready-two-requirements `
  --reviewer human:reviewer-name `
  --label accept `
  --notes "원문과 정확한 READY 후보를 직접 검토함"
```

case-level review 파일은 이 명령이 읽지 않는다. frontier LLM은 label을 제안할 수는
있지만 `human:` attestation을 대신 만들 수 없다.

## 평가

```powershell
python tools/eval/evaluate_semantic_grounding_learning.py `
  --label-budgets 0,5,20,100 `
  --checkpoint-root artifacts/semantic_grounding_learning `
  --output artifacts/semantic_grounding_learning/report.json
```

2026-07-18의 고정 seed 실행은 validation 24개/도메인, untouched test 48개/도메인을
사용했다.

| raw labels/domain | promoted | parameters | artifact | test completion | coverage | selective accuracy | false accepts |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | no | 0 | 249 B | 0.0% | 0.0% | n/a | 0 |
| 5 | no, rollback | 0 | 249 B | 0.0% | 0.0% | n/a | 0 |
| 20 | yes | 344 | 31,460 B | 91.7% | 91.7% | 100% | 0 |
| 100 | yes | 607 | 55,309 B | 100% | 100% | 100% | 0 |

20-shot completion은 100-shot의 `91.7%`여서 현재 저자원 gate `>=90%`를 통과했다.
5-shot 후보는 정확도는 높았지만 coverage와 math/vision domain coverage가 부족해
승격되지 않았고, active policy는 0-parameter abstain baseline으로 rollback했다.

기존 20-case semantic seed에 대한 100-shot policy의 curated label 기준 결과는
coverage와 selective accuracy가 각각 100%였다. 그러나 candidate-level human review는
현재 `0`개이므로 이 값은 informational regression일 뿐 `semantic_correctness`나
open-domain 성능으로 보고하지 않는다. report의 `human_semantic_gate`는 정확히
`not_evaluated`다.

## 연구 연결

- [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html)은 raw
  입력과 최종 예측 사이에 사람이 수정할 수 있는 concept 경계를 둔다. SemOp의
  candidate review는 이 개입을 typed atom과 digest 수준으로 제한한다.
- [DeepProbLog](https://proceedings.neurips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html)은
  neural predicate와 논리 추론을 결합한다. SemOp은 end-to-end 확률 추론 대신 neural
  output을 proposal로 격리하고 proof replay를 신뢰 경계로 둔다.
- [A Generalist Neural Algorithmic Learner](https://proceedings.mlr.press/v198/ibarz22a.html)은
  여러 알고리즘이 graph processor를 공유할 수 있음을 보였다. 공통
  `operator.support_margin`은 훨씬 작은 sparse setting에서 같은 가설을 시험한다.
- [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5)는 neural guidance와
  symbolic deduction의 역할 분리를 강하게 보여 준다. 다만 대규모 합성 theorem을
  사용하므로 이번 수백 개 controlled case 결과와 성능을 직접 비교할 수 없다.
- [DreamCoder](https://arxiv.org/abs/2006.08381)는 반복 program을 library abstraction으로
  압축한다. SemOp의 다음 단계인 verifier-retained macro operator 발견의 근거다.
- [ExeDec](https://openreview.net/forum?id=oTRwljRgiv)는 실행 분해가 program synthesis의
  compositional generalization에 도움이 될 수 있음을 보여 준다. 향후 candidate가
  어떤 primitive 조합에 의존했는지 supervision하는 실험과 연결된다.

이 연구들은 방향의 근거이지 SemOp의 일반지능 또는 open-domain 이해를 입증하지
않는다.

## 다음 증거 gate

1. 세 도메인 candidate-level 사람 review를 train/calibration/untouched test로 고정한다.
2. 0/5/20/100 사람 review 곡선을 최소 20개 seed와 신뢰구간으로 재실행한다.
3. 언어는 자유 paraphrase, 관계 방향, 부정 범위, 현실 증거 출처를 추가한다.
4. 수학은 candidate AST의 잘못된 상수와 operator를 직접 제안하는 contrast grounder를
   추가한다.
5. 비전은 작은 learned object encoder의 자연 이미지 proposal을 넣되 계속
   `PROPOSED`로 유지한다.
6. shared margin 제거, domain-only margin, surface hash 제거 ablation을 수행한다.
7. grounding 선택 정확도와 최종 operator-search solve rate를 별도로 측정한다.

## 검증

```powershell
python -m pytest tests/test_typed_operator_semantic_grounding.py -q
python -m pytest tests/test_typed_operator_semantic_grounding_eval.py -q
python -m pytest -k typed_operator
python -m pytest
```
