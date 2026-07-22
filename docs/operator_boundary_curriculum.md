# 검증형 Operator-Boundary 데이터와 저자원 Curriculum

## 무엇을 자동화할 수 있나

SemOp은 데이터를 직접 만들 수도 있고, 공개 인터넷 데이터에서 가져올 수도 있다. 다만 출처가 다르면 학습 권한도 달라진다.

| 경로 | 처음 상태 | 학습에 들어가는 조건 |
| --- | --- | --- |
| typed generator가 만든 합성 문제 | `programmatic_oracle` 후보 | 같은 operator program의 실행과 proof replay가 일치해야 함 |
| 인터넷 공개 데이터의 기존 라벨 | `published_label` 후보 | 라이선스, 고정 revision, SHA-256 확인 후 SemOp adapter와 별도 verifier를 통과해야 함 |
| 코아 또는 frontier LLM이 만든 문제와 라벨 | `model_proposal` | 정답으로 간주하지 않으며 exact verifier 또는 사람 검토가 필요함 |
| 사람이 확인한 candidate | `human_review` | raw input, typed candidate, label을 묶은 digest와 reviewer attestation이 필요함 |

즉 코아는 새 표현, hard negative, operator 조합, 실패 사례를 대량으로 제안할 수 있다. 그러나 코아 자신이 만든 답을 코아의 판단만으로 정답 데이터로 승격시키지는 않는다.

현재 고정된 공개 소스 목록은 `data/sources/semantic_sources.v1.json`에 있다.

- 언어: AllenAI RuleTaker
- 수학: DeepMind Mathematics Dataset generator
- 비전: DeepMind dSprites

`tools/data/fetch_semantic_sources.py`는 명시적 소스 선택, 라이선스 확인, 다운로드 상한, HTTPS, revision과 SHA-256 receipt를 강제한다. 다운로드한 파일은 기본적으로 `artifacts/`에 두며 Git에 포함하지 않는다.

## 이번에 추가한 합성 데이터

`generate_operator_boundary_semantic_benchmark()`는 production language, math, raster-vision adapter를 실제로 통과하는 세 도메인 문제를 만든다.

- 언어: 1-4개 요구사항, 누락, 차단, 모순, 잘못된 entity binding, 문장 순서 변경
- 수학: exact arithmetic, 부호, 영과 단위 경계, 여섯 비교 연산자의 참/거짓 경계, 선형식
- 비전: 크기가 다른 정사각형, 방향 관계, 반복 색상 count, area, hard negative

각 예제의 라벨은 generator 문자열에서 바로 복사하지 않는다. `programmatic_semantic_learning_examples()`가 raw 입력을 production adapter로 다시 grounding하고, kernel 결과와 replay를 확인한 뒤에만 `EXTERNAL_VERIFIER` 학습 레코드를 만든다.

`generate_extrapolation_semantic_benchmark()`는 더 큰 언어 결합, 더 깊은 exact expression, 더 큰 raster와 distractor를 만든다. 현재 이 split은 feature 설계 중 여러 번 열어보았으므로 **개발 진단용**이다. sealed final 또는 독립적인 open-domain 증거로 부르지 않는다.

## 작은 모델을 위한 feature v4

`grounding_features.py`의 v4 표현은 entity 이름과 gold label을 사용하지 않는다.

1. 모든 numeric sensor를 `zero/nonzero`, `positive/negative`, `unit/fractional/superunit` 경계로 분해한다.
2. 수학 비교와 비전 방향·면적을 canonical operator role로 투영한다.
3. typed predicate/relation과 sensor state의 희소 interaction을 만든다.
4. 관계 연산자와 결정 경계·argument role을 canonical decision signature로 묶는다.
5. 원시 sensor contract는 verified training에서 최소 2회 지원되어야 한다.
6. 핵심 decision signature를 한 번도 보지 못하면 policy는 점수를 내더라도 `ABSTAIN`한다.

interaction은 모델 내부 파생 basis이며 외부 관찰 사실이 아니다. 따라서 원시 sensor와 같은 권한을 갖지 않고, policy가 fact를 직접 만들 수도 없다. 성공 사실은 계속 typed executor와 proof replay만 만들 수 있다.

feature format이 달라졌으므로 v1-v3 `.npz`/JSON grounding policy artifact는 v4 loader에서 fail closed로 거절된다.

## Feature-Novel Curriculum

`select_feature_novel_grounding_examples()`는 이미 독립 검증된 예제만 선택한다.

- 도메인별 고정 quota
- accept/reject 최소 quota
- 새 feature state 2점
- predictor의 두 번째 support 관측 1점
- 희귀 상태와 라벨 균형을 이용한 결정적 tie-break
- record digest 중복과 pool 밖 선택 거절

감사 결과에는 한 번 이상 본 feature-state coverage와 support 2회를 충족한 coverage를 모두 기록한다. 같은 입력에서는 선택 순서와 SHA-256 selection digest가 항상 같다.

## 개발 A/B 결과

실행 명령:

```powershell
python tools/eval/evaluate_semantic_grounding_curriculum.py `
  --require-pass `
  --output artifacts/semantic_grounding_curriculum_v10.json
```

고정 seed 개발 결과는 다음과 같다. `primitives_only` profile이라 surface hash와 support margin을 사용하지 않는다.

| labels/domain | strategy | candidate completion | deployed completion | selective accuracy | false accepts | retained |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 5 | prefix | 19.4% | 0.0% | 100% | 0 | no |
| 5 | feature-novel | 5.6% | 0.0% | 100% | 0 | no |
| 20 | prefix | 66.7% | 66.7% | 100% | 0 | yes |
| 20 | feature-novel | 91.7% | 91.7% | 100% | 0 | yes |

20-shot feature-novel policy는 1,071개 희소 파라미터를 사용했다. 도메인별 개발 coverage는 언어 83.3%, 수학 91.7%, 비전 100%였고 결정한 항목은 모두 맞았다.

5-shot에서는 novelty 후보가 prefix보다 나빴다. 둘 다 validation coverage gate를 넘지 못해 실제 배포는 0-parameter abstain fallback으로 돌아갔다. 보고서는 candidate 회귀를 evidence에 남기되, 배포 정책 비회귀와 false accept 0을 별도 gate로 계산한다.

이 결과가 증명하는 것은 controlled adapter 위에서 검증형 curriculum이 적은 라벨을 더 효율적으로 고를 수 있다는 점까지다. 다음은 아직 증명하지 않았다.

- 자유형 자연어 이해
- 자연 이미지 object discovery
- 사람이 작성한 semantic gold 정확도
- 인터넷 데이터의 자동 정답 승격
- 새 primitive operator의 자율 발명
- 범용 지능

## 연구 연결

- [Neural Arithmetic Logic Units](https://proceedings.neurips.cc/paper/8027-neural-arithmetic-logic-units)는 작은 모듈이 학습 범위 밖 수치로 외삽하려면 일반 MLP와 다른 산술 inductive bias가 필요함을 보여준다.
- [Deep Lattice Networks](https://papers.neurips.cc/paper/6891-deep-lattice-networks-and-partial-monotonic-functions.pdf)는 알려진 단조성과 부분 구조를 모델에 넣어 적은 데이터에서 합리적인 경계를 유지하는 방법을 제공한다.
- [Relation Networks](https://arxiv.org/abs/1706.01427)는 객체 쌍 관계를 명시적으로 계산하는 작은 relational module의 근거다.
- [Neural Logic Machines](https://iclr.cc/virtual/2019/poster/816)는 predicate와 논리 연산의 반복 적용이 크기와 조합이 달라진 문제로 전이될 수 있음을 보인다.

SemOp은 이 연구를 그대로 복제하지 않는다. 현재 v4는 산술·관계·경계 inductive bias를 typed sparse feature로 먼저 검증하고, neural controller는 trusted executor 밖에서 순위만 학습한다.

## 다음 증거 Gate

1. 개발 중 열지 않는 sealed transfer split을 별도 manifest와 digest로 고정한다.
2. 공개 소스에서 adapter 입력을 만들고 source license와 SHA-256 lineage를 보존한다.
3. 언어·수학·비전 candidate-level human gold를 train/validation/sealed test로 분리한다.
4. 20개 이상 seed의 평균, 중앙값, 신뢰구간을 기록한다.
5. candidate grounding뿐 아니라 최종 operator-search solve rate와 CPU/RSS를 함께 측정한다.

## 검증

```powershell
python -m pytest tests/test_typed_operator_grounding_learning.py -q
python -m pytest tests/test_typed_operator_semantic_grounding.py -q
python -m pytest tests/test_typed_operator_semantic_grounding_curriculum.py -q
python -m pytest -k typed_operator
python -m pytest
```
