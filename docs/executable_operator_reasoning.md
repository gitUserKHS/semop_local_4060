# Executable Operator Reasoning

## 판정

“기존 Transformer의 의미 처리 능력을 유지하면서 자유형 CoT 대신 연산자 프로그램을
생성하게 하자”는 제안은 SemOp의 주 방향으로 채택한다. 단, 연산자 이름이 붙은 설명문은
증명이 아니다. 연산자가 실제 typed state를 읽고 바꾸며, 외부 실행기와 proof replay가 그
변화를 다시 확인할 때만 실행 가능한 추론으로 인정한다.

```text
raw input
  -> semantic producer
  -> typed proposal
  -> parser + type checker
  -> operator search/controller
  -> executor
  -> external/domain verifier
  -> proof replay
  -> natural-language rendering
```

## 신뢰 경계

연산자를 단순히 “결정적”과 “신경망” 두 종류로만 나누지 않는다. 실행 권한에 따라 세
계층으로 나눈다.

| 계층 | 예 | 사실 추가 권한 |
|---|---|---|
| verified primitive | 계산, 정렬, 컴파일, schema 검사 | 검증된 관측 또는 derived fact |
| proposal operator | 분해, 의미 비교, 가설, 객체 후보 | `proposed`만 가능 |
| renderer | 한국어 설명, 코드 주석, 대화 스타일 | world state 변경 불가 |

모델 confidence가 높아도 proposal은 observed fact가 아니다. 결정적 verifier, 독립된 외부
도구, digest에 묶인 사람 검토 중 하나가 승인해야 증명 전제가 된다. 작은 controller도 적용
가능한 action의 순위만 매기며 halt 요청은 목표가 이미 증명된 경우에만 수용한다.

## 연산자 크기

`DECOMPOSE`, `INFER`, `VERIFY` 같은 이름은 controller가 공유할 **family**로는 유용하지만,
그 자체를 거대한 만능 함수로 구현하면 연산자 하나가 다시 LLM 전체가 된다. 실행 registry에는
입출력 타입, 전제, 효과, guard가 구체적인 schema만 등록한다.

```text
family: VERIFY
schema: accept_compiled_tested_program(
  CodingProblem, Program, Algorithm
) -> VERIFIED_SOLUTION
```

따라서 MVP는 10~12개 family와 여러 작은 domain schema로 구성한다. 새 schema는 positive,
near-miss, contradiction, proof replay fixture를 가져야 한다.

## 저자원 학습 순서

1. **결정론적 기준선**: 모델 없이 BFS/A*와 verifier로 정답 trace를 만든다.
2. **희소 controller**: typed 구조만 보고 action 순위를 학습한다. 답이나 fact를 생성하지 않는다.
3. **작은 semantic parser**: 필요할 때 0.5B~3B 모델이 typed proposal만 생성한다.
4. **rejection/SFT**: 파서·타입·실행·검증을 통과한 복합 trace만 학습한다.
5. **bounded RL**: 보상 해킹 검사가 끝난 verifier가 있을 때만 새 조합 탐색에 사용한다.

그래서 로컬 encoder나 1~3B 가중치를 먼저 다운로드하지 않는다. 규칙 기반 grounding의 실제
실패율과 sparse controller 대비 이득을 측정한 뒤, 그 병목을 줄이는 가장 작은 모델을 고른다.

## 필수 실험

비교군은 `직접 응답`, `자유형 CoT`, `연산자 라벨`, `실행형 연산자`, `실행형+controller`로
분리한다. 무작위 문장 분할 대신 다음 외삽을 측정한다.

- 훈련에 없던 **타입상 유효한** operator composition
- 훈련 깊이 2~3, 평가 깊이 4~7의 길이 외삽
- 새 이름과 새 인자 조합
- 타입이나 전제가 깨진 순열 hard negative
- 중간 state를 바꿨을 때 결론도 바뀌는 causal intervention

지표도 분리한다.

- `primitive_replay_integrity`: 실행 trace가 재생되는가
- `semantic_correctness`: 사람 gold와 실제 의미가 맞는가
- `verified_solve_rate`: 검증된 목표 완성률
- expansion, CPU p50/p95, peak RSS, artifact 크기
- false accept와 abstention risk-coverage

## 현재 구현과 다음 병목

현재 SemOp은 typed facts, proposal 권한, operator kernel, verifier, proof replay, sparse/recurrent
controller 경계를 이미 갖고 있다. 코딩도 `CANDIDATE_PROGRAM -> COMPILES -> TESTS_PASSED ->
VERIFIED_SOLUTION` 경로로 같은 runtime에 들어왔다. 따라서 다음 큰 과제는 새 실행기를 또
만드는 것이 아니라 자유형 채팅 입력을 typed proposal로 바꾸는 작은 semantic parser와,
코딩을 포함한 구조 holdout controller 전이 평가다.

현재 네 도메인 계약 게이트는 다음과 같다.

```powershell
python tools/eval/evaluate_operator_core_gate.py --require-pass
```

이 게이트는 일반 채팅, 자유형 코딩, 고등수학, 자연 이미지 이해를 입증하지 않는다.

## 연구 근거

- [Plan-and-Solve](https://arxiv.org/abs/2305.04091): 먼저 계획하고 하위 문제를 실행하는 구조의 이점
- [Unfaithful CoT](https://arxiv.org/abs/2305.04388): 자연어 설명이 실제 원인과 다를 수 있음
- [Program of Thoughts](https://arxiv.org/abs/2211.12588): 계산을 외부 프로그램 실행으로 분리
- [PAL](https://proceedings.mlr.press/v202/gao23f.html): 언어 분해와 interpreter 실행의 결합
- [Breaking the Chain](https://arxiv.org/abs/2603.16475): 중간 구조 intervention과 외부 derivation의 필요성
- [Reusable Modules](https://arxiv.org/abs/2606.18089): compound trace의 모듈 재조합 가설
- [Tool-Integrated Reasoning](https://arxiv.org/abs/2508.19201): 외부 도구가 가능한 전략 공간을 넓힌다는 분석

마지막 세 논문은 비교적 최근의 preprint이므로 확정된 법칙이 아니라 실험 설계 가설로만
사용한다.
