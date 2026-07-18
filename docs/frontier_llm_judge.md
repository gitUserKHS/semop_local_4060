# Frontier LLM Teacher/Judge Boundary

## 결론

초기 SemOp은 프론티어 LLM을 `teacher`와 의미 `judge`로 사용할 수 있다. 이는 적은
데이터로 typed operator vocabulary, 후보 전제, 후보 프로그램을 빠르게 수집하는 데
유용하다. 그러나 LLM의 문장이나 confidence는 증명이 아니다. LLM은 아래 항목을
직접 확정할 권한이 없다.

- `observed`, `assumed`, `derived` 사실 추가
- verifier goal의 성공 처리
- operator guard 우회
- proof replay 생략
- 자기 출력을 정답 trace로 바로 저장

핵심 규칙은 한 줄이다.

```text
LLM proposes; typed operators execute; deterministic verifiers decide.
```

## 허용 역할

프론티어 LLM에는 다음 작업을 맡길 수 있다.

- 자유 문장을 기존 type, predicate, operator 후보로 grounding
- 숨은 전제와 반례 후보 제안
- 현재 state와 goal에 맞는 bounded operator program 제안
- 여러 symbolic proof 후보의 의미 적합성 비교
- 실패 trace의 오류 분류와 다음 탐색 방향 제안
- 사람이 검토할 사례의 우선순위 지정

LLM의 긍정 판정은 `JudgeDecision(SUPPORTS, ...)`로 기록한다. 사실 후보는
`stage_judged_fact()`를 거치며 항상 proof-ineligible `FactStatus.PROPOSED`가 된다.
독립적인 도메인 verifier가 해당 atom을 직접 검사한 경우에만
`promote_judged_fact()`가 `OBSERVED`로 승격한다.

이 API는 이제 공통 `GroundingCandidate -> GroundingDecision -> GroundingRecord`
경계를 재사용한다. 따라서 frontier judge, 작은 local grounder, 언어 heuristic과 비전
proposal이 서로 다른 우회 경로로 observed fact를 만들 수 없다.

## 프로그램 검증 경로

LLM이 operator program을 제안한 경우에는 다음 경계를 통과한다.

```text
frontier LLM
  -> JudgeDecision + typed GroundAction 후보
  -> type / arity / registered schema 검사
  -> precondition / guard / effect 재계산
  -> 초기 WorldState부터 proof replay
  -> 모든 verifier goal 확인
  -> 불필요한 step 제거 검사
  -> replay-verified SolveResult
  -> TraceCorpus
  -> tiny controller 학습
```

`verify_judged_program()`은 기본 6-step 제한 안에서 프로그램을 materialize하고,
각 action이 새 fact를 만들며 모든 goal을 증명하는지 확인한다. 기본 설정에서는 한
step씩 제거해도 성공하는 중복 프로그램을 거절한다.

`teacher_review_to_solve_result()`는 이미 `accepted=True`인 객체도 그대로 믿지 않는다.
호출 시 받은 registry, 초기 state, goal에 대해 action과 proof를 다시 검증한다. 따라서
다른 문제에서 얻은 review를 재사용하거나 final state를 바꾸어 학습 corpus에 넣을 수
없다. 성공 결과만 `TraceCorpus.add_result(source="verifier")`에 들어간다.

## 감사 정보

각 판정은 적어도 다음 정보를 남긴다.

- 고정된 `candidate_id`와 domain
- 정확한 `model_id`
- prompt 내용 대신 재현 가능한 `prompt_fingerprint`
- `supports`, `rejects`, `abstains` verdict와 confidence
- evidence reference
- typed execution과 proof replay 통과 여부

`teacher_review_metadata()`가 corpus용 표준 metadata를 만든다. rationale 전체는
민감 정보나 불필요한 장문을 저장할 수 있어 기본 metadata에 넣지 않는다.

## 운영 원칙

1. judge가 모르면 `ABSTAINS`를 반환하게 한다.
2. 같은 모델이 만든 답을 같은 prompt로 자기 채점한 점수만 사용하지 않는다.
3. held-out benchmark의 정답이나 테스트 전용 operator를 teacher prompt에 넣지 않는다.
4. model과 prompt 버전이 바뀌면 fingerprint와 데이터 lineage를 분리한다.
5. verifier가 없는 자유 의미 판단은 `PROPOSED`로 유지하거나 사람 검토로 보낸다.
6. 자동 생성 trace와 사람 검토 trace의 수를 별도로 보고한다.
7. 성공으로 보고한 proof의 soundness 목표는 계속 100%로 둔다.

## 현재 범위

현재 코드는 provider-neutral `SemanticJudge` protocol과 안전 경계를 구현한다. 특정
상용 API 호출, 비밀키 관리, 다중 judge 합의, 비용 제어는 아직 연결하지 않았다.
따라서 지금 단계는 "프론티어 LLM을 붙일 수 있는 검증된 소켓"이지, LLM이 스스로
진실을 확정하거나 완전 자율 학습하는 시스템은 아니다.

실행 가능한 로컬 경계 예제:

```powershell
python examples/typed_frontier_judge_demo.py
python -m pytest tests/test_typed_operator_judging.py -q
```
