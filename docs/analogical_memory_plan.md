# Analogical Memory Plan

## Why This Matters

사람은 새로운 상황을 볼 때 보통 한 개의 정답 규칙만 적용하지 않는다.
먼저 `이거 전에 봤던 것과 비슷한데?`를 떠올리고,
그 다음에는 여러 개의 비슷한 사례를 동시에 비교하면서
- 무엇이 같은지
- 무엇이 다른지
- 어디서 실패했는지
- 어떤 우회가 가능했는지
를 정리한다.

SemOp도 이 능력이 필요하다.
단순 top-1 retrieval로는 충분하지 않다.
연산자 중심 시스템에서는 `표면 문장 유사도`보다
`숨은 목표`, `필수 전제`, `실패 패턴`, `연산자 계열`이 얼마나 겹치는지가 더 중요하다.

## Current Problem

기존 메모리 경로는 이미 존재한다.
- `src/semop/memory_retrieval.py`
- `src/semop/corpus_store.py`
- `src/semop/analogy_policy.py`
- `src/semop/pipeline.py`

하지만 기존 상태의 약점은 다음과 같았다.
- retrieval이 top-k 유사 쿼리를 가져오더라도 왜 비슷한지 구조적으로 설명하지 못한다.
- 비슷한 사례가 여러 개 잡혀도 거의 같은 패턴만 반복될 수 있다.
- 응답에 `연상된 사례`가 노출되지 않아 사용자가 메모리 기반 비교를 보기 어렵다.
- memory prior는 operator 가족 승격에는 쓰이지만, 다중 유사사례 비교 자체는 별도 객체로 다뤄지지 않았다.

## Implemented Direction

새로운 계층은 `analogical memory`다.

핵심 규칙:
1. 현재 질의를 구조 그래프로 바꾼다.
2. 메모리에서 여러 유사 그래프를 가져온다.
3. 각 후보에 대해 다음 축의 겹침을 계산한다.
   - hidden goals
   - required premises
   - missing requirements
   - relation patterns
   - operator families
   - concept/node overlap
4. 가장 점수가 높은 사례만 고르지 않고,
   같은 시그니처가 반복되면 일부를 버려서 다양한 유사사례를 남긴다.
5. 선택된 사례를 `analogical_matches`로 그래프에 저장한다.
6. 응답에서 `연상된 사례` 섹션으로 보여 준다.

현재 구현 위치:
- `src/semop/memory_analogies.py`
- `src/semop/structures.py`
- `src/semop/pipeline.py`
- `src/semop/response_synthesizer.py`
- `src/semop/corpus_store.py`

현재 바로 적용된 개선:
- retrieval은 이제 질의 문자열만 보지 않고 `heuristic probe graph`를 먼저 만든 뒤 구조 겹침으로 재정렬한다.
- `retrieve_item_from_drawer_goal`, `retrieve_item_from_box_goal`, `retrieve_item_from_pouch_goal` 같은 사례는 공통 `goal pattern`으로 묶어서 비교한다.
- 연상 사례 선택은 같은 query 중복만 제거하고, 실제로 여러 개의 비슷한 사례를 남기도록 완화했다.
- plan 앞단에 `analogy_requirement_guard`와 `analogy_compare_cases`를 넣어, 공통 전제를 먼저 확인하고 표면 차이와 구조 차이를 분리하게 했다.
- verifier는 연상 사례와 겹치는 전제와 실패 패턴이 있으면 premise validation, goal-preservation check, operator runtime warning을 함께 강화한다.
- analogy policy는 메모리 그래프 쌍에서 retrieval/verification/operator-priority 가중치를 학습하고, 파이프라인이 그 정책 파일을 읽어 탐색 순위를 재정렬한다.
- operator runtime에는 composition verifier가 붙어서 higher operator가 필요한 basis를 실제로 닫았는지 compiler finding으로 남긴다.

## Data Shape

`StructuredMeaningGraph`는 이제 `analogical_matches`를 가진다.
각 항목은 다음을 담는다.
- `query`
- `score`
- `analogy_type`
- `shared_basis`
- `shared_nodes`
- `shared_goals`
- `shared_requirements`
- `shared_operator_families`

이렇게 해야 나중에 teacher trace, evaluation, GUI에서도 같은 객체를 재사용할 수 있다.

## Analogy Types

현재는 다음 정도로 시작한다.
- `goal_premise_analogy`
  - 숨은 목표와 필수 전제가 함께 겹친 경우
- `failure_analogy`
  - 빠진 요구조건이나 실패 상태가 겹친 경우
- `operator_analogy`
  - 연산자 family가 겹친 경우
- `relation_analogy`
  - relation pattern이 겹친 경우
- `surface_analogy`
  - 약한 표면 유사만 있는 경우

이 분류는 아직 휴리스틱이지만, 중요한 점은 `무엇이 비슷한지`를 명시적으로 남긴다는 것이다.

## Near-Term Engineering Plan

### Step 1. Structural Analogy Retrieval
- 현재처럼 query-level retrieval을 유지하되, retrieval 후 구조 겹침 점수로 재정렬한다.
- 최소 2~3개의 서로 다른 analogy signature를 유지한다.
- top-1이 아니라 `diverse top-k analogies`를 기본 정책으로 둔다.

### Step 2. Analogy-Aware Planning
- 연상된 사례 중 `failure_analogy`가 강하면 plan 앞부분에 예방 단계가 더 강하게 들어가게 한다.
- `goal_premise_analogy`가 강하면 숨은 전제 신뢰도를 높인다.
- `operator_analogy`가 강하면 operator family prior에 가중치를 더 준다.

### Step 3. Analogy-Aware Verification
- 현재 후보 행동이 과거 실패 사례와 같은 missing requirement 구조를 가지면 verifier 경고를 올린다.
- 반대로 과거 성공 사례와 같은 satisfied structure를 가지면 실행 가능성 점수를 올린다.

### Step 4. Analogy Benchmark
- 평가셋에 `single best memory hit`가 아니라 `multi-analogy usefulness`를 넣는다.
- 지표 예시:
  - analogy recall
  - diverse analogy count
  - failure analogy precision
  - analogy-conditioned plan improvement

## Longer-Term Plan

1. analogy retrieval을 단순 query embedding이 아니라 graph embedding 또는 operator-signature embedding으로 확장한다.
2. VLSO에도 같은 틀을 적용해서 `이 가방 구조가 전에 본 파우치/서랍과 비슷하다` 수준의 교차모달 연상을 만든다.
3. CP에서는 `이 문제 구조가 전에 본 range query / prefix / offline processing 패턴과 비슷하다`를 같은 객체로 다룬다.
4. retained operator와 analogy memory를 연결해서 `비슷한 사례 집합 -> 상위 operator proposal` 경로를 강화한다.

## Engineering Rule

새로운 analogy 기능은 단순 추천 기능이 아니다.
반드시 아래 중 하나로 연결되어야 한다.
- hidden premise recovery 강화
- operator prior 강화
- verifier 경고 강화
- transfer/operator proposal 강화

그렇지 않으면 이 프로젝트의 operator-intelligence 목표와 맞지 않는다.




## 2026-03-13 Integration Update
- `src/semop/unified_parser.py`의 learned unified parser가 `StructuredMeaningPipeline` 초기화 단계에 직접 연결되어 query + source_context 토큰에서 intent/domain/goal/operator-family priors를 준다.
- `src/semop/pipeline.py`는 이제 `visual_input`과 `source_context`를 같은 `StructuredMeaningGraph`로 병합해서, 이미지도 언어와 동일한 analogy/planning/compiler/context-frame 루프를 지난다.
- `src/semop/retained_operator_algebra.py`가 반복적으로 검증된 decomposition/functor만 retained record로 남기고, 파이프라인은 이 retained algebra를 실행 시 higher operator prior로 재주입한다.
- `src/semop/operator_runtime.py`의 compiler verifier는 basis closure뿐 아니라 input/output type support, functor consistency, illegal basis usage, counterexample-based repair 힌트를 함께 남긴다.
- `src/semop/unified_benchmark.py`는 unseen transfer, analogy usefulness, compiler validity, grounded explanation fidelity, repair success rate를 하나의 benchmark summary로 묶는다.
