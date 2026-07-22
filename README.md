# SemOp Local 4060

## 처음 실행하는 사람을 위한 30초 시작

코드나 typed DSL을 몰라도 된다. Windows에서는 저장소 루트의
`start_semop.bat`를 더블클릭하면 로컬 브라우저 화면이 열린다.

명령어로 실행하려면 다음 두 줄이면 된다.

```powershell
$env:PYTHONPATH = "src"
python -m semop.beginner_web
```

기본 화면은 도메인을 고를 필요 없는 통합 채팅이다. 질문이나 수학식을 적고 필요하면
이미지를 붙이면 SemOp이 먼저 결정론적 typed compiler를 시도한다. 기존 `코딩`, `언어 조건`,
`수학식`, `색상 비전` 폼은 `고급 직접 입력` 안에 그대로 남아 있다. 정확한 수식과 통제된
조건 문장, 작은 색상 이미지는 LLM 없이 실행되며 코딩 검증만 로컬 `g++`가 필요하다.
로컬 의미 모델이 설치되어 있으면 일반 설명·대화도 `best_effort`로 답하며, 같은 대화의
최근 8개 메시지(최대 6,000자)를 미검증 작업 문맥으로 참고한다. 전체 대화는 로컬 SQLite에
남아 서버를 다시 시작해도 브라우저에서 복원되고 `새 대화`로 문맥을 분리할 수 있다.
새 세션에서는 현재 질문과 비슷한 과거 exchange를 최대 2개 회상한다. 과거 assistant 답은
틀릴 수 있는 미검증 경험으로 표시되며 proof 사실이나 학습 정답으로 자동 승격되지 않는다.
`장기기억 관리`에는 사용자가 직접 짧은 정보를 저장하거나 삭제할 수 있다. 현재 질문과
단어·글자 조각이 겹치는 기억 최대 3개만 저비용으로 회상하며, 어떤 기억이 사용됐는지는
답변의 기술 정보에 표시한다. 저장된 기억도 개인화 문맥일 뿐 proof 사실은 아니다.
balanced/auto에서는 lexical 검색이 실패했을 때만 질문과 기억을 작은 typed facet으로 먼저
맞춘 뒤 multilingual-E5-small이 같은 facet 안의 후보를 재정렬한다. 현재 facet은 답변 길이,
답변 순서, GPU, 프로젝트 목표, 테스트 일정 다섯 가지이며 `semantic_e5` 출처를 남긴다.
채팅에 `기억해: 내용` 또는 `기억해줘: 내용`을 입력해도 모델을 거치지 않고 같은 로컬
장기기억에 저장된다. 다음 관련 질문부터 자동 회상하며 저장 ID와 write provenance가 남는다.
틀린 답 바로 다음에 `정정해: 올바른 답`을 입력하면 직전 질문·답·교정을 SQLite에 결속한다.
같은 텍스트 질문에는 교정 답을 one-shot으로 재사용하고 correction ID를 남기되, 사용자 제공
교정이므로 `best_effort`를 넘거나 typed proof 사실로 승격되지는 않는다.
같은 교정 답이 서로 다른 질문 표현 3개에서 반복되면 model-free correction prototype을
파생해 비슷한 새 표현에도 재사용한다. 두 예시에서는 일반화하지 않으며 support·similarity와
근거 correction ID를 provenance에 기록한다.
`장기 작업 관리`에서는 여러 날 이어갈 작업의 완료 조건, 진행 상황, 결정 사항, 다음 행동을
revision별로 저장한다. 사용자가 고르거나 이름으로 고유하게 검색된 작업 하나만 현재 채팅의 미검증 문맥으로 전달되며,
완료한 작업은 마지막 revision에서 고정된다. 모델이 대화를 몰래 요약해 작업 상태를 바꾸지는
않는다.
활성 작업이 하나면 `장기 작업 계속하자`, 여러 개면 `<작업 이름> 작업 계속하자`라고 입력해
클릭 없이 최신 revision을 재개할 수 있다. 여러 작업이 모호하게 맞으면 임의로 선택하지 않는다.
작업을 고른 뒤에는 `어디까지 했지?`, `결정이 뭐였지?`, `다음 행동은?`처럼 짧게 물어도
최신 revision의 진행·결정·다음 행동을 모델 없이 바로 조회한다. 작업 이름을 질문에 넣으면
브라우저 재시작 뒤에도 해당 작업을 먼저 찾아 연결한다.
선택된 작업에는 `작업 기록: 진행 내용 | 다음: 할 일 | 결정: 결정`으로 채팅에서 새 checkpoint를
append할 수 있다. 모델이 요약한 내용이 아니라 사용자가 명시한 필드만 저장한다.
`작업 만들기: 이름 | 완료 조건: 결과`와 `작업 완료: 최종 결과`도 지원하므로 생성부터 immutable
완료까지 장기 작업 전체 생명주기를 채팅만으로 실행할 수 있다.
미증명·파싱 실패는 기본적으로 로컬 검토 큐에만 저장된다. 화면에서 사용자가 정확한
입력과 기대 결과를 직접 승인한 사례만 검증 학습에 들어가며, 새 typed 규칙은 네 분할
회귀 gate와 proof replay를 모두 통과한 경우에만 해시 확인 가능한 파일로 활성화된다.
모델 해석 검토함은 pending trace를 최신순으로 쏟아내지 않고, proof/이미지 근거의 검토
가능성, 실패 교정 가치, 기존 검토와의 표현 차이, 부족한 도메인·학습 역할을 사용해 상위
8개 replay 후보와 선정 이유만 보여 준다. 이 선택은 읽기 전용이며 review나 모델을 자동으로
바꾸지 않는다.
같은 typed payload로 사람이 승인한 서로 다른 텍스트 표현이 3개 이상 모이면
`ReviewedSemanticMemory`가 digest-bound semantic prototype을 만든다. 유사한 새 문장에는
이 prototype을 analogy로 제공하지만 모델 출력은 계속 `PROPOSED/best_effort`이며 proof
권한은 늘지 않는다. 수학 질문에 명시된 더하기·빼기·곱하기·나누기 단서가 prototype의
expression과 충돌하면 해당 hint를 사용하지 않는다.
`tools/eval/evaluate_memory_continuity.py`는 모델 없이 100턴 대화 복원, bounded 작업기억,
작업기억 밖 같은 세션 및 cross-session episode 회상, explicit-memory recall과 장기 작업
revision을 약 0.48초에
검사한다. `--semantic-tier balanced`를
붙인 실제 Qwen A/B에서는 기억 없이는 코드명을 모른다고 답하고, 기억이 있을 때 저장된
`별빛-27`을 정확히 반환했다. 최적화 전 기억 없음 Qwen은 81 tokens와 46.35초가 걸렸다.
현재는 기억이 없으면 `ASK(missing memory)`, 있으면 explicit-memory `RETRIEVE → SELECT`로
처리해 각각 생성 0 tokens와 약 0.139ms/0.383ms다. 이 검사는 기억 효과를 답변 생성과
분리해 반복 측정한다.
episodic 후보 검색은 SQLite FTS5와 기존 lexical reranker를 결합해 최신 400개가 아니라
전체 저장 이력을 대상으로 한다. `--long-history-exchanges 5000` 실측에서는 5,000개를 모두
인덱싱했고 기존 turn backfill 약 101.5ms, 오래된 cross/same-session 조회 약 20.6/24.0ms,
DB 약 5.01MB였다. 검색 범위만 넓어지며 과거 답의 `best_effort` 권한은 그대로다.
별도 `--semantic-paraphrase-ab` 실측에서는 lexical `0/5`였던 한국어 바꿔 말하기가 typed
facet+E5에서 `5/5`, 무관 질문 오회상 `0/3`이었다. 첫 import/load 포함 5건은 약 12.94초,
같은 5건 warm replay는 약 70.5ms였다. 이는 다섯 facet의 작은 smoke이지 자유 의미 검색
전체의 정확도 주장이 아니다.
최적화 전 같은 모델의 cross-session episode A/B에서는 문맥 없는 답
`hippocampus_recall`이 과거 exchange를 제공하자 `해마-7319`로 바뀌었다. 현재의 고관련
직접 회상은 같은 `best_effort` 경계를 유지하면서 모델 대신 episode retriever를 사용한다.
긴 단일 세션 A/B에서도 최근 8개 메시지 밖으로 밀려난 score `0.590461` episode를 제공하자
추측 답 `SemOp`이 저장된 `등대-4821`로 바뀌었다. 계측 결과 99/131 tokens, repair 0회,
token limit 미도달 상태에서도 Qwen 생성이 50.12초/45.66초 걸렸다. 이후 score `0.55` 이상의
명시적 회상 질문은 `RETRIEVE → SELECT`, 기억이 없으면 `ASK`로 처리한다. 현재 long-session
없음/있음 경로는 모두 생성 0 tokens와 약 0.120ms/0.092ms다.
같은 평가의 `task_ab`는 최신 checkpoint를 직접 `RETRIEVE → SELECT`한다. 실제
revision 2/3 충돌에서는 작업 미선택 안내 약 0.162ms, 최신 `검증-단계-42` 반환 약
0.131ms, 진행·결정·다음 행동 요약 약 0.073ms였다. 이전 행동은 제외됐고 세 경로 모두
Qwen을 호출하지 않았다.
`task_auto_resume`는 두 활성 작업의 모호한 요청을 거절하고, 이름이 포함된 요청에서 score와
선택 근거를 남기며 목표 작업의 최신 revision만 자동 연결한다.
검증된 sparse controller checkpoint가 있으면 같은 작은 정책 인터페이스가 네 도메인의 연산자
순서만 안내하고, 없으면 자동으로 결정론적 탐색을 사용한다. 어느 쪽이든 결론은 typed
executor와 proof replay가 결정한다. `--no-experience`, `--no-learned-rules`,
`--no-controller`로 각 기능을 독립적으로 끌 수 있다.

현재 쉬운 화면의 **검증 범위**는 등록된 C++ 알고리즘 문제, 통제된 목표·필요조건 문장,
정확한 계산식·일차방정식·한 변수 이차방정식의 실수해, 작은 색상 격자다. 일반 자유 대화는
가능하지만 정답 보장 대상이
아니며, 임의 코딩 문제와 자연 사진 이해도 검증된 능력으로 오해하지 않도록 각 결과에
검증 범위와 현실 증거 여부를 함께 표시한다. 자세한 그림 설명은
[한국어 첫걸음 가이드](docs/beginner_guide_ko.md)에 있다.

## Prompt-first Operator Intelligence v1

```powershell
$env:PYTHONPATH = "src"
python -m semop.chat_cli "수학: 어떤 수에 2를 더하면 5야" --tier symbolic --show-proof
python -m semop.chat_cli "수학: x² - 5x + 6 = 0" --tier symbolic --show-proof
python -m semop.model_manager doctor
```

공개 Python API는 다음처럼 사용한다.

```python
from semop import PromptRequest, SemOpAssistant

answer = SemOpAssistant().solve(PromptRequest("수학: (2 + 3) * 4"))
print(answer.status.value)  # verified
print(answer.answer)        # 검증된 계산 결과는 20이야.
```

`AnswerEnvelope.status`는 `verified`, `conditional`, `best_effort`, `unsupported` 중 하나다.
Qwen 같은 의미 모델이 만든 해석은 typed 실행이 성공해도 `best_effort`를 넘지 않는다.
모델 출력은 언제나 `SemanticProposal(disposition="proposed")`이며 원문 의미와 proof replay를
서로 다른 지표로 기록한다. 자세한 흐름은
[prompt-first 아키텍처](docs/prompt_first_operator_intelligence.md), 모델 설치는
[로컬 모델 가이드](docs/local_semantic_models.md), 승격 절차는
[자가 학습 승인 경계](docs/self_learning_activation.md)를 참고한다.

`economy` 등급은 승인되지 않은 0.8B base를 대신 실행하지 않는다. 봉인 평가와 명시적 사람
승인을 통과한 digest-bound LoRA가 없으면 오류로 닫히며, `semop-promote`로 stage, approve,
status와 rollback을 수행한다.

`balanced`에서 Qwen이 실제로 사용된 답은 같은 로컬 SQLite 파일에 의미 trace로 남는다.
쉬운 화면의 `해석이 맞아/틀려`에서 직접 확인한 뒤에만 학생 모델 corpus로 export된다.
지나간 trace는 `모델 해석 검토함`에서 다시 확인할 수 있다. PNG/JPEG는 exact bytes와
SHA-256 manifest를 로컬 SQLite에 보존해 다시 보여 주며, 비전 승인은 semantic label일 뿐
proof 성공이나 text LoRA positive로 계산되지 않는다. exact 동일 이미지의 다음 요청에는
최신 승인·거절을 Qwen의 untrusted analogy로 다시 제공한다. 언어·수학도 정규화 후 exact
동일 문장에 최신 검토를 재사용한다. 어느 경우든 출력은 계속 `PROPOSED`다.
대화 문맥을 사용한 답은 현재 문맥을 떼어 내면 같은 학습 표적이 아니므로 의미 증류
journal에 넣지 않는다. 대화 기록은 자동으로 장기 지식이나 모델 가중치로 승격되지 않으며,
명시적 장기기억과 장기 작업 checkpoint도 사용자가 직접 추가·수정·완료한 내용만 보존한다.

```powershell
semop-student-cycle status
semop-student-cycle run --confirm --device cuda
```

첫 명령은 모델을 불러오지 않고 사람이 검토한 `train`, `validation`, `sealed` 역할별 수와
학습/봉인 세트 중복을 보여 준다. 두 번째 명령만 GPU 학습을 시작한다. 준비된 train/sealed
positive가 없으면 모델 로드 전에 `not_ready`로 끝나며, 통과한 LoRA만 digest-bound candidate로
stage한다. 봉인 gate 실패 후보는 stage하지 않고, 통과 후보도 자동 활성화하지 않는다.
활성화에는 출력된 candidate ID를 사람이 리포트와 함께 확인한 뒤 별도의
`semop-promote --root artifacts/promotion/semantic-model approve <candidate-id> --reviewer <이름> --confirm`
이 필요하다.

텍스트 LoRA가 질문만 보고 비전 답을 외우지 않도록 image trace와 exact media는 journal에
보존하지만 현재 text export에서는 제외한다. 학습 결과는 candidate일 뿐이며 sealed 평가와
`semop-promote` 승인을 통과하기 전에는 `economy` 런타임에 들어가지 않는다.
학습기는 `train` record만, 학생 평가기는 `sealed` record만 허용해 같은 trace를 학습과
최종 gate에 재사용하지 못하게 한다.

2026-07-22 RTX 4060 smoke에서 Qwen3.5-2B 수학 후보는 exact replay까지 이어졌고,
자연 이미지 경로는 첫 요청 약 25.3초, 같은 입력의 warm 요청 약 8.04초와 GPU 메모리
약 4.54GB를 기록했다. 이미지 의미는 `best_effort`로 남았다. Florence-only fallback은
실행됐지만 합성 글자 이미지에서 OCR box를 찾지 못했으므로 정확도 성과로 세지 않는다.
일반 채팅은 초기 cold 약 18.8초, 같은 프로세스의 문맥 포함 후속 요청 약 4.82초였고 후속 축약
지시를 반영했다. 정체성 기억 고정 후 cold 재실행은 약 21.3초와 GPU 메모리 약 4.55GB였다.
invalid typed language payload는 실행하지 않되 이미 생성된 자연어 답을
`best_effort`로 보존한다. 첫 smoke에서 프로젝트 이름을 임의의 약자로 푼 semantic 오류도
관찰했으므로 문맥 사용 여부와 의미 정확도를 분리하며, SemOp 자체의 정체성은 고정된 system
semantic memory로 제공한다. 수정 후 재실행에서는 임의의 약자 확장이 사라졌다.
일반 Python 비교 질문의 연속 smoke는 cold 34.09초, warm 20.78초와 peak VRAM 약
4.62GB를 기록했다. 같은 경로에서 깨진 JSON 때문에 관찰됐던 69.79초/56.32초와 repair
2회 병목은, 첫 모델 출력의 list-comprehension AST를 결정론적 `for + append` 비교 예시로
조립하면서 두 요청 모두 repair 0회로 줄었다. 이는 문법과 요청된 비교 구조를 확인한 것이며
코드를 실행하거나 설명의 의미 정확성을 증명한 결과는 아니다. 일반 코딩 도움은 이 경계에
머물고, 현재 자동 컴파일·테스트 증명은 명시적인 C++17 알고리즘 문제에만 적용한다.
0.8B 원본은 같은 산술 입력을 잘못 분류해 승인 없는 economy fallback을 금지하는 경계를
실제로 확인했다. 이 실패만 겨냥한 48개 positive LoRA 개발 학습은 서로 다른 문구 계열의
12개 heldout에서 원본 `8/12`를 후보 `12/12`로 높이고 false acceptance 0건을 유지했다.
하지만 합성 generator-known semantics만 사용했으므로 후보는 활성화하지 않았고
`promotion_eligible: false`인 사람 검토 대기 artifact로만 남겼다. 세부 조건과 해석은
[로컬 모델 가이드](docs/local_semantic_models.md)에 있다.

SemOp Local 4060 is a local prototype for prompt-first operator intelligence on an ordinary PC.

The current product direction is a local general assistant, without claiming frontier-LLM parity. It:
- uses a small local semantic model for ordinary conversation and marks those answers `best_effort`
- routes exact language, mathematics, vision, and coding tasks through typed executors when available
- persists conversation episodes while keeping recalled context outside the proof boundary
- consolidates only reviewed or replay-verified traces into semantic and procedural learning candidates
- leaves an audit trace that separates semantic correctness from executor replay integrity

At the research level, the longer-term target is broader: learn how logical words such as `has`, `is`, `requires`, `if`, `before`, and `can` bind to concept frames, then reuse those learned grammar priors during reasoning across domain QA, math, olympiad proof search, competitive programming, and VLSO world-model reasoning.

The architectural target is a logical-operator-based intelligence system organized around four axes: operator learning, world-model construction, reusable memory, and verifier loops.
The central hypothesis is combinatorial: a small shared controller should learn to assemble many typed operator programs, while language, mathematics, and vision enter through domain adapters and every claimed result remains executor-verifiable.
New controller training now defaults to a `typed_structure` feature profile. It removes operator, predicate, function, and domain-tag identities while preserving nominal types, domain-neutral operator families, graph roles, and goal compatibility. A dependency-free leave-one-domain-out gate compares it with the legacy `full` profile before larger recurrent training.
The resource doctrine is CPU-first and sample-efficient: the symbolic core should run offline on an ordinary PC, while small local models and RTX 4060-class GPUs remain optional parsing, perception, and training accelerators.
A verifier-first typed operator core now runs beside the legacy runtime. It turns domain inputs into immutable typed facts, uses goal-relevant monotonic agenda chaining instead of enumerating fact subsets, slices the first supporting operator DAG, and replays every successful proof before returning it. The pipeline defaults to `shadow`: legacy output remains user-facing while typed proof, timing, allocation, and expansion measurements are written to the audit trace.
A shared typed grounding boundary now records language, math, and vision inputs as candidate, authority decision, fact, and immutable trace. Neural and heuristic producers can only propose; deterministic verifiers and explicit human reviews create the accept/reject examples used by later grounding-policy learning.
A dependency-free sparse grounding head now learns `ACCEPT/REJECT/ABSTAIN` from those independently verified examples. It shares typed and sensor-contract features across language, math, and vision, forces abstention on unseen sensor contracts, replays old labels during online updates, and promotes a new generation only after an untouched validation gate. Even an accepted prediction remains `PROPOSED` until a separate verifier or human promotes it.
A candidate-level semantic bridge now replaces the artificial sensor signal with controlled raw requirement text, exact expressions, and RGB rasters passed through the production adapters. Exact candidate reviews bind the case, typed atom, label, and human attestation to one digest. In the fixed repeated-template run, 20 labels per domain reach 91.7% completion with 344 parameters and zero false accepts; this is not a structural holdout result. A newer structural ablation reaches 60.0% completion at 20 labels and 72.5% at 100 with the full profile. Removing surface text changes neither result, while removing only the shared target-relative support margin already causes vision false accepts and fail-closed rollback. Human semantic gold remains at zero and is reported as `not_evaluated`.
A dependency-free raster adapter now adds a narrow real-pixel path: it detects small color components, verifies exact bounding-box or touching relations, keeps centroid-only guesses as `proposed`, and sends the resulting facts through the same operator proof replay.
A new hidden-premise layer now sits between surface parsing and later reasoning so the system can recover implicit goals and prerequisites before giving advice.
The current refactor direction is premise-first: candidate retrieval, premise proposal, and premise validation now precede later answer selection, CP code generation, and cross-modal alignment.
An operator-algebra layer now also records how higher operators decompose into simpler basis operators and stores category-style functor hypotheses for cross-modal alignment.
The current premise-first refactor is implemented end-to-end: hidden-premise candidate retrieval, proposal, validation, SQLite premise/operator memory, parser-first CP evaluation, and a shared evaluator snapshot are all wired into the codebase.
An operator self-evolution loop is now also present: repeated higher-operator decompositions can be mined into evolved operator proposals, utility-scored, retained, and tested on a small cross-domain transfer benchmark.
A new operator-proposal engine now sits in front of that loop: repeated decomposition and geometry/topology patterns are summarized into model-proposed higher operators, then normalized, merged, and passed to the verifier and transfer bench instead of being accepted directly.

## What You Can Run Today

### Typed Operator Core v1

```powershell
python -m pip install -r requirements-core.txt
python tools/eval/evaluate_low_resource_transfer.py --run-fast-tests
python tools/eval/evaluate_low_resource_transfer.py
python tools/eval/evaluate_low_resource_transfer.py --suite language-math-vision
python tools/eval/evaluate_low_resource_transfer.py --suite composed-v4
python tools/eval/evaluate_lmv_core_gate.py --require-pass
python tools/eval/evaluate_operator_core_gate.py --require-pass
python tools/eval/evaluate_controller_feature_transfer.py --require-pass
python tools/eval/evaluate_typed_self_learning.py
python tools/eval/evaluate_semantic_flow_self_learning.py
python tools/eval/evaluate_active_macro_learning.py
python tools/eval/evaluate_hierarchical_self_learning.py
python tools/eval/evaluate_raw_grounded_self_learning.py
python tools/eval/evaluate_semantic_benchmark.py
python tools/eval/evaluate_semantic_benchmark.py --gate-semantic-correctness 1.0 --gate-min-gold 3 --gate-min-gold-per-domain 1 --gate-domains language,math,vision
python tools/eval/evaluate_grounding_self_learning.py --checkpoint-root artifacts/grounding_self_learning --output artifacts/grounding_self_learning/report.json
python tools/eval/evaluate_semantic_grounding_learning.py --checkpoint-root artifacts/semantic_grounding_learning --output artifacts/semantic_grounding_learning/report.json
python tools/eval/evaluate_semantic_grounding_ablation.py --output artifacts/semantic_grounding_ablation_v1.json
python tools/eval/evaluate_semantic_grounding_curriculum.py --require-pass --output artifacts/semantic_grounding_curriculum.json
python tools/eval/review_semantic_grounding.py --case-id language-ready-two-requirements
python tools/eval/review_typed_experience.py --db artifacts/experience/typed-experience.db stats
python tools/eval/review_typed_experience.py --db artifacts/experience/typed-experience.db list --status pending
python tools/eval/run_lodo_controller_experiment.py --output-dir artifacts/lodo_debug --feature-profile typed_structure
python examples/typed_multidomain_demo.py
python examples/typed_compositional_v2_demo.py
python examples/typed_cross_domain_scene_demo.py
python examples/typed_frontier_judge_demo.py
python examples/typed_self_learning_demo.py --output artifacts/self_learning_run_01 --examples-per-structure 3
python examples/typed_self_discovery_demo.py --output artifacts/self_discovery_run_01 --examples-per-structure 1
python examples/typed_raw_self_learning_demo.py
python examples/typed_raster_vision_demo.py
```

`typed_multidomain_demo.py` sends a Korean premise sentence, an exact arithmetic
expression, and a verified vision scene through the same runtime and policy.
`typed_compositional_v2_demo.py` exercises Horn-style language inheritance, exact
linear/quadratic real equations, and raster shape/count/area operators without external runtime
dependencies.
`typed_cross_domain_scene_demo.py` runs one replayable vision -> exact comparison ->
language classification program instead of solving the three domains independently.
`typed_frontier_judge_demo.py` shows the frontier-LLM boundary: judge output remains
proposed until typed execution and proof replay admit the program to the trace corpus.
`typed_self_learning_demo.py` closes that loop: an active curriculum balances six
real language/math/vision adapter structures, replay-verified traces train a tiny
sparse action policy, and three entirely held-out capability compositions plus
negative controls gate promotion before an atomic hash-checked checkpoint is written.
`typed_self_discovery_demo.py` expands the pool without an LLM: verifier-backed
2/3-domain composition, bounded verified suffix scaffolds, and support-ablation
counterfactuals create new tasks before the same held-out promotion gate runs.
The typed kernel also exposes `VerifiedRuleDiscovery`, a separate bounded path that
anti-unifies repeated unsolved tasks into executable flat Horn-rule candidates.
Only rules that newly solve digest-recorded, human-reviewed validation and untouched
held-out positives while preserving matched negatives enter a hash-checked library;
activation remains explicit and every use is proof-replayed.
`VerifiedRuleLearningLoop` then combines those rules with the incumbent library and
uses a fourth final joint holdout to catch unsafe rule interactions. Promotion writes
an atomic hash-checked envelope containing the library and exact joint-review
certificate; rejection leaves the incumbent artifact untouched.
`TypedExperienceCollector` now connects this rule path to real language, math, and
raster-vision executions. Noteworthy runs enter an append-audited SQLite queue;
user or frontier-judge labels remain proposals until a separate digest-bound human
review. Approved cases are assigned by a precommitted four-way hash partition,
grounded again, checked for cross-split semantic leakage, and only then enter joint
rule promotion. A promoted library is activated explicitly through
`UnifiedTypedReasoner(augmenters=(library,))` and every derived result is replayed.
`evaluate_semantic_flow_self_learning.py` trains only on short verified
vision-to-math-to-language flows, then gates promotion on new phrasing, larger images,
reused measurements, deeper operator programs, and sound negative controls.
`evaluate_active_macro_learning.py` learns repeated primitive programs from verified
language, math, and raster-vision traces, validates them on a separate split, and uses
only promoted programs as search priors on held-out groundings. The kernel still
executes and replays every primitive step; a macro cannot inject a fact or bypass a
guard.
`evaluate_hierarchical_self_learning.py` trains and promotes a tiny shared-family
controller on language only, verifies zero-shot family transfer to math and vision,
and independently promotes procedural memory. The same evaluator supports sparse,
29K diagnostic recurrent, and full 5.84M recurrent profiles. It then evaluates their
combination on a final joint holdout that neither component used for selection.
Registry-specific macro activation and controller inference are packed into one
portable, hash-checked brain artifact.
`evaluate_raw_grounded_self_learning.py` removes the prebuilt-IR assumption from the
learning boundary. It grounds raw requirement text, exact math strings, and RGB pixel
problems through the production adapters, rejects failed or overlapping splits, and
trains on language only before evaluating untouched math and vision transfer.
`evaluate_semantic_benchmark.py` runs digest-bound language, math, and raster-vision
near misses through the same adapters. Seed labels remain `curated_unreviewed`; only
an exact, separately attested review can contribute to `semantic_correctness`.

Optional controller training:

```powershell
python -m pip install -r requirements-train.txt
python tools/train/train_tiny_controller.py --output artifacts/tiny_debug.npz --examples-per-domain 1 --epochs 1 --debug-small --feature-profile typed_structure
```

- `src/semop/kernel/`: immutable typed IR, operators, forward search, proof replay, adapters, traces, and verifier-gated MDL macro activation
- `src/semop/kernel/domain_catalog.py`: self-describing LMV adapter/codec/capability contract used by runtime and semantic artifacts
- `src/semop/kernel/experience.py`: audited raw input grounding, split fingerprints, bounded hard negatives, and the end-to-end raw self-learning API
- `src/semop/kernel/experience_queue.py`: append-audited LMV execution queue, digest-bound reviews, and deterministic four-way partitioning
- `src/semop/kernel/experience_collection.py`: production failure collection, reviewed-corpus grounding, leakage audit, and joint rule-learning bridge
- `src/semop/kernel/grounding.py`: shared LMV candidate authority, proposal promotion, hard-negative lineage, and verified learning examples
- `src/semop/kernel/semantic_grounding*.py`: exact candidate review/learning bridge, input-only operator features, and controlled semantic case generation
- `src/semop/tiny_controller/grounding_*.py`: anonymized LMV features, sparse selective policy, verified replay, online promotion, and hash-checked artifacts
- `src/semop/tiny_controller/`: sparse incumbent, 1.46M compact challenger, 5.84M ablation; NumPy inference and isolated PyTorch training
- `src/semop/tiny_controller/features.py`: full-vs-typed-structure canonical graph profiles that expose or hide domain identities deterministically
- `docs/typed_operator_core.md`: execution contract and extension workflow
- `docs/lmv_domain_catalog.md`: shared LMV boundary refactor, fast contract gate, research basis, and extension steps
- `docs/controller_feature_profiles.md`: identity-shortcut ablation, typed-structure contract, sparse LODO evidence, and artifact migration
- `docs/trust_provenance_and_metrics.md`: assertion/evidence/logical provenance, conditional proofs, honest metric names, and CI gates
- `docs/typed_grounding_boundary.md`: shared language/math/vision grounding trace, authority rules, review promotion, and research basis
- `docs/sparse_grounding_self_learning.md`: shared accept/reject/abstain policy, continual replay, risk-coverage gates, evaluation, and honest limits
- `docs/semantic_grounding_self_learning.md`: exact candidate reviews and real-adapter 0/5/20/100 semantic grounding evaluation
- `docs/semantic_data_acquisition.md`: pinned public sources, audited downloads, and generated/model-proposed data trust rules
- `docs/operator_boundary_curriculum.md`: verified synthetic/public/model data authority, typed decision features, feature-novel selection, and honest low-shot results
- `docs/lmv_semantic_benchmark.md`: digest-bound review workflow and authority-separated three-domain semantic evaluation
- `docs/language_math_vision_typed_runtime.md`: direct three-domain API, trust boundary, and current limits
- `docs/language_text_adapter.md`: high-precision Korean/English claims, proposed fallback, and contradiction handling
- `docs/typed_compositional_extensions.md`: v2 language logic, exact equations, raster quantification, and controller scoring contract
- `docs/composed_operator_runtime.md`: v4 registry composition, conjunctive scene conditions, operator frontier, and verified vision-math-language programs
- `docs/typed_dataflow.md`: reusable numeric measurement-to-condition-to-conclusion compiler and semantic-flow holdout
- `docs/frontier_llm_judge.md`: safe frontier-LLM teacher/judge roles and mandatory verifier/replay boundary
- `docs/verifier_gated_self_learning.md`: active three-domain curriculum, structural holdout promotion, rollback, and checkpoints
- `docs/raw_grounded_self_learning.md`: raw language/math/pixel grounding, leakage audit, failure handling, and measured self-learning transfer
- `docs/active_macro_learning.md`: primitive-expanded procedural memory, schema pinning, promotion gates, and rollback
- `docs/hierarchical_operator_brain.md`: shared controller plus procedural memory, independent split gates, and portable brain artifacts
- `docs/self_discovered_curriculum.md`: verifier-backed task composition, failure signals, counterfactual generation, lineage, and bounded self-discovery
- `docs/verified_rule_discovery.md`: bounded typed Horn induction, joint-library promotion, review provenance, held-out falsification, and atomic rollback
- `docs/online_verified_self_learning.md`: persistent LMV runtime experience, independent review, four-way split, rule promotion, and explicit activation
- `docs/raster_vision.md`: dependency-free raster input, pixel trust boundary, and learned-detector extension point
- `docs/tiny_controller.md`: architecture, losses, data limits, and artifact format
- `docs/low_resource_transfer_evaluation.md`: three-domain A/B benchmark and promotion gates
- `tools/eval/evaluate_typed_self_learning.py`: machine-readable structural-transfer, active-selection, promotion, and resource gates
- `tools/eval/evaluate_typed_task_discovery.py`: machine-readable task novelty, replay, depth extrapolation, and self-discovery transfer gates
- `tools/eval/evaluate_semantic_flow_self_learning.py`: machine-readable cross-domain semantic-flow transfer and resource gates
- `tools/eval/evaluate_active_macro_learning.py`: machine-readable three-domain macro induction, primitive replay, and resource gates
- `tools/eval/evaluate_hierarchical_self_learning.py`: machine-readable controller/macro/joint ablation and family-transfer gates
- `tools/eval/evaluate_raw_grounded_self_learning.py`: language-only raw training followed by untouched raw math/pixel transfer gates
- `tools/eval/evaluate_semantic_grounding_learning.py`: controlled raw LMV candidate learning, rollback, untouched test, and human-review audit
- `tools/eval/evaluate_semantic_grounding_ablation.py`: surface/margin shortcut audit against unseen compositions and raster structures
- `tools/eval/evaluate_semantic_grounding_curriculum.py`: prefix-vs-feature-novel low-resource A/B with fail-closed development extrapolation gates
- `tools/eval/evaluate_controller_feature_transfer.py`: dependency-free full-vs-typed-structure identity audit and LMV LODO replay gate
- `tools/eval/review_semantic_grounding.py`: inspect and attest one exact typed candidate label
- `tools/eval/review_typed_experience.py`: inspect, attest, review, and export persistent typed runtime experience
- `docs/lodo_controller_experiment.md`: leakage-controlled language/math/vision holdout training and evaluation

No trained controller artifact is committed yet. A local 1.46M compact
`typed_structure` candidate now trains from three raw language traces and reduces
untouched math and pixel expansion from 6 to 2 in each domain. Its fresh NumPy CPU
process adds about 20.5 MB peak RSS and the 5.42 MB artifact is loadable only when an
explicit all-pass evaluation report and SHA-256 match. A separate compact plus macro
split reduces 22 deterministic expansions to 10 with replay integrity 100% and zero
false positives. These are fixed-seed synthetic structural-transfer results, not
human-reviewed semantic accuracy. An earlier full 5.84M synthetic
leave-one-domain-out snapshot passed the expansion gate, and the v3 frontier-aware
contract passes a fresh 29K training/export diagnostic. The hierarchical split now
also trains the current 5.84M controller from three language traces and verifies its
math/vision transfer plus macro composition. The broader frontier-aware 5.84M LODO
rerun and verified human-reviewed 20/100-shot gates remain unevaluated. With the
digest-bound LMV benchmark and verified typed
online reviewed-learning and sparse grounding self-learning milestones, the local
2026-07-22 `python -m pytest -q` regression passes 884 tests plus 35 subtests in
297.27 seconds.
The CI-equivalent dependency-free fast-core gate, including the
prompt-first, verified-curriculum, development-evaluation, and sealed-student
boundaries, passes 333 tests plus 24 subtests in 28.46 seconds;
`shadow` remains the default.

- `app.py`
  - research-oriented structured reasoning CLI
- `ops_copilot.py`
  - product-style warehouse/operations copilot CLI
- `semop_easy_gui.py`
  - one-page beginner GUI for ops, CP, image QA, concept-store comparison, cluster review, approved-cluster retraining, VLSO impact evaluation, CP parser comparison, labeled dataset download, and VLSO image download staging
- `ops_copilot_gui.py`
  - local browser GUI for testing queries, baselines, and review queue items
- `solve_olympiad.py`
  - symbolic proof-search CLI for olympiad-style math questions
- `solve_contest.py`
  - competitive programming approach + C++17 template generator
- `cp_copilot_gui.py`
  - beginner-friendly local browser GUI for CP analysis, incident ingest, and episode memory
- `solve_hard_problem.py`
  - structured hard-problem solving, verification, and pattern-weight learning
- `vlso_demo.py`
  - Vision-Language Semantic Operators demo for shared language and visual operator reasoning
- `tools/vlso/index_vlso_visual_memory.py`
  - index structured visual observations or image-backed visual memories into a local VLSO embedding store
- `tools/vlso/index_visual_concepts.py`
  - index few-shot visual concept exemplars into a local VLSO concept-memory store
- `tools/vlso/build_visual_concept_candidates.py`
  - build multi-object concept-label candidates for manual few-shot labeling
- `tools/vlso/train_visual_concepts.py`
  - compress labeled concept examples into prototype memory for sample-efficient VLSO learning
- `tools/vlso/train_visual_operators.py`
  - learn higher-level visual operator prototypes such as container-body, opening-control, and attached-grasp from a few labeled images
- `src/semop/operator_proposal.py`
  - propose higher operators from repeated decomposition patterns and geometry/topology signatures before self-evolution validation
- `tools/vlso/run_geometry_reasoning_pipeline.py`
  - end-to-end geometry visual pipeline for candidates, pseudo labels, concept store, operator store, and grounded QA eval and structural operator recovery eval
- `tools/vlso/generate_geometry_dataset.py`
  - generate synthetic geometry images, paired detector-style JSON payloads, and a starter geometry QA eval set
  - retrain approved VLSO clusters into fresh concept/operator stores
- `tools/vlso/recommend_visual_labels.py`
  - rank the next most informative targets to label based on novelty and uncertainty
- `tools/eval/compare_ops_baseline.py`
  - SemOp vs baseline benchmark on labeled cases
- `tools/eval/evaluate_ops_kpis.py`
  - KPI averages on operations case sets
- `tools/eval/evaluate_vlso_grounded_qa.py`
  - grounded VLSO QA evaluator on image-or-observation jsonl cases
- `tools/eval/evaluate_vlso_review_impact.py`
  - compare grounded QA before and after approved-cluster retraining
- `tools/eval/evaluate_cp_parser.py`
  - evaluate heuristic, learned, or side-by-side CP parsers on DSL/frame labels
- `tools/eval/evaluate_hidden_premises.py`
  - evaluate hidden-goal recovery, critical premise recall, unsupported-premise precision, and goal-preservation checks
- `tools/eval/evaluate_semop_stack.py`
  - run a common evaluation snapshot across hidden premises, CP parser structure, and VLSO grounded QA
- `tools/eval/evaluate_operator_intelligence_progress.py`
  - convert the common evaluation snapshot into progress estimates for operator architecture, premise reasoning, world-model quality, and cross-domain transfer
- `tools/eval/evaluate_operator_algebra.py`
  - evaluate operator decomposition recovery and functor-hypothesis recovery
- `tools/eval/evaluate_operator_transfer.py`
  - evaluate evolved operators on a starter cross-domain transfer benchmark
- `tools/eval/evaluate_visual_signal_operator_impact.py`
  - compare retained operators before and after ablating symmetry, closure, and axis-alignment geometry signals
- `tools/eval/compare_operator_proposals.py`
  - compare heuristic and LLM-backed operator proposal engines on the same graph set
- `docs/operator_algebra_and_functors.md`
  - explain operator decomposition and functor-hypothesis alignment
- `tools/ops/learn_feedback_rules.py`
  - converts resolved review items into reusable feedback rules

## Recommended Environment

This repository has already been tested with a Python 3.12 virtual environment and GPU PyTorch.

Known working environment:
- Python `3.12`
- virtual environment: `.venv312`
- torch `2.10.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4060`

Run commands with the script path shown in the docs, for example:

```bash
.\.venv312\Scripts\python.exe tools\cp\train_cp_parser.py ...
```

Current code layout is documented in:
- `docs/project_structure.md`
- `docs/pdf_direction_and_plan_2026_03_10.md`
- `docs/operator_intelligence_system.md`
- `docs/operator_intelligence_roadmap.md` and `docs/operator_intelligence_execution_steps.md` and `docs/jepa_relevance_and_integration.md`
- `docs/operator_intelligence_to_100_plan.md`

Refresh it after structural changes with:

```bash
.\.venv312\Scripts\python.exe tools\maintenance\update_code_structure_docs.py
```

## Quick Start

### 1. Open the easiest GUI

```bash
.\.venv312\Scripts\python.exe semop_easy_gui.py
```

Then open `http://127.0.0.1:8770`.

From the easy GUI you can click through:
- download labeled CP datasets
- build a VLSO image-collection plan
- generate an object-family manifest for bag, box, drawer, door, bottle, tool, cabinet, suitcase, jar, bin, and pouch
- run a family-target batch collector that writes manifest, records, approved, and download-manifest files automatically
- preview public-image cards and approve downloads
- run downloaded-image learning
- label downloaded images with a review queue
- retrain concept and operator stores from approved labels only
- generate synthetic geometry images
- generate CP geometry eval/train starter sets
- run CP LoRA training or resume the latest checkpoint from the easy GUI

### 2. Run the operations copilot on a single SOP question

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

### 3. Open the full operations GUI

```bash
.\.venv312\Scripts\python.exe ops_copilot_gui.py ^
  --review-queue data\ops_review_queue.db ^
  --baseline-config examples\customer_baseline_config.json
```

Then open `http://127.0.0.1:8765`.

### 4. Try the olympiad proof-search prototype

```bash
.\.venv312\Scripts\python.exe solve_olympiad.py ^
  --query "Prove that the sum of two odd integers is even."
```

### 4B. Try the VLSO prototype with structured observations

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --query "How do I put a book into a bag?" ^
  --visual-json examples\vlso\bag_closed_observation.json
```

### 4C. Try the VLSO prototype in deep mode on a raw image

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What shapes are visible here?" ^
  --image-path data\scene.png ^
  --format json
```

If a local DINOv2 checkpoint exists under `models\vision\dinov2\...`, it is auto-resolved.
If not, SemOp falls back to `token_geometry_v1` and keeps the raw-image shape parser active.
This is the intended current architecture: `deep-first + structural fallback`, not `deep-only`.

You can also load learned affordance weights and request a grounded answer directly:

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "How can I access the bag opening?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

For local Qwen-style answer generation on top of the grounded world model, switch `--answer-mode llm` and optionally override `--answer-model-id`.

You can also ground new images through a segmentation-style detector payload and few-shot concept memory:

```bash
python tools\vlso\index_visual_concepts.py ^
  --labels examples\vlso_visual_concepts_template.jsonl ^
  --store data\vlso_visual_prototypes.db
```

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What objects are visible here?" ^
  --image-path data\vlso_samples\backpack_public_domain.jpg ^
  --concept-store data\vlso_visual_prototypes.db ^
  --affordance-weights data\vlso_samples\trained_affordance_weights.json ^
  --answer-mode structured ^
  --format json
```

Segmentation-style detector JSON is accepted through `--detector-json`, including `annotations`, `segments`, and `instances` payloads.
Open Images box annotations can be converted into this detector JSONL shape with `tools\vlso\ingest_open_images_annotations.py`.

For sample-efficient learning, build candidate rows, train compact concept prototypes, and label only the most novel or uncertain targets next. The full loop is documented in `docs/vlso_data_collection_guide.md`. Public API and dataset options are summarized in `docs/data_collection_api_research.md`.
The raw-image path now adds mask refinement, dominant border-frame suppression, and lightweight object/part/affordance inference before graph construction.
Small public tuning samples are stored under `data\vlso_samples\` and listed in `data\vlso_samples\SOURCES.md`.
A collection guide for growing this set to 10-20 images is in `docs\vlso_data_collection_guide.md`.
Label-candidate generation and weight re-estimation CLIs are `tools\vlso\build_affordance_label_candidates.py` and `tools\vlso\train_affordance_classifier.py`. Unsupervised or weakly supervised prototype bootstrapping is available through `tools\vlso\self_train_visual_concepts.py`. You can compare a hand-labeled concept store against a pseudo-labeled store with `vlso_demo.py --compare-concept-store ...`, and review cluster summaries in `semop_easy_gui.py`. Manifest-driven API collection planning is available through `tools\vlso\collect_visual_data.py`, and whitelist/download staging is available through `tools\vlso\prepare_visual_downloads.py`.

### 3D. Try the VLSO prototype with detector output or an explicit local backbone

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What geometric structure is visible here?" ^
  --detector-json examples\vlso\detector_output_example.json
```

```bash
.\.venv312\Scripts\python.exe vlso_demo.py ^
  --mode deep ^
  --query "What geometric structure is visible here?" ^
  --image-path path\to\scene.png ^
  --vision-backbone dinov2_adapter ^
  --vision-model-path models\vision\dinov2\dinov2-small
```

### 4. Generate a contest-programming approach, validate it, and store the episode

```bash
.\.venv312\Scripts\python.exe solve_contest.py ^
  --query "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities." ^
  --episode-store data\cp_episodes.db
```

Competitive-programming knowledge is stored in:
- `data\knowledge\cp_knowledge.json`

Solved contest episodes are stored in:
- `data\cp_episodes.db`

Those episodes are later reused as episodic retrieval and reranking priors for new contest statements.
You can also download and normalize labeled CP datasets with `tools\cp\download_cp_labeled_datasets.py`.

Geometry data bootstrap shortcuts:

```bash
.\.venv312\Scripts\python.exe tools\vlso\bootstrap_geometry_visual_data.py ^
  --workspace data\vlso_geometry_bootstrap
```

```bash
.\.venv312\Scripts\python.exe tools\cp\bootstrap_geometry_corpus.py ^
  --manifest examples\cp_geometry_labeled_manifest.json ^
  --download-root data\cp_geometry_downloads ^
  --output data\cp_geometry_labeled.jsonl
```

The VLSO bootstrap writes a geometry/access preset manifest automatically and can later be re-run with `--execute-collect` or `--execute-downloads`.
The CP bootstrap can consume normal URLs or Hugging Face datasets through the manifest. You can also generate synthetic geometry scenes and eval assets locally with `tools\vlso\generate_geometry_dataset.py`.

Full geometry reasoning bootstrap:

```bash
.\.venv312\Scripts\python.exe tools\vlso\run_geometry_reasoning_pipeline.py ^
  --inputs data\vlso_samples ^
  --workspace data\vlso_geometry_pipeline ^
  --eval-mode heuristic ^
  --answer-mode structured
```

Build a geometry-only CP parser eval set from normalized labeled corpus rows:

```bash
.\.venv312\Scripts\python.exe tools\cp\build_geometry_parser_eval.py ^
  --input data\cp_geometry_labeled.jsonl ^
  --output examples\cp_geometry_parser_eval.jsonl
```

Generate a starter geometry-only CP eval set directly from built-in templates:

```bash
.\.venv312\Scripts\python.exe tools\cp\generate_geometry_eval_templates.py ^
  --output examples\cp_geometry_parser_eval.jsonl
```

### 4B. Open the CP GUI

```bash
.\.venv312\Scripts\python.exe cp_copilot_gui.py ^
  --episode-store data\cp_episodes.db
```

Then open `http://127.0.0.1:8787`.

### 4C. Ingest real WA/TLE/editorial incidents into episodic memory

```bash
.\.venv312\Scripts\python.exe tools/cp/ingest_cp_episodes.py ^
  --inputs examples\cp_incident_cases.jsonl ^
  --store data\cp_episodes.db
```

### 5. Prepare a CP corpus, then build DSL and SFT datasets

```bash
.\.venv312\Scripts\python.exe tools/cp/prepare_cp_corpus.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_statement_corpus_expanded.jsonl

.\.venv312\Scripts\python.exe tools/cp/build_cp_dsl_dataset.py ^
  --inputs examples\cp_corpus_inputs examples\cp_statement_seeds.jsonl ^
  --output examples\cp_dsl_expanded_dataset.jsonl ^
  --corpus-output examples\cp_statement_corpus_expanded.jsonl ^
  --sft-output examples\cp_dsl_expanded_sft.jsonl
```

### 6. Build a train/val bundle, then dry-run or train a small CP parser with optional LoRA

```bash
.\.venv312\Scripts\python.exe tools/cp/build_cp_training_bundle.py ^
  --inputs examples\cp_dsl_expanded_dataset.jsonl ^
  --episode-store data\cp_episodes.db ^
  --train-output data\cp_train.jsonl ^
  --val-output data\cp_val.jsonl ^
  --train-sft-output data\cp_train_sft.jsonl ^
  --val-sft-output data\cp_val_sft.jsonl
```

```bash
.\.venv312\Scripts\python.exe tools/cp/train_cp_parser.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --train-jsonl data\cp_train.jsonl ^
  --output-dir data\cp_parser_dry_run ^
  --dry-run ^
  --use-lora
```

You can score the heuristic parser or a learned parser with:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input data\cp_val.jsonl ^
  --mode heuristic
```

Or compare heuristic vs learned parser side by side:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_cp_parser.py ^
  --input examples\cp_parser_eval.jsonl ^
  --mode compare ^
  --model path\to\your_cp_parser_model
```

Compare VLSO grounded QA before and after approved-cluster retraining:

```bash
.\.venv312\Scripts\python.exe tools/eval/evaluate_vlso_review_impact.py ^
  --input examples\vlso_eval.jsonl ^
  --primary-concept-store data\vlso_visual_prototypes.db ^
  --primary-operator-store data\vlso_visual_operators.db ^
  --compare-concept-store data\vlso_geometry_pipeline_gui\approved_review_concepts.db ^
  --compare-operator-store data\vlso_geometry_pipeline_gui\approved_review_operators.db
```

Build a starter real-image VLSO eval set from downloaded family-batch images:

```bash
.\.venv312\Scripts\python.exe tools\vlso\build_real_image_eval.py ^
  --records data\vlso_family_batch\family_records.jsonl ^
  --manifest data\vlso_family_batch\family_download_manifest.jsonl ^
  --candidate-output data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --seed-output examples\vlso_real_image_eval.jsonl ^
  --limit 24
```

Run grounded QA on that starter real-image benchmark:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_vlso_grounded_qa.py ^
  --input examples\vlso_real_image_eval.jsonl ^
  --mode deep ^
  --answer-mode structured
```

This seed benchmark is intentionally difficult. The current run is a gap-finding benchmark and shows that arbitrary internet-image grounding is still weak.

Run a one-command CP LoRA experiment workspace build and dry-run training plan:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace tests\cp_lora_experiment_smoke ^
  --model local-test-model ^
  --inputs examples\cp_parser_eval.jsonl ^
  --execute-train ^
  --dry-run-train ^
  --local-files-only ^
  --max-steps 4
```

If you replace `local-test-model` with a real local base model, the same command can build train/val bundles, emit SFT files, run LoRA training, and compare learned vs heuristic parsing.

To save checkpoints during training and resume later, add `--save-steps`, `--save-total-limit`, and `--resume-from-checkpoint`:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace data\cp_lora_run ^
  --model path\to\your_local_base_model ^
  --inputs examples\cp_parser_eval.jsonl examples\cp_hidden_constraint_eval.jsonl ^
  --eval-inputs examples\cp_geometry_parser_eval.jsonl ^
  --execute-train ^
  --max-steps 200 ^
  --save-steps 25 ^
  --save-total-limit 3 ^
  --resume-from-checkpoint data\cp_lora_run\training_run\checkpoint-100
```

If you already reviewed `data\vlso_family_batch\real_image_eval_candidates.jsonl`, finalize the approved rows into a gold real-image eval set with:

```bash
.\.venv312\Scripts\python.exe tools\vlso\finalize_real_image_eval.py ^
  --candidates data\vlso_family_batch\real_image_eval_candidates.jsonl ^
  --output examples\vlso_real_image_eval_gold.jsonl
```

To run the CP LoRA experiment against an extra held-out evaluation set, add `--eval-inputs`:

```bash
.\.venv312\Scripts\python.exe tools\cp\run_cp_lora_experiment.py ^
  --workspace data\cp_lora_run ^
  --model path\to\your_local_base_model ^
  --inputs examples\cp_parser_eval.jsonl ^
  --eval-inputs examples\cp_geometry_parser_eval.jsonl ^
  --execute-train ^
  --max-steps 200
```

### 7. Run a hard-problem analysis

```bash
.\.venv312\Scripts\python.exe solve_hard_problem.py ^
  --query "If the bag has no open access, check the zipper before inserting the book." ^
  --memory-store data\semop_memory.db ^
  --memory-source grammar_demo
```

### 8. Compare SemOp against a baseline

```bash
.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py ^
  --input examples\ops_labeled_eval_ko.jsonl ^
  --baseline configurable_keyword ^
  --baseline-config examples\customer_baseline_config.json
```

### 9. Learn feedback rules from reviewed items

```bash
.\.venv312\Scripts\python.exe tools/ops/learn_feedback_rules.py ^
  --review-queue data\ops_review_queue.db ^
  --output data\feedback_rules.json
```

### 10. Re-run the copilot with learned rules

```bash
.\.venv312\Scripts\python.exe ops_copilot.py ^
  --domain warehouse_exception ^
  --scenario exception_response ^
  --feedback-rules data\feedback_rules.json ^
  --query "The aisle is blocked and approval is still missing. What should I do?" ^
  --context-file examples\customer_sop_sample.md
```

## Where To Put Data

There are two main data paths.

### A. Public reasoning datasets

You do not need to download these manually if you use the curated manifest scripts.

1. Generate a manifest:

```bash
.\.venv312\Scripts\python.exe tools/corpus/write_curated_manifest.py --preset reasoning_core --output data\reasoning_core_manifest.json
```

2. Ingest it:

```bash
.\.venv312\Scripts\python.exe tools/corpus/ingest_public_manifest.py ^
  --manifest data\reasoning_core_manifest.json ^
  --download-root data\reasoning_core_downloads_312 ^
  --normalized-root data\reasoning_core_normalized_312 ^
  --store data\semop_reasoning_core_312.db ^
  --mode heuristic ^
  --overwrite
```

What goes where:
- raw downloaded files: `data\reasoning_core_downloads_312\`
- normalized JSONL files: `data\reasoning_core_normalized_312\`
- final SQLite memory DB: `data\semop_reasoning_core_312.db`

### B. Competitive-programming knowledge, episode memory, and local compiler setup

The CP pipeline uses:
- `data\knowledge\cp_knowledge.json`
- `data\cp_episodes.db`

These files store:
- official online source URLs
- logical problem frames
- CP DSL operators
- algorithm/data-structure triggers
- complexity metadata
- memory schema layers
- compiler and hardware notes
- successful and failed validation episodes when `solve_contest.py --episode-store ...` is used

The current local compiler observation is conservative:
- `g++ = MinGW.org GCC 6.3.0-1`
- generated code avoids fragile syntax and is syntax-checked locally

### C. Customer SOP or manual documents

Put customer `.md` or `.txt` files in a folder such as:
- `data\customer_docs\`

Then build evaluation stubs from those documents:

```bash
.\.venv312\Scripts\python.exe tools/ops/build_customer_eval_from_docs.py ^
  --inputs data\customer_docs ^
  --output data\customer_eval.jsonl ^
  --max-cases 50
```

If you do not have customer data yet, start from:
- `examples\customer_sop_sample.md`
- `examples\customer_ops_eval_template.jsonl`
- `examples\ops_labeled_eval_ko.jsonl`

## Important Config Files

- baseline config JSON:
  - example: `examples\customer_baseline_config.json`
  - purpose: define a customer-style baseline retriever
- learned feedback rules JSON:
  - example output: `data\feedback_rules.json`
  - purpose: apply supervisor-reviewed corrections back into the copilot
- review queue SQLite:
  - example: `data\ops_review_queue.db`
  - purpose: triage risky answers and record resolution notes

## Main Docs

- logical grammar goal and implementation: `docs/logical_grammar_goal_and_implementation.md`
- hard-problem training and verification: `docs/hard_problem_training_and_verification.md`
- competitive-programming focus and implementation: `docs/cp_focus_research_and_implementation.md`
- usage manual: `docs/usage_manual.md`
- data layout and download guide: `docs/data_layout.md`
- docs index: `docs/index.md`
- GUI and evaluation workflow: `docs/gui_and_eval_workflow.md`
- feedback loop workflow: `docs/feedback_loop_workflow.md`
- customer eval schema: `docs/customer_eval_schema.md`
- architecture and internals: `docs/architecture_and_features.md`
- data-collection API research: `docs/data_collection_api_research.md`
- product framing review: `docs/productization_review.md`

## Validation Status

Latest verified commands:
- `python -m pytest -q`
- `python tools/eval/evaluate_lmv_core_gate.py --require-pass`
- `python tools/eval/evaluate_operator_core_gate.py --require-pass`
- `python tools/eval/evaluate_controller_feature_transfer.py --require-pass`
- `python -m unittest discover -s tests -v`
- `.\.venv312\Scripts\python.exe -m unittest discover -s tests -v`
- `python solve_contest.py --query "Given a weighted graph with N cities and M roads, answer the shortest path from city 1 to all cities."`
- `.\.venv312\Scripts\python.exe tools/eval/compare_ops_baseline.py --input examples\ops_labeled_eval_ko.jsonl --baseline configurable_keyword --baseline-config examples\customer_baseline_config.json`

The old 129-test snapshot referred to the pre-typed product baseline. Current typed
milestone counts and timings are recorded near the typed-core overview above and are
independently exercised by `.github/workflows/typed-core.yml`.

At the time of the legacy ops validation:
- SemOp on labeled ops eval achieved:
  - `avg_relation_recall = 1.0`
  - `avg_answer_term_recall = 0.89`
  - `forbidden_phrase_hit_rate = 0.0`

## Current Limits

- customer evaluation sets are still small unless you add real customer documents
- KPI scores are useful for PoC work, but should still be calibrated against human labels
- review-derived feedback rules are simple and heuristic, not full training updates
- GUI is for internal demos and PoCs, not hardened production deployment


## CP Learning Docs

For the detailed CP-specific workflow, see:
- `docs/cp_focus_research_and_implementation.md`
- `docs/cp_gui_manual.md`
- `docs/cp_training_manual.md`

VLSO examples:
- `examples/vlso_eval.jsonl`
  - starter grounded QA eval and structural operator recovery eval set for VLSO
- `examples/vlso_geometry_eval.jsonl`
  - starter geometry-grounded QA eval and structural operator recovery eval set for VLSO
- `examples/cp_parser_eval.jsonl`
  - starter held-out statement-to-DSL eval set for CP parser comparison
- `examples/cp_geometry_parser_eval.jsonl`
  - geometry-only CP parser eval set generated from normalized labeled corpus rows
- `examples/vlso/bag_closed_observation.json`
- `examples/vlso/geometry_scene.json`
- `examples/vlso/detector_output_example.json`

Recent geometry upgrades:
- VLSO now derives polygon edge entities and grounded geometry relations such as `PARALLEL`, `PERPENDICULAR`, and `EQUAL_LENGTH`
- VLSO can infer shape hypotheses including `triangle`, `right_triangle`, `isosceles_triangle`, `rectangle`, `square`, `parallelogram`, and `quadrilateral`
- CP parsing now recognizes geometry-heavy statements and routes them into `computational_geometry_analysis`
- CP code generation now includes a geometry template with `Point`, `cross`, `dot`, and orientation-style predicates

## New evaluation paths

- Real-image VLSO reviewed gold set: `examples/vlso_real_image_eval_gold.jsonl`
- CP hidden-constraint parser bench: `examples/cp_hidden_constraint_eval.jsonl`
- Script compatibility before/after experiment: `tools/eval/run_premise_compatibility_experiment.py`

Example commands:
```bash
python tools\vlso\finalize_real_image_eval.py --candidates data\vlso_family_batch\real_image_eval_candidates.jsonl --output examples\vlso_real_image_eval_gold.jsonl --auto-approve-limit 8
python tools\eval\evaluate_semop_stack.py --hidden-premises examples\hidden_premise_eval.jsonl --cp-input examples\cp_parser_eval.jsonl --cp-hidden-input examples\cp_hidden_constraint_eval.jsonl --vlso-input examples\vlso_eval.jsonl --vlso-real-image-input examples\vlso_real_image_eval_gold.jsonl
python tools\eval\run_premise_compatibility_experiment.py --input examples\hidden_premise_eval.jsonl --memory-store data\semop_memory.db --output-model data\script_compatibility_model.json
```


## QLoRA and distillation

Teacher traces for hidden-premise reasoning, CP structuring, and VLSO grounded QA can be exported with `tools/eval/export_teacher_traces.py`. The roadmap is documented in `docs/qlora_distillation_roadmap.md`.

The same exporter also supports operator proposal and self-evolution traces via `--operator-transfer-input examples\operator_transfer_eval.jsonl`.

You can then build a generic operator-learning curriculum and dry-run a small student model:

```bash
.\.venv312\Scripts\python.exe tools\eval\build_operator_learning_bundle.py ^
  --teacher-traces data\teacher_traces.jsonl ^
  --workspace data\operator_learning_bundle
```

```bash
.\.venv312\Scripts\python.exe tools\eval\run_operator_training.py ^
  --workspace data\operator_learning_bundle ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dry-run ^
  --use-lora
```

The full workflow is documented in `docs/operator_learning_plan.md`.

To reduce black-box reasoning, SemOp now also compiles hidden-premise and operator-algebra outputs into a deterministic operator program before final response synthesis. This runtime records satisfied facts, missing facts, and goal-risk decisions instead of leaving the whole reasoning path inside the model.

## Overall Understanding Benchmark

Run the broad understanding check across hidden-premise reasoning, CP structuring, CP hidden constraints, starter VLSO, and reviewed real-image VLSO:

```bash
.\.venv312\Scripts\python.exe tools\eval\evaluate_understanding.py
```

In the easy GUI, use `5. Geometry starter tools -> Evaluation shortcuts -> Run overall understanding benchmark`.

## Shared Agent Doctrine

All future agent work in this repository should follow `docs/multi_agent_operator_doctrine.md`: recover shared basis operators first, compose higher operators through algebra, and only retain operators that survive verifier and transfer checks.
