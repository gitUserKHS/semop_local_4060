# 로컬 의미 모델 설치와 운영

## 프로필

| 프로필 | 구성 | 용도 |
|---|---|---|
| `symbolic` | 모델 없음, lexical operator 검색 | 가장 빠르고 완전 오프라인인 기본 검증 경로 |
| `balanced` | Qwen3.5-2B + multilingual-e5-small | 텍스트·이미지 typed 후보 생성 |
| `economy` | 승인된 Qwen3.5-0.8B LoRA + multilingual-e5-small | 검증 trace로 증류하고 봉인 평가를 통과한 저사양 경로 |

Florence-2-base-ft는 OCR이나 영역 후보가 필요할 때만 지연 로드한다. Qwen이 이미 이미지
영역을 제안했다면 Florence를 다시 호출하지 않는다. Qwen 후보가 부족해 Florence fallback이
필요한 요청은 두 모델이 잠시 함께 메모리를 사용할 수 있으므로 더 느리다. SigLIP과 Gemma는
v1 의존성이 아니다.

## Windows RTX 4060 환경

```powershell
C:\Users\imade\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements-semantic.txt
$env:PYTHONPATH = "src"
.\.venv312\Scripts\python.exe -m semop.model_manager doctor
.\.venv312\Scripts\python.exe -m semop.model_manager download --profile balanced
```

런타임은 네트워크에서 모델을 자동으로 받지 않는다. 다운로드 명령은 검토된 네 모델 ID만
허용하며 Hugging Face가 돌려준 commit과 모든 파일 SHA-256을
`artifacts/models/receipts/`에 기록한다. 가중치와 receipt runtime 산출물은 Git에 넣지 않는다.

Qwen 입력은 처음에는 4,096 token, 출력은 512 token으로 제한한다. CUDA에서는 BF16을
우선하며 bitsandbytes를 사용하지 않는다. `semop.model_manager doctor --json`으로 CUDA,
패키지, 캐시, 디스크 상태를 확인한다.

## 2026-07-22 로컬 실측

RTX 4060 Windows 환경에서 Python 3.12.10, PyTorch 2.10.0+cu128,
Transformers 5.3.0 조합을 직접 확인했다. 검토된 revision의 Qwen3.5-2B,
Qwen3.5-0.8B, multilingual-e5-small, Florence-2-base-ft가 로컬 캐시에 있으며
`doctor`가 네 snapshot을 모두 찾는다. Florence 실행에는 `timm`, `einops`,
`safetensors`도 필요하므로 `requirements-semantic.txt`에 고정했다.

실측 smoke는 성능 벤치마크가 아니라 한 컴퓨터의 기능 확인값이다.

| 경로 | 관찰값 | 해석 |
|---|---:|---|
| Qwen3.5-2B 수학 typed 제안, 첫 요청 | 약 16.8초, GPU 메모리 약 4.53GB | `12-5=7` 후보를 exact executor가 `12-5`로 정규화하고 replay 성공 |
| Qwen3.5-2B 이미지 후보, 첫 요청 | 약 25.3초, GPU 메모리 약 4.54GB | 모델이 영역을 함께 제안해 Florence는 로드되지 않음 |
| 같은 이미지의 두 번째 요청 | 약 8.04초 | warm 요청에서 목표 p95 10초 안쪽인 단일 관찰값 |
| Florence-only fallback, 첫 요청 | 약 28초 | pinned 호환 경로가 실행됐지만 합성 글자 이미지의 OCR box는 비어 있었음 |
| Qwen3.5-2B 일반 채팅, 정체성 기억 고정 후 cold 요청 | 약 21.3초, GPU 메모리 약 4.55GB | invalid typed language payload를 실행하지 않고 정확한 프로젝트 설명을 `best_effort`로 보존 |
| Qwen3.5-2B 일반 Python 비교, cold 요청 | 약 34.09초, GPU 메모리 약 4.62GB | 첫 출력의 list-comprehension AST를 결정론적 비교 예시로 조립, repair 0회, `best_effort` |
| 같은 Python 비교의 warm 요청 | 약 20.78초 | 같은 프로세스에서 repair 0회; 직전 관측 병목 56.32초보다 감소 |
| 같은 프로세스의 문맥 포함 후속 채팅 | 약 4.82초 | 최근 대화를 사용해 `더 짧고 쉽게`라는 후속 지시를 반영 |
| 명시적 장기기억 current A/B | 없음 0.139ms / 있음 0.383ms | 양쪽 생성 0 tokens; missing-memory ASK 또는 explicit retriever, `별빛-27` 반환 |
| 명시적 기억 semantic paraphrase A/B | cold 5건 12.94초 / warm 5건 70.5ms | lexical `0/5` → typed facet+E5 `5/5`, 무관 오회상 `0/3`; 다섯 facet smoke |
| Qwen3.5-2B cross-session episode A/B, 최적화 전 | 없음 5.23초 / 있음 4.80초 | lexical score `0.655797`; `hippocampus_recall`이 `해마-7319`로 바뀌고 모델 사용 기록, 모두 `best_effort` |
| cross-session episode current A/B | 없음 0.196ms / 있음 0.121ms | 양쪽 생성 0 tokens; missing-memory ASK 또는 episode retriever, `해마-7319` 반환 |
| same-session archive current A/B | 없음 0.120ms / 있음 0.092ms | 양쪽 생성 0 tokens; missing-memory ASK 또는 `RETRIEVE → SELECT`, `등대-4821` 반환 |
| 최신 장기 작업 조회 | 미선택 0.162ms / 다음 행동 0.131ms / 전체 요약 0.073ms | Qwen 없이 revision 3의 진행·결정·`검증-단계-42`를 반환하고 revision 2 행동은 제외 |
| episodic recall CPU gate | 약 0.480초, SQLite 167,936 bytes | 100 turn 복원, cross/same-session archive top-1 `1.0`, 최근 작업기억 중복 0건; 전체-history 인덱스 포함 |
| 5,000 exchange 전체-history gate | backfill 101.5ms / cross 20.6ms / same-session 24.0ms, DB 5.01MB | 최신 400 밖 첫 episode를 양쪽 scope에서 회상; Qwen·인터넷 없이 실행 |

이미지 경로의 후보 답 `SEMOP 42`와 영역은 계속 `PROPOSED`였고 결과 상태는
`best_effort`였다. 이는 자연 이미지 의미 정확도나 OCR 성공률을 입증하지 않는다.
Qwen3.5-0.8B 원본은 같은 사과 뺄셈 smoke를 잘못 `language` 도메인으로 제안했으므로,
원본 base를 `economy`로 자동 대체하지 않고 승인된 증류 LoRA만 허용하는 현재 정책을
유지한다. 냉시작 25.3초는 아직 작은 요청 p95 10초 목표를 충족하지 못하며, warm 측정과
분리해 후속 평가한다.

일반 채팅 smoke의 첫 답은 문장 형태는 자연스러웠지만 SemOp 이름을 임의의 약자로 풀었다.
이는 문맥 전달 성공과 semantic correctness가 같지 않다는 실제 반례다. 이후 system prompt에
프로젝트 정체성을 고정했으며, 사용자 사실은 이 고정 지식에 섞지 않고 명시적 장기기억과
reviewed semantic memory를 통해 별도로 회상한다. 수정 후 실제 재실행은 SemOp의 typed
operator·deterministic executor·proof replay 구조를 설명했고 repair는 0회였다. 모델 후보는
계속 `best_effort`다.

같은 날 `evaluate_memory_continuity.py --semantic-tier balanced`로 실제 explicit-memory A/B도
수행했다. 질문은 저장한 프로젝트 코드명이 무엇인지였고, 기억이 없을 때 모델은 문맥에 이름이
없다고 답했다. lexical retrieval score `0.245536`으로 관련 메모리 하나를 제공하자 정확히
`별빛-27`만 답했다. 최적화 전 기억 없는 Qwen은 81 tokens와 약 46.35초였다. 현재는 기억
없음 `ASK`와 기억 제공 `RETRIEVE → SELECT`가 모두 0 tokens이며 약 0.139ms/0.383ms였다.
provenance의 `memory_context_used_by_retriever`가 참이었다. 이 결과는 한 marker의 단일 smoke이며
장기기억 전반의 의미 일반화를 주장하지 않는다.

같은 실행의 episodic A/B는 고유 표식 `해마-7319`를 과거 session의 exchange에만 저장했다.
새 session에서 episode 없이 물으면 모델은 `hippocampus_recall`을 만들었고, score `0.655797`인
episode를 제공하면 정확히 `해마-7319`라고 답했다. `episode_context_used_by_model=true`였지만
상태는 계속 `best_effort`였다. 이는 직접 회상 최적화 전 기준선이고, 현재 같은 고관련 회상은
`episode_context_used_by_retriever=true`를 남긴다. 이 단일 표식 결과는 장기간 자연 대화
일반화 증거가 아니다.

긴 단일 session 실험은 첫 exchange의 `등대-4821`을 네 개의 후속 exchange로 최근 작업기억
밖에 밀어냈다. `same_session_archive` 없이 모델은 `SemOp`이라고 추측했고, score `0.590461`인
episode를 제공하면 정확한 marker를 반환했다. 최적화 전 계측은 99/131 tokens, repair 0회,
token limit 미도달인데도 50.12초/45.66초였다. 고관련 직접 회상만 작은 deterministic
retriever로 바꾼 뒤 기억 제공 경로는 생성 0 tokens와 약 0.18ms가 됐다. 이어 미제공 개인
회상도 missing-memory ASK로 바꾼 현재 측정은 없음/있음 약 0.120ms/0.092ms다. 새로운 추론
요청만 Qwen의 현재 GPU 지연시간을 가진다.

일반 Python 질문은 현재 C++ contest verifier로 보내지 않는다. 모델이 `language`나 `coding`
도메인으로 잘못 표시해도 Python source 후보가 있으면 AST parse와 요청 구조를 확인해 코드
블록으로 조립하고, typed 논리 사실로는 실행하지 않는다. list-comprehension 대 `for` 예시에서
모델이 단일 generator list-comprehension만 제안하면 작은 AST 연산자가 초기화된 입력과
동등한 `for + append` 결과를 조립한다. 최종 블록에는 실제 `ast.For`와 `ast.ListComp`가 모두
있어야 한다. 이 검사는 문법과 응답 형태만 다루며 코드 실행, 테스트 통과, 알고리즘 정답,
자연어 설명의 정확성을 보장하지 않는다. 로컬 cold/warm 실측을 재현하려면:

```powershell
.\.venv312\Scripts\python.exe tools/eval/probe_local_assistant.py --case coding --case coding --jsonl artifacts/probes/coding.jsonl
```

파서 진단이 필요할 때만 `--raw-jsonl artifacts/probes/coding-raw.jsonl`을 추가한다. 원문 모델
출력에는 사용자 요청 내용이 포함될 수 있으므로 이 파일은 로컬 ignored artifact로만 둔다.

같은 평가에서 Qwen은 최신 task checkpoint를 제공했을 때 올바른 행동을 찾았지만 요청보다
긴 설명을 덧붙였고, checkpoint가 없을 때는 `deploy`를 임의로 제안했다. 관측된 이 실패를
근거로 현재 제품 경로는 명시적인 “선택한 장기 작업의 상태·결정·다음 행동” 질문을 모델에
보내지 않는다. `task_context_retriever`가 최신 revision에서 필요한 필드를 직접 선택하며,
작업 미선택은 `unsupported`, 선택 후 직접 조회는 `best_effort`와
`answer_source=task_context`로 기록한다.

### 0.8B 산술 LoRA 개발 실험

원본 0.8B의 실제 실패를 좁게 고치기 위해 한국어 덧셈·뺄셈·곱셈 curriculum을 생성하고
RTX 4060에서 LoRA 후보를 학습했다. 정답 식은 생성기가 알고 있으며 exact arithmetic
executor가 모두 replay한다. train, validation, development-heldout은 숫자 무작위 분할이
아니라 서로 겹치지 않는 문구 template family와 명사 집합으로 나뉜다.

```powershell
python tools/data/generate_verified_semantic_curriculum.py `
  artifacts/distillation/verified-arithmetic-v1
python tools/train/train_semantic_student.py `
  artifacts/distillation/verified-arithmetic-v1/train.jsonl `
  artifacts/models/qwen35-08b-arithmetic-v1 --epochs 2 --device cuda --seed 4060
python tools/eval/evaluate_verified_semantic_student.py `
  artifacts/distillation/verified-arithmetic-v1/sealed.jsonl `
  --adapter artifacts/models/qwen35-08b-arithmetic-v1
```

실측 결과는 다음과 같다. 시간은 bounded repair를 포함한 요청당 중앙값이다.

| 분할 | 원본 0.8B | LoRA 후보 | 원본/후보 중앙 지연 | 후보 false acceptance |
|---|---:|---:|---:|---:|
| validation, 12 positive | 4/12 (33.3%) | 7/12 (58.3%) | 17.96초 / 3.19초 | 0 |
| development-heldout, 12 positive | 8/12 (66.7%) | 12/12 (100%) | 16.95초 / 3.14초 | 0 |

후보 peak VRAM은 약 1.83GB였고 학습 중 관찰값은 약 4.08GB였다. 원래 실패한
`사과가 12개 있었는데 5개 먹었어`도 후보에서는 정확한 typed 뺄셈으로 컴파일됐다.
validation 곱셈은 여전히 0/4였지만 heldout 문구에서는 4/4였으므로, 작은 표본에서 문구
민감도가 남아 있다는 뜻으로 해석한다.

이 결과는 **승격 평가가 아니다**. 전부 생성기 의미를 정답으로 삼은 작은 development
corpus이고 사람 semantic gold가 없다. corpus manifest와 평가 report는
`promotion_eligible: false`이며 후보는 `retain_for_human_sealed_review` 상태로만 보존한다.
`economy`의 `active.json`은 바뀌지 않았다.

## 학생 모델 증류

승인된 corpus가 준비됐는지 확인한 뒤 one-run cycle로 candidate LoRA를 만든다.

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements-distill.txt
$root = "artifacts\promotion\semantic-model"
.\.venv312\Scripts\python.exe -m semop.model_manager download --profile economy
semop-student-cycle status
semop-student-cycle run --confirm --device cuda
```

설정은 BF16, sequence 2,048, batch 1, gradient accumulation 16, LoRA rank 16이다. 이 명령은
candidate만 저장하고 활성 모델을 바꾸지 않는다. 학생은 sealed set에서 teacher의
replay-verified completion 95% 이상을 유지하고 지연시간 또는 VRAM을 30% 이상 줄여야 한다.

학습 API는 `train` trace만 받아들인다. one-run cycle은 처음부터 역할이 고정된 `sealed`
trace를 별도 snapshot으로 보존하고 실제 0.8B LoRA와 2B teacher를 같은 입력에서 실행한다.

봉인 evaluator는 reviewed typed payload 일치율, proof replay integrity, 사람이 거절한
의미의 재출력, median latency와 peak VRAM을 측정한다. corpus와 LoRA 디렉터리 digest를
결과에 묶는다. `sealed` positive가 없거나 train record가 섞이면 모델을 로드하기 전에
실패한다. 합격하면 candidate record까지만 stage하며 활성 모델은 바꾸지 않는다.

`economy`는 원본 0.8B 모델을 바로 실행하는 별칭이 아니다. 아래 순서로 승인된 LoRA가
`active.json`에 연결되어 있어야 하며, 없거나 파일 digest가 바뀌면 fail-closed로 거절된다.

```powershell
.\.venv312\Scripts\python.exe -m semop.promotion_cli --root $root approve <candidate-id> `
  --reviewer "human:local-user" --confirm
.\.venv312\Scripts\python.exe -m semop.promotion_cli --root $root status
```

cycle의 stage 단계는 학습 요약에서 정확한 base model ID와 revision을 자동으로 묶는다.
승격 CLI의 `approve`는 사람이 봉인 리포트를 읽었다고 명시한 뒤에만 활성화한다. 채팅 CLI에서
다른 저장소를 쓰려면 `--tier economy --promotion-root <경로>`를 함께 지정한다. `peft`는
승인된 어댑터를 로드할 때도 필요하므로 economy 운영 환경에는
`requirements-distill.txt`를 설치한다.
