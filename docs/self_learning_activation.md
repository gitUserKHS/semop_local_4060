# 자가 학습과 사람 승인 경계

## 데이터 권한

인터넷 데이터는 임의 crawl하지 않는다. source manifest에 HTTPS 주소, 라이선스, revision,
최대 크기와 SHA-256이 있는 자료만 내려받는다. frontier judge와 로컬 모델의 결과는 label이
아니라 후보이며 SemOp verifier 또는 digest-bound 사람 review가 필요하다.

도메인당 합성 record는 5,000개, operator 깊이는 2~6, positive당 hard negative는 4개로
제한한다. 사람 검토량은 `0/5/20/100-shot`으로 따로 보고한다. 문장만 무작위로 나누지 않고
operator 조합, 그래프 구조, 깊이와 문제 크기를 heldout으로 둔다.

## 자동화 범위

```text
수집 -> dedup/license 검사 -> 후보 학습 -> sealed 평가 -> candidate 저장
                                                       |
                                            사람의 명시적 승인
                                                       |
                                             active.json 원자 교체
```

`ArtifactPromotionStore`는 controller, macro, semantic model 후보를 저장한다. replay integrity
100%, false acceptance 0건이 공통 조건이다. controller는 solve rate 1%p 초과 회귀가 없어야
하고 두 도메인 이상에서 30% expansion 절감을 보여야 한다. 학생 모델은 teacher retention
95%와 자원 30% 절감을 충족해야 한다.

승격에는 정확한 `PROMOTION_APPROVAL_ATTESTATION`과 reviewer ID가 필요하다. 초보자용
`semop-promote approve ... --confirm`도 내부적으로 같은 attestation을 기록하며 자동 승격은
하지 않는다. 후보 ID는 artifact hash뿐 아니라 전체 gate report와 metadata까지 묶는다.
평가 뒤 artifact hash나 후보 JSON이 바뀌면 승격 또는 로드가 거절된다. `active.json`은 이전
pointer를 보존해 rollback할 수 있다.
primitive operator는 자동 승격 대상이 아니며 사람이 타입·guard·효과를 검토해야 한다.

## 합성 development lane

사람 검토 corpus가 충분하지 않을 때는 `VerifiedSemanticCurriculum`으로 관찰된 실패 하나를
재현하는 작은 합성 curriculum을 만들 수 있다. 이 lane은 generator-known semantics와 exact
executor replay로 학습 코드와 모델 후보를 시험하지만, 다음 경계를 고정한다.

- 생성된 train, validation, sealed-development 파일은 모두 `promotion_eligible: false`다.
- validation과 sealed-development는 train과 다른 문구 template family를 사용한다.
- executable hard negative는 학습 정답으로 사용하지 않고 false acceptance 검사에 남긴다.
- `evaluate_verified_semantic_student.py`는 train 평가를 거절하고 base 대비 실제 개선,
  replay integrity, model coverage, 원래 실패 probe를 함께 기록한다.
- 통과한 후보도 `retain_for_human_sealed_review`일 뿐 stage 또는 activate되지 않는다.

따라서 합성 heldout에서 100%를 얻어도 사람 semantic correctness나 실제 배포 성능을 뜻하지
않는다. 이후 별도의 사람 검토 trace와 untouched sealed set에서 아래 정식 승격 gate를 다시
통과해야 한다.

## 로컬 의미 trace에서 학생 후보까지

`balanced` Qwen이 사용된 요청은 `SemanticTraceStore`가 원문 요약, canonical typed JSON,
모델 fingerprint, replay 결과를 하나의 digest로 묶어 로컬 SQLite에 기록한다. 결정론적
compiler 결과는 중복 저장하지 않는다. 쉬운 화면에서 사람이 정확한 trace를 확인하고
`해석이 맞아/틀려`를 선택해야 review digest가 생긴다. 답 직후 검토하지 않은 텍스트
trace는 persistent `모델 해석 검토함`에서 다시 볼 수 있다. 이 inbox는 최신 pending trace만
보여 주며 review가 생긴 항목은 즉시 사라진다. PNG/JPEG 입력은 image index, MIME, 크기,
SHA-256 manifest가 trace ID에 결속되고 exact bytes가 같은 로컬 SQLite에 저장된다. inbox는
digest를 다시 검사한 이미지에만 preview와 승인·거절 권한을 준다. 저장되지 않았거나
브라우저 preview를 지원하지 않는 media는 fail-closed로 막는다.

검토가 끝난 trace는 다음 `balanced` 또는 승인된 `economy` 요청부터 0-parameter episodic
memory로 사용된다. 텍스트는 Unicode·대소문자·공백 정규화 후 exact 일치, 한 장의 이미지는
SHA-256 exact 일치일 때만 최신 사람 승인·거절과 이전 proposal을 Qwen의 untrusted guidance에
넣는다. 나중에 사람이 판정을 고치면 이전 memory를 덮어쓰지 않고 latest review가 즉시 검색
결과를 바꾼다. 이 경로는 언어·수학·비전 반복 입력에서 검토 결과를 잊지 않기 위한 첫
소비자이며, paraphrase나 다른 이미지로의 일반화 또는 proof 승격을 의미하지 않는다.

정규화된 원질문·workspace·정확한 이미지 digest로 `train` 80%, `validation` 10%,
`sealed` 10% 역할을 처음부터 고정한다. 모델 출력이 달라져 trace ID가 바뀌어도 같은 입력은
같은 역할에 남는다. 기본
export는 `train`만 포함한다. 다른 역할은 `semop-learn export --split validation`처럼
명시적으로 분리해 내보낸다.

```powershell
semop-student-cycle status
semop-student-cycle run --confirm --device cuda
```

`status`는 모델을 로드하지 않는다. 최신 사람 review에서 text-only train, validation, sealed
corpus를 메모리로 구성해 positive와 hard negative 수를 보여 주고, 정규화한 원문 질문 또는
typed semantic target이 역할 사이에 중복되면 누수로 차단한다. `run --confirm`은 한 번의
고유한 `artifacts/promotion/semantic-model/runs/<run-id>` 아래에 train/sealed snapshot,
candidate LoRA, 봉인 평가 리포트를 함께 보존한다. 순서는 항상
`export -> train -> sealed evaluation -> passing candidate stage`이며 중간 실패나 gate 탈락은
`active.json`을 만들거나 바꾸지 않는다.

텍스트 positive에는 사람 semantic review와 typed proof replay가 모두 필요하다. 거절된
해석은 hard negative로 보존하지만 현재 SFT loss에는 넣지 않고 이후 ranking용으로 예약한다.
exact media가 있는 이미지 review는 사람 기준 semantic correctness label로 기록할 수 있지만
proof 성공으로 승격되지 않는다. 검토 픽셀을 함께 소비하는 multimodal trainer가 생기기
전까지 image trace는 text-only export에서 제외한다. 따라서 현재 cycle은 실제로 자동
수집·후보 학습까지 수행하지만 승격은 자동화하지 않는다.

내부 학습 API는 `train` 이외의 record를 거절하고, 봉인 평가 API는 반대로 `sealed`
이외의 record를 거절한다. 평가는 학생과 pinned 2B teacher를 직접 실행해 reviewed typed
payload 유지율 95%, replay integrity 100%, proof-eligible false acceptance 0건, latency 또는
VRAM 30% 절감을 계산한다. 결과 digest는 corpus, adapter, case별 결과와 자원 측정을 함께
묶는다. one-run cycle은 통과 리포트만 `stage`하며 자동 활성화 권한은 없다. 사람의
`semop-promote --root artifacts/promotion/semantic-model approve <candidate-id> --reviewer <이름> --confirm`
이 별도로 필요하다.

학생 모델은 기본적으로 `artifacts/promotion/semantic-model`에 stage한다. `economy` 런타임은
여기의 승인된 `semantic_model` 후보만 읽고, 0.8B base ID와 검토된 정확한 revision이
일치하는지 확인한다. 서버가 실행 중 새 후보를 승인하면 candidate ID가 바뀌어 다음 요청부터
새 backend를 구성하며, 실제 LoRA 로드 직전에도 디렉터리 digest를 다시 계산한다.

## 개발자가 확인할 것

1. `python -m pytest`로 기존 proof·grounding·promotion 불변식을 모두 실행한다.
2. `python tools/eval/evaluate_low_resource_transfer.py`로 expansion과 자원 gate를 확인한다.
3. semantic correctness와 replay integrity를 같은 숫자로 합치지 않는다.
4. candidate 선택용 validation과 마지막 untouched test를 분리한다.
5. 실패한 승격은 기존 active artifact를 바꾸지 않는지 확인한다.
