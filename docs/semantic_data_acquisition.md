# Semantic Data Acquisition

## 목적

SemOp은 데이터를 세 경로에서 얻는다.

1. SemOp 검증기가 직접 생성한 합성 사례
2. 라이선스가 확인된 공식 공개 데이터
3. 코아나 frontier judge가 만든 후보 사례

세 경로의 학습 권한은 같지 않다. 데이터가 많다는 이유만으로 typed fact나 정답이
될 수 없으며, 원시 입력에서 typed candidate로 변환되는 경계를 별도로 검증한다.

## 신뢰 규칙

| 라벨 출처 | 최초 상태 | 학습 편입 조건 |
| --- | --- | --- |
| `programmatic_oracle` | 검증 대기 | 같은 operator program을 replay |
| `published_label` | imported proposal | SemOp adapter와 verifier를 통과 |
| `model_proposal` | model proposal | 독립 verifier 또는 사람 검토 |
| `human_review` | review evidence | 정확한 candidate digest에 결속된 검토 |

공개 벤치마크의 정답은 그 벤치마크 안에서는 gold일 수 있다. 그러나 자연어 문장이나
이미지를 SemOp의 typed predicate로 옮긴 결과까지 자동으로 보증하지는 않는다. 따라서
`published_label`도 직접 학습을 허용하지 않는다.

코아나 다른 LLM은 다음 작업을 수행할 수 있다.

- 새로운 표현과 hard negative 제안
- 공개 사례를 SemOp DSL 후보로 변환
- 실패 trace를 묶어 review 우선순위 제안

모델이 만든 라벨을 `human_review`로 기록하거나, 자기 답을 자기 검증만으로 승격시키는
것은 금지한다.

## 공식 v1 소스

매니페스트는 `data/sources/semantic_sources.v1.json`에 있다.

- 언어: [AllenAI RuleTaker](https://github.com/allenai/ruletaker) 생성기와 theorem-prover labeling 도구
- 수학: [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) 생성기
- 비전: [DeepMind dSprites](https://github.com/google-deepmind/dsprites-dataset) 이미지와 ground-truth latent factors

각 URL은 branch 이름이 아니라 불변 commit revision을 포함한다. 라이선스, 최대 허용
크기, adapter id, 라벨 권한과 semantic gate도 함께 기록한다.
[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)는 scene graph와
functional program이 있어 다음 relational-vision 단계에 유용하지만, no-image 묶음도
약 86 MB이므로 v1 기본 소스에서는 제외했다.

## 사용법

네트워크 없이 계획만 확인한다.

```powershell
python tools/data/fetch_semantic_sources.py --list
```

한 소스를 명시적으로 받는다. 다운로드에는 라이선스 확인 플래그가 필요하다.

```powershell
python tools/data/fetch_semantic_sources.py `
  --source deepmind-mathematics-generator `
  --fetch `
  --accept-license
```

현재 v1 소스는 실제 파일을 받아 SHA-256까지 고정했으므로 bootstrap 없이 받는다.
새 소스를 catalog에 추가하는 관리 작업에서만 `--bootstrap-lock`을 사용한다. 출력
receipt의 SHA-256을 검토해 매니페스트의 `expected_sha256`에 넣은 뒤에는 bootstrap
없이 받아야 한다.

```powershell
python tools/data/fetch_semantic_sources.py `
  --source deepmind-mathematics-generator `
  --fetch `
  --accept-license `
  --bootstrap-lock
```

수집기는 다음을 강제한다.

- HTTPS와 HTTPS redirect
- source별 최대 byte 수
- SHA-256 일치
- 임시 파일에 받은 뒤 원자적으로 교체
- source revision, manifest digest, 시각, 파일 digest receipt
- 명시적인 source 선택과 라이선스 확인

다운로드 결과는 기본적으로 `artifacts/datasets/`에 저장되어 Git에 들어가지 않는다.

## 자가 생성 데이터

현재 `generate_controlled_semantic_benchmark()`는 언어, 수학, raster vision 사례를
로컬에서 만들고 `programmatic_semantic_learning_examples()`가 원시 입력을 다시
실행해 라벨을 검증한다. 비용이 거의 없고 조합을 많이 만들 수 있다는 장점이 있지만,
생성기와 같은 문법만 반복하면 일반화가 과대평가된다.

따라서 평가에서는 반드시 다음을 분리한다.

- 표현 template holdout
- operator composition holdout
- 더 깊은 수학 식과 더 큰 시각 구조
- 공개 소스 holdout
- 사람이 작성한 작은 semantic gold

이 구조에서 인터넷 데이터는 합성 데이터의 양을 대체하는 것이 아니라, 생성기 밖의
표현과 구조를 제공하는 독립 검사축이다.
