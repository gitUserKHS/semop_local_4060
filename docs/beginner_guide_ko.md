# SemOp 쉬운 시작 가이드

이 문서는 코드를 처음 보는 사용자도 SemOp의 현재 기능을 직접 실행해 볼 수 있도록 설명한다.
Typed DSL, 연산자 이름, Python API를 몰라도 된다.

## 가장 쉬운 실행 방법

Windows에서 저장소 폴더의 `start_semop.bat`를 더블클릭한다.

1. 검은 터미널 창이 열린다.
2. 잠시 뒤 기본 브라우저에 `SemOp 쉬운 시작` 화면이 열린다.
3. `코딩`, `언어 조건`, `수학식`, `색상 비전` 중 하나를 누른다.
4. 예제 버튼을 누르거나 빈칸을 채운다.
5. 아래쪽의 `검증하기` 버튼을 누른다.
6. 사용을 마치면 터미널 창에서 `Ctrl+C`를 누른다.

브라우저가 자동으로 열리지 않으면 터미널에 표시된 주소를 브라우저 주소창에 입력한다.
기본 주소는 `http://127.0.0.1:8765/`이다. 해당 포트가 사용 중이면 SemOp이 다음 빈 포트를 고른다.

## 명령어로 실행하기

Python 3.11 이상이 설치되어 있어야 한다. 프로젝트 루트에서 다음을 실행한다.
코딩 탭에서 실제 컴파일·실행 검증을 하려면 `g++`도 PATH에 있어야 한다. 나머지 세
탭은 Python 표준 라이브러리만으로 실행된다.

```powershell
$env:PYTHONPATH = "src"
python -m semop.beginner_web
```

패키지를 설치한 경우에는 더 짧게 실행할 수 있다.

```powershell
python -m pip install -e .
semop-easy
```

브라우저 자동 열기를 끄거나 포트를 정하려면 다음 옵션을 사용한다.

```powershell
semop-easy --no-browser --port 9000
```

쉬운 시작 실행은 미증명 문제와 파싱 실패처럼 개선에 도움이 되는 입력만 다음 로컬
검토 큐에 저장한다.

```text
artifacts/experience/beginner-experience.db
```

검증된 정상 성공을 전부 저장하지 않으며, 큐의 후보는 사람 검토 전에는 규칙이나
모델에 반영되지 않는다. 입력은 외부 서비스로 전송되지 않는다. 저장을 원하지 않으면
다음처럼 끌 수 있다.

```powershell
semop-easy --no-experience
start_semop.bat --no-experience
```

다른 위치를 쓰려면 `--experience-db 경로`를 지정한다. 저장된 후보는
`tools/eval/review_typed_experience.py`로 확인하고 승인해야 학습 corpus가 될 수 있다.

## 학습 후보를 쉽게 검토하기

쉬운 시작 화면에서는 터미널 명령 없이 두 방식으로 사람 검토를 남길 수 있다.

1. 문제를 푼 직후 `이 문제는 풀려야 해` 또는 `이 문제는 풀리지 않아야 해`를 누른다.
2. 미증명·파싱 실패로 자동 수집된 입력은 `로컬 학습 후보 검토`에서 원문을 펼쳐 본다.
3. 두 경우 모두 정확한 입력과 기대 결과를 직접 확인했다는 체크가 있어야 기록된다.

이 버튼은 현재 SemOp의 답을 정답으로 복사하는 기능이 아니다. 예를 들어 새로운 문구를
SemOp이 읽지 못했지만 읽을 수 있어야 한다고 판단했다면 `풀려야 해`로 표시할 수 있다.
반대로 전제가 부족한 입력이라면 `풀리지 않아야 해`로 표시한다. 승인 직후에는 실행
규칙이나 모델이 바뀌지 않고 SQLite 검토 corpus에만 추가된다.

## 검증된 자가학습 시도

`검증 학습 시도`는 승인된 사례만 사용한다. 입력 digest에 따라 사례를 `train`,
`validation`, `candidate_heldout`, `joint_heldout`으로 나누고 다음 조건을 모두 확인한다.

- 언어·수학·비전 세 도메인의 필요한 사례가 모두 있는가
- 각 역할에 grounding할 수 있는 사례가 있는가
- 새 규칙이 이전의 참인 사례를 깨뜨리지 않는가
- 거짓 양성을 만들지 않고 primitive proof replay를 통과하는가
- 마지막 joint heldout에서 실제 새 해결 능력이 확인되는가

처음 몇 개만 검토한 상태에서는 `아직 준비가 안 됐어`라고 거절되는 것이 정상이다.
검증을 통과한 경우에만 다음 두 파일이 갱신된다.

```text
artifacts/rules/beginner-active-rules.json
artifacts/rules/beginner-active-rules.checkpoint.json
```

다음 실행에서는 checkpoint의 SHA-256, 내부 library SHA-256, 규칙 수를 다시 확인한 뒤
규칙을 활성화한다. 파일이 변조되었거나 둘 중 하나만 남아 있으면 시작을 거절한다.
이 해시는 파일 손상과 불일치를 찾는 무결성 검사이며, 디지털 서명이나 로컬 공격자
방어를 뜻하지는 않는다.
저장된 규칙의 로드와 새 승격을 모두 끄려면 다음 옵션을 쓴다.

```powershell
semop-easy --no-learned-rules
start_semop.bat --no-learned-rules
```

규칙 파일 위치는 `--rules-artifact 경로`로 바꿀 수 있다. 이 과정은 현재 작은 typed
Horn 규칙 발견이며, 자유형 언어 모델이나 자연 이미지 모델을 자동 재학습하는 기능은 아니다.

## 선택 사항: 작은 탐색 컨트롤러

SemOp은 controller 파일이 없어도 정상 실행된다. 이때는 결정론적 탐색으로 연산자를
고른다. 별도 연구 실행에서 held-out 문제, 거짓 양성, proof replay gate를 통과한 sparse
controller checkpoint가 다음 기본 폴더에 있으면 시작할 때 자동으로 불러온다.

```text
artifacts/controller/beginner-controller/manifest.json
```

현재 통제된 합성 문제로 이 연결을 시험하려면 다음 연구용 예제를 실행할 수 있다.

```powershell
python examples/typed_self_learning_demo.py `
  --output artifacts/controller/beginner-controller `
  --examples-per-structure 3
```

이 예제의 통과는 정해진 symbolic 구조 사이의 전이를 뜻하며, 실제 자유 언어·고등수학·
자연 사진을 이해했다는 뜻은 아니다.

이 controller는 코딩·언어·수학·비전의 답이나 사실을 직접 만들지 않는다. 현재 적용 가능한
typed operator의 실행 순서만 정하고, 점수 계산이 실패하거나 guided search가 풀지 못하면
결정론적 탐색으로 돌아간다. 화면 아래와 시작 터미널에서 현재 활성 여부를 확인할 수 있다.
현재 학습된 controller의 정량 전이 근거는 주로 언어·수학·비전 fixture에서 나온 것이며,
코딩까지의 학습 전이는 별도 구조 holdout 평가가 더 필요하다.

controller를 명시적으로 끄거나 다른 checkpoint 폴더를 쓰려면 다음처럼 실행한다.

```powershell
semop-easy --no-controller
semop-easy --controller-checkpoint-root artifacts/controller/my-controller
```

checkpoint의 artifact hash가 다르거나 지원하지 않는 policy 종류면 조용히 무시하지 않고
시작을 중단한다. 잘못된 작은 모델보다 검증 가능한 탐색을 선택하기 위한 동작이다.

## 코딩

알고리즘 문제를 한 문장으로 적고 `C++ 풀이 만들고 검증하기`를 누른다. SemOp은 후보
알고리즘 세 개까지 생성한 뒤 구조 적합도, 컴파일 결과, 등록된 표본·무작위 테스트를 함께
사용해 하나를 고른다.

성공 결과에는 다음 네 단계의 typed 증거가 남는다.

```text
CANDIDATE_PROGRAM
  -> COMPILES
  -> TESTS_PASSED
  -> VERIFIED_SOLUTION
```

여기서 `VERIFIED_SOLUTION`은 이 PC의 컴파일러와 해당 알고리즘 계열용 검증기를 통과했다는
좁은 뜻이다. 아직 보지 못한 온라인 저지 테스트의 정답까지 보장하지 않는다. 검증기가 없는
알고리즘 계열은 코드를 보여 주더라도 `미증명`으로 닫힌다.

빠른 네 도메인 계약 검사는 다음 명령으로 실행한다.

```powershell
python tools/eval/evaluate_operator_core_gate.py --require-pass
```

## 언어 조건

네 칸만 사용한다.

- `무엇을 하려는 거야?`: 최종 목표 하나
- `꼭 필요한 조건`: 목표 전에 충족해야 할 항목
- `이미 충족한 조건`: 입력에서 완료됐다고 명시한 항목
- `막혔거나 실패한 조건`: 입력에서 차단됐다고 명시한 항목

여러 조건은 쉼표로 나눈다.

```text
목표: 배포
필요 조건: 테스트 통과, 관리자 승인
충족: 테스트 통과, 관리자 승인
막힘: 없음
```

이 경우 SemOp은 각 필요 조건을 확인한 뒤 `배포 준비 완료` 결론을 증명한다.
관리자 승인을 `막힘`에 넣으면 `배포 준비 안 됨`을 증명한다.
아무 상태도 지정하지 않은 조건이 있으면 준비 완료라고 추측하지 않고 미증명으로 남긴다.

중요한 한계가 있다. 이 기능은 사용자가 입력한 문장 안에서 논리가 맞는지 확인한다.
실제로 관리자가 승인했는지, 테스트가 현실에서 통과했는지까지 확인한 것은 아니다.
화면은 이 차이를 `현실 증거 미검증` 안내로 표시한다.

## 수학식

식 하나를 입력한다.

```text
(2 + 3) * 4 == 20
3*x + 2 = 11
7 < 10
```

지원되는 범위에서는 숫자와 연산을 typed term으로 바꾸고 정확한 계산 연산자를 실행한다.
성공 결과는 전체 계산 프로그램을 처음부터 다시 재생해 확인한다.
현재 범위 밖의 수식은 그럴듯한 답을 만들지 않고 입력 오류 또는 미증명으로 표시한다.

## 색상 비전

목록에서 장면을 고르면 색상 격자와 질문이 함께 나타난다.

- 빨간 도형의 정사각형 여부
- 빨강과 파랑의 좌우 관계
- 서로 떨어진 도형의 개수
- 빨강과 파랑의 픽셀 면적 비교

현재 비전은 일반 사진을 이해하는 모델이 아니다.
작은 RGB 격자에서 연결된 색상 픽셀, 경계 상자, 정확한 픽셀 수를 측정하는 검증형 MVP다.
이 제한은 실수로 일반 이미지 인식 능력처럼 보이지 않도록 화면에도 표시된다.

## 결과 읽는 법

- `검증된 결론`: typed 연산자 프로그램이 목표를 만들었고 proof replay도 통과했다.
- `진행 조건 미충족`: 목표가 가능하다는 뜻이 아니라, 막힌 조건 때문에 준비되지 않았음이 증명됐다.
- `미증명`: 현재 사실과 연산자로 목표에 도달하지 못했다. 거짓이라고 단정한 것은 아니다.
- `검증 과정 보기`: 관찰 사실, 적용된 정리, 최종 결론을 순서대로 보여 준다.
- `연구용 기술 정보 보기`: expansion, grounding trace, proof metadata 등 개발자용 JSON을 보여 준다.

## 개인정보와 네트워크

쉬운 시작 화면은 `127.0.0.1`에만 열리는 로컬 서버다.
입력은 기본적으로 이 PC의 SemOp 프로세스에서 처리되며 외부 LLM이나 웹 API로 보내지 않는다.
향후 외부 judge나 데이터 수집 기능을 연결할 때는 별도 설정과 명시적 동의 경계를 유지해야 한다.
쉬운 시작의 로컬 경험 수집이 켜져 있으면 미증명·파싱 실패 raw 입력은 위 SQLite
검토 큐에 남는다. 화면과 터미널이 활성 상태를 함께 표시하며 `--no-experience`로
비활성화할 수 있다.
백그라운드에서 몰래 학습하지 않는다. 사람이 기대 결과를 승인하고
`검증 학습 시도`를 직접 누른 경우에만 로컬 rule gate를 실행한다.

## 화면 뒤에서 일어나는 일

초보자 화면은 별도 추론 엔진을 흉내 내지 않는다. 입력 형식만 단순하게 만들고 기존 typed 커널을 그대로 호출한다.

```text
브라우저의 쉬운 입력
  -> BeginnerReasoner 입력 검사
  -> TypedDomainRequest
  -> 언어·수학·비전 adapter
  -> 검증된 sparse controller의 연산자 순위 또는 결정론적 탐색
  -> OperatorKernel만 사실을 도출
  -> proof replay
  -> 한국어 요약과 검증 기록
  -> 주목할 실패/불확실 사례를 로컬 큐에 저장
  -> 사람 기대 결과 검토
  -> 네 분할 rule gate
  -> 통과한 규칙만 다음 실행에 활성화
```

관련 파일은 다음처럼 나뉜다.

- `src/semop/beginner.py`: 쉬운 입력 검사, typed 요청 변환, 결과 요약
- `src/semop/beginner_learning.py`: 승인 corpus의 규칙 검증과 controller checkpoint 로드
- `src/semop/beginner_web.py`: 로컬 HTTP 서버와 브라우저 화면
- `start_semop.bat`: Windows 더블클릭 실행기
- `tests/test_beginner_experience.py`: 세 도메인의 첫 사용 흐름과 HTTP 통합 테스트
- `tests/test_beginner_learning.py`: 승인 사례에서 규칙 발견·저장·재로드·변조 거절 테스트

초보자 기능만 빠르게 검증하려면 다음 명령을 사용한다.

```powershell
python -m pytest tests/test_beginner_experience.py tests/test_beginner_learning.py -q
```

전체 저장소 회귀 검증은 다음 명령으로 실행한다.

```powershell
python -m pytest
```

## 문제가 생기면

1. 터미널에 `Python 3.11 이상`이 설치되어 있는지 확인한다.
2. 브라우저가 열리지 않으면 터미널의 `브라우저 주소`를 직접 연다.
3. 입력 오류는 결과 카드의 안내대로 고친다.
4. 서버 오류는 터미널에 표시된 마지막 오류 줄을 개발자에게 전달한다.

전체 코드 지도는 [project_structure.md](project_structure.md), typed 코어의 신뢰 경계는
[typed_operator_core.md](typed_operator_core.md)에서 이어서 볼 수 있다.
