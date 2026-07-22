# SemOp 쉬운 시작 가이드

이 문서는 코드를 처음 보는 사용자도 SemOp의 현재 기능을 직접 실행해 볼 수 있도록 설명한다.
Typed DSL, 연산자 이름, Python API를 몰라도 된다.

## 가장 쉬운 실행 방법

Windows에서 저장소 폴더의 `start_semop.bat`를 더블클릭한다.

1. 검은 터미널 창이 열린다.
2. 잠시 뒤 기본 브라우저에 `SemOp 쉬운 시작` 화면이 열린다.
3. 첫 화면의 큰 채팅칸에 질문을 적거나 예제 버튼을 누른다.
4. 이미지가 있으면 첨부하고, 보통은 자원 등급을 `자동`으로 둔다.
5. `연산자로 풀기` 버튼을 누른다.
6. 사용을 마치면 터미널 창에서 `Ctrl+C`를 누른다.

도메인을 직접 고르고 싶으면 `고급 직접 입력`을 펼친다. 기존 `코딩`, `언어 조건`,
`수학식`, `색상 비전` 폼이 그 안에 있다. 결과의 배지는 다음 의미다.

- `검증됨`: 입력과 연산자 proof를 결정론적으로 재실행했다.
- `입력 사실을 전제로 검증됨`: 논리는 맞지만 적어 준 현실 사실을 별도로 확인하지 않았다.
- `모델 해석은 미검증`: 계산은 실행했어도 자유 문장이나 사진의 모델 해석은 확인되지 않았다.
- `추가 정보 필요`: 안전한 typed 목표를 만들지 못했다.

브라우저가 자동으로 열리지 않으면 터미널에 표시된 주소를 브라우저 주소창에 입력한다.
기본 주소는 `http://127.0.0.1:8765/`이다. 해당 포트가 사용 중이면 SemOp이 다음 빈 포트를 고른다.

## 대화와 기억

기본 화면은 같은 브라우저의 대화 세션 ID를 기억한다. 질문과 답변 전체는 이 PC의
`beginner-experience.db`에 저장되므로 서버를 껐다 켜도 마지막 대화를 다시 보여 준다.
`새 대화`를 누르면 기존 기록을 지우지 않고 별도 세션을 시작한다.

답을 만들 때는 전체 기록을 매번 모델에 넣지 않는다. 가장 최근 8개 메시지, 최대 6,000자만
작업기억으로 전달한다. 이 문맥은 “그거”, “앞에서 말한 구조” 같은 표현을 잇기 위한 참고일 뿐,
typed proof의 관찰 사실로 사용되지 않는다. 일반 대화는 로컬 Qwen 모델이 설치된
`자동`, `economy`, `balanced` 등급에서 `모델 해석은 미검증` 상태로 답할 수 있다.
`symbolic` 등급이나 모델이 없는 환경에서는 검증 가능한 입력 밖의 대화에 추가 정보를 요청한다.

비슷한 과거 질문이 있으면 이전 user/assistant exchange를 최대 2개까지 자동으로 회상한다.
다른 session뿐 아니라 같은 session에서도 최근 8개 작업기억 밖으로 밀려난 exchange를 찾는다.
후보는 SQLite의 로컬 전체 기록 인덱스에서 찾은 뒤 기존 관련도 점수로 다시 정렬하므로, 이제
최신 400개보다 오래된 대화도 검색할 수 있다. 별도 모델이나 인터넷 연결은 사용하지 않는다.
최근 작업기억에 아직 들어 있는 turn은 중복 회상하지 않는다. 각 episode의 `scope`는
`cross_session` 또는 `same_session_archive`로 표시된다. 이 episode는
`해석·자원 정보 보기`의 `recalled_episodes`에서 확인할 수 있고, 과거 SemOp 답이 틀렸을
가능성을 포함한 미검증 경험이다. 따라서 proof 전제나 자동 학습 정답으로 사용되지 않는다.

이미지가 없고 score `0.55` 이상의 episode가 있으며 “뭐였지?”, “기억나?”처럼 과거 답을
그대로 묻는 요청이면 SemOp은 Qwen을 다시 부르지 않고 `RETRIEVE → SELECT → EXPLAIN`으로
답한다. `코드명만` 요청은 과거 문장의 명시적 코드명만 반환한다. 답은 계속 `best_effort`이며
기술 정보에는 `answer_source=episodic_context`와
`episode_context_used_by_retriever=true`가 남는다. 과거 내용을 바탕으로 새 설계·요약·추론을
요구하면 이 빠른 경로를 사용하지 않고 모델이 미검증 문맥으로 읽는다.

오래 기억시킬 내용은 채팅창 위의 `장기기억 관리`를 펼쳐 직접 적는다.
더 간단하게는 채팅에 명시적인 저장 명령을 입력해도 된다.

```text
기억해: 내 프로젝트 이름은 SemOp이고 Python을 사용해
기억해줘: 나는 구현과 검증 결과를 함께 보는 것을 선호해
```

`기억해:`와 `기억해줘:`는 의미 모델이 추측하는 명령이 아니다. 문장 맨 앞의 정확한 접두사를
로컬 write operator가 확인하고 뒤의 내용만 SQLite에 저장한다. 답변 기술 정보에는
`answer_source=explicit_memory_write`와 memory ID가 남고, 장기기억 목록도 즉시 갱신된다.
다음 질문부터 관련도가 기준을 넘으면 자동으로 회상한다.

관련도 `0.18` 이상인 저장 기억이 있고 “뭐였지?”, “기억나?”처럼 그 내용을 그대로 묻는
text-only 요청이면 Qwen을 다시 부르지 않고 `RETRIEVE → SELECT → EXPLAIN`으로 답한다.
`코드명만` 요청은 저장 문장의 코드명만 반환한다. 기술 정보에는
`answer_source=explicit_memory`, `memory_context_used_by_retriever=true`가 남고 상태는
`best_effort`다. 기억을 이용한 새 설계·비교·요약은 계속 모델 경로를 사용한다.
개인적인 직접 회상 질문인데 관련 기억과 episode가 모두 없으면 Qwen이 답을 추측하지 않는다.
`ASK(missing memory)`가 즉시 `기억해:` 저장 방법을 안내하고 상태는 `unsupported`, 기술 정보는
`answer_source=memory_context_missing`을 남긴다. 일반 지식 질문은 이 guard를 사용하지 않는다.

`auto` 또는 `balanced`에서는 빠른 단어·글자 검색이 하나도 찾지 못했을 때만 의미 검색을
시도한다. 먼저 질문과 저장 기억을 `답변 길이`, `답변 순서`, `GPU`, `프로젝트 목표`,
`테스트 일정` typed facet으로 분리하고, 같은 facet 안에서만 multilingual-E5-small이 후보를
정렬한다. score `0.75` 미만은 사용하지 않으며 결과에는 `retrieval=semantic_e5`가 남는다.
`symbolic`에서는 이 모델을 부르지 않는다. 처음 한 번은 import와 CPU 모델 로드 때문에 현재
약 12.94초가 걸렸고, 로드 뒤 다섯 질문 replay는 합계 약 70.5ms였다.

답이 틀렸다면 바로 다음 메시지에서 올바른 답을 직접 가르칠 수 있다.

```text
나: 내 프로젝트 코드명은 뭐야?
SemOp: 아직 모르겠어.
나: 정정해: 별빛-27이야
```

`정정해:`와 `정정해줘:`는 바로 앞의 질문·답변과 교정 답을 하나의 local correction
record로 묶는다. 이후 공백·대소문자를 정규화한 같은 텍스트 질문에는 모델을 다시 부르지 않고
교정 답을 사용한다. 다시 정정하면 같은 질문의 기존 record를 최신 답으로 교체한다.
답변에는 `answer_source=user_correction_memory`와 correction ID가 남고 상태는
`best_effort`다. 사용자 교정은 유용한 장기 경험이지만 수학적 proof나 현실 증거는 아니기
때문이다. 이미지가 첨부된 질문에는 아직 이 text-only 교정을 적용하지 않는다.

같은 교정 답이 서로 다른 질문 표현 3개에서 반복되면 SemOp은 별도 모델 학습 없이
`CorrectionPrototype`을 파생한다. 처음 보는 네 번째 표현이 저장 예시와 충분히 비슷하면 이
prototype의 답을 사용할 수 있다. 화면에는 `반복 교정에서 일반화`라고 표시되며 support,
similarity, 근거 correction ID가 기술 정보에 남는다. 두 예시만 있을 때는 일반화하지 않는다.
prototype도 SQLite 교정 records에서 매번 다시 만들기 때문에 원본 교정이 바뀌면 함께 바뀌며,
여전히 `best_effort`이고 proof 전제가 아니다.

SemOp은 현재 질문과 관련도가 있는 기억 최대 3개만 불러온다. 먼저 별도 모델 없이 단어와
한국어 글자 조각의 겹침을 사용하므로 기존 관련 질문은 CPU에서 즉시 동작한다. 위 다섯 facet의
lexical miss에서만 선택적으로 E5를 지연 로드한다. 검색된 내용은 답변의
`해석·자원 정보 보기`에 표시된다. 잘못 저장한 기억은 목록의 `지우기`로 SQLite에서도
즉시 삭제한다. 대화 내용을 모델이 마음대로 검증된 장기 사실로 옮기지는 않는다. 대화 episode
자동 회상은 연속성을 위한 참고 기능이며, 명시적 장기기억 저장이나 의미 지식 승격과 구분된다.

저장을 원하지 않으면 `--no-experience`를 사용한다. 이 옵션은 학습 후보와 의미 trace뿐 아니라
대화 기록과 명시적 장기기억도 함께 끈다. 현재는 대화 내용을 몰래 학습하거나 장기 지식으로
자동 승격하지 않는다. 직접 저장한 장기기억도 개인화 참고 문맥이며 현실 증거나 proof 전제가 아니다.

### 기억 지속성 직접 확인

모델을 전혀 불러오지 않는 빠른 검사는 다음 한 줄이다.

```powershell
python tools\eval\evaluate_memory_continuity.py
```

기본 검사는 100턴 대화의 재시작 복원, 최근 8개 작업기억, 과거 세션 episode 회상,
작업기억 밖 같은 세션 episode 회상과 최근 turn 중복 차단, 관련 명시 기억 top-1 회상,
무관 질문 오회상, 3-shot 교정 통합의 held-out 표현, 12개 장기 작업 revision과 완료 불변성을
함께 확인한다. `correction_consolidation`에는 2-shot 미매치, 3-shot 매치, support, score와
재시작 지속성이 기록된다. 실제 로컬 Qwen이
기억을 답에 사용하는지도 비교하려면 검토된 모델이 설치된 `.venv312`에서 실행한다.

최근 400개보다 오래된 episode까지 직접 확인하려면 별도 장기 기록 gate를 켠다.

```powershell
python tools\eval\evaluate_memory_continuity.py `
  --long-history-exchanges 5000 `
  --output artifacts\probes\long_history_fts.json
```

현재 5,000 exchange 실측은 전체 인덱싱 성공, 기존 DB backfill 약 101.5ms,
cross-session 조회 약 20.6ms, same-session archive 조회 약 24.0ms, DB 약 5.01MB였다.
이는 오래된 표식 회상의 단일 CPU gate이며 장기간 자연 대화의 의미 정확도 전체를 뜻하지 않는다.

저장한 기억의 한국어 바꿔 말하기 A/B는 다음처럼 별도로 실행한다.

```powershell
python tools\eval\evaluate_memory_continuity.py `
  --semantic-paraphrase-ab `
  --output artifacts\probes\semantic_memory_e5_ab.json
```

현재 결과는 lexical `0/5`, typed facet+E5 `5/5`, 무관 질문 오회상 `0/3`이다. 모델 첫 로드를
포함한 cold 5건은 약 12.94초, warm 5건은 약 70.5ms다. 이는 지정된 다섯 memory facet을
검사하는 smoke이며 이름·관계·취향 전체를 자유롭게 이해한다는 의미는 아니다.

```powershell
.\.venv312\Scripts\python.exe tools\eval\evaluate_memory_continuity.py `
  --semantic-tier balanced
```

이 A/B는 같은 질문을 기억 없이 한 번, 명시적 기억과 함께 한 번 보낸다. 결과의
`semantic_ab.measured_gain`은 기억이 있을 때만 기대 표식이 답에 나타났는지 보여 준다.
`episodic_ab`는 고유 코드명을 과거 session의 exchange에만 넣고, 새 session의 답이
`recalled_episodes` 제공 전후에 실제로 달라지는지 같은 방식으로 기록한다.
`long_session_ab`는 한 session의 첫 exchange를 네 번의 다른 대화로 작업기억 밖에 밀어낸 뒤,
같은 session archive 제공 전후의 답을 비교한다. `generated_tokens`, `hit_token_limit`,
`repair_attempts`도 함께 기록한다. 특정 실험만 빠르게 다시 돌리려면 다음처럼 선택한다.

```powershell
.\.venv312\Scripts\python.exe tools\eval\evaluate_memory_continuity.py `
  --semantic-tier balanced --model-ab long-session
```
`task_ab`는 오래된 revision과 최신 revision에 서로 다른 다음 행동을 저장한 뒤, 최신 행동과
작업 요약만 반환하는지 확인한다. 이 직접 조회는 `RETRIEVE → SELECT` operator로 실행되어
Qwen을 부르지 않는다. 작업이 선택되지 않았으면 임의 행동을 만들지 않고 먼저 작업을 선택해
달라고 안내한다. 현재 실측은 미선택 안내 약 0.162ms, 다음 행동 약 0.131ms, 최신 revision의
진행·결정·다음 행동 요약 약 0.073ms다.
`task_auto_resume`는 활성 작업 두 개를 만든 뒤 모호한 요청은 선택하지 않고, 작업 이름이 든
새 요청만 최신 revision으로 자동 연결하는지 확인한다.
이는 저장된 개인화 사실의 회상 검사이며 일반 지식 정확도나 신경망 가중치 학습 성과는 아니다.

## 여러 날 작업 이어가기

채팅창 위의 `장기 작업 관리`를 펼치면 한 번의 대화보다 오래 걸리는 일을 이어갈 수 있다.

1. `작업 이름`에는 알아보기 쉬운 이름을 적는다.
2. `완료 조건`에는 무엇이 되면 끝인지 적고 `작업 만들기`를 누른다.
3. 작업을 마칠 때마다 `현재까지 한 일`, `확정한 결정`, `다음 행동`을 적고
   `새 checkpoint 저장`을 누른다.
4. 다음 실행에서 목록의 `이어가기`를 누르거나 채팅에 `SemOp 언어 모델 학습 작업 계속하자`처럼
   작업 이름을 적으면 그 작업 하나가 현재 채팅에 연결된다.
5. 완료 조건을 충족하면 `완료로 고정`을 누른다.

각 저장은 기존 내용을 덮어쓰지 않고 새 revision을 추가한다. 완료된 작업은 수정할 수 없고,
이전 revision도 SQLite에 남는다. `채팅에서만 해제`를 누르면 기록은 보존하면서 현재 질문에서
제외할 수 있다. 여러 작업을 한꺼번에 넣어 문맥을 흐리지 않도록 쉬운 화면은 한 번에 하나만
선택한다.

같은 생명주기를 채팅만으로 실행할 수도 있다.

```text
작업 만들기: SemOp 로컬 챗봇 | 완료 조건: 언어·수학·비전 통합 검증 통과
작업 기록: 교정 기억 평가 완료 | 다음: student 후보 봉인 평가
작업 완료: 전체 회귀 통과 | 결정: 완료 revision은 immutable
```

`작업 만들기:`는 제목과 완료 조건으로 revision 1을 만들고 바로 현재 작업으로 선택한다.
`작업 완료:`는 선택되었거나 고유하게 검색된 활성 작업에 마지막 revision을 추가하고 다음 행동을
비운 뒤 수정할 수 없게 고정한다. 세 명령 모두 사용자가 직접 적은 내용만 저장한다.

활성 작업이 하나뿐이면 `장기 작업 계속하자`만으로도 자동 재개한다. 여러 작업이 있으면
제목·완료 조건과 가장 잘 맞는 작업이 하나일 때만 선택한다. 점수가 같거나 너무 낮으면 임의로
고르지 않고 작업 이름을 알려 달라고 한다. 자동 선택된 작업은 브라우저의 현재 작업에도
표시되며 `task_selection.mode=automatic`, score, 선택 이유와 revision이 기술 정보에 남는다.

작업을 선택한 뒤에는 다음처럼 평범하게 물어볼 수 있다.

```text
어디까지 했지?
결정이 뭐였지?
다음 행동은?
SemOp 로컬 챗봇 작업 어디까지 했지?
```

앞의 세 문장은 현재 선택된 작업을 사용한다. 마지막 문장은 작업 이름으로 활성 작업을 고른 뒤
최신 revision을 읽으므로 브라우저나 서버를 다시 시작한 뒤에도 쓸 수 있다. `어디까지 했지?`는
제목·revision·상태·목표·진행·결정·다음 행동을, `결정이 뭐였지?`와 `다음 행동은?`은 각각
해당 항목만 돌려준다. 저장 checkpoint를 그대로 보여 주는 `best_effort` 응답이며 Qwen은
호출되지 않는다.

작업을 선택했거나 활성 작업이 하나뿐이면 채팅에서 checkpoint도 저장할 수 있다.

`작업 기록:`에서는 첫 번째 `|` 앞의 라벨 없는 내용이 진행 상황이다.

`진행:`, `다음:`, `결정:`을 명시해도 되고,
결정과 다음 행동 여러 개는 세미콜론으로 나눈다. 적은 필드만 갱신하므로 진행만 기록하면 기존
결정과 다음 행동은 유지된다. 새 결정은 기존 결정 뒤에 중복 없이 덧붙고, 새 다음 행동은 이전
목록을 교체한다. 답에는 `task_checkpoint_write`, task ID와 새 revision이 남는다. 이는 사용자가
명시적으로 확인한 write operator이며 모델이 대화를 자동 요약해 저장하는 기능이 아니다.

선택한 작업의 목표와 진행 상황은 로컬 의미 모델이 “어디부터 계속할지” 이해하는 참고
문맥이다. 수학 정답, 현실 사실, 코드 테스트 성공을 증명하는 자료로는 사용할 수 없다.
모델이 작업 상태를 자동 수정하거나 완료시키지도 않는다. `--no-experience`를 사용하면
대화·명시적 장기기억과 함께 장기 작업 저장도 꺼진다.

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

쉬운 시작 실행은 미증명 문제와 파싱 실패 같은 typed 학습 후보와, Qwen이 실제로 사용된
의미 변환 trace, 대화 세션을 다음 로컬 SQLite에 저장한다.

```text
artifacts/experience/beginner-experience.db
```

결정론적으로 검증된 정상 성공을 학습 후보로 전부 저장하지 않으며, 큐의 후보는 사람 검토 전에는
규칙이나 모델에 반영되지 않는다. 입력은 외부 서비스로 전송되지 않는다. 저장을 원하지 않으면
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

통합 채팅에서 로컬 Qwen이 해석을 제안한 경우에는 결과 아래에 `해석이 맞아`와
`해석이 틀려` 버튼이 별도로 나타난다. 여기서는 원문이 typed 식·조건으로 정확하게
옮겨졌는지를 검토한다. `맞아`는 typed proof replay까지 성공한 경우에만 positive가 되며,
`틀려`는 hard negative 후보가 된다.

검토하지 않고 페이지를 닫아도 텍스트 trace는 첫 화면의 `모델 해석 검토함`에 남는다.
검토함은 최근 항목을 전부 나열하지 않고 한 번에 최대 8개의 replay 후보를 보여 준다.
각 카드의 `우선순위`와 `선정 이유`는 다음 네 가지 저비용 신호로 계산한다.

- proof replay 또는 보존된 PNG/JPEG 근거가 있어 positive로 검토할 수 있는가
- 미검증·미지원 결과라서 hard negative 교정 가치가 있는가
- 이미 검토한 질문과 다른 단어·글자 구조인가
- 아직 positive가 부족한 도메인이나 `train/validation/sealed` 역할인가

점수는 모델 예측이나 정답 판정이 아니라 사람이 먼저 볼 순서다. 목록을 새로고침해도
SQLite review, 모델 가중치, typed fact는 바뀌지 않는다. 질문과 typed JSON을 직접 읽고
확인 체크를 한 뒤에만 승인하거나 거절할 수 있다.

사람이 승인한 서로 다른 문장 3개 이상이 같은 typed 의미로 이어지면 `통합 prototype` 수가
올라간다. 예를 들어 같은 뺄셈을 표현한 여러 문장이 모이면, 처음 보는 비슷한 문장에 과거
세 사례를 analogy로 보여 줄 수 있다. prototype은 원본 trace ID와 review digest를 모두
보존하고 모델 가중치를 바꾸지 않는다. 새 해석은 여전히 `모델 해석은 미검증`이며 executor
검증을 새로 거쳐야 한다. 더하기와 빼기처럼 질문에 명시된 수학 연산이 충돌하면 유사도가
높아도 prototype을 건너뛴다.
텍스트에서 proof replay가 실패한 항목은 `맞아` 버튼이 비활성화된다. 첨부한 PNG/JPEG는
최대 16MiB까지 exact bytes와 SHA-256이 같은 로컬 SQLite에 저장되어 검토함에서 다시
보인다. 이미지의 `맞아`는 사람 기준 semantic label만 만들며 proof 성공으로 계산되지
않는다. 지원하지 않는 형식이거나 저장하지 못한 이미지는 검토 버튼이 비활성화된다.

검토한 것과 exact하게 같은 이미지 한 장을 나중에 다시 질문하면 로컬 Qwen은 최신 사람
승인·거절 기록을 참고 자료로 받는다. 이 기억은 답을 복사하거나 검증된 사실로 만드는 기능이
아니다. 같은 이미지를 다시 해석할 때 이전 교정을 잊지 않게 하는 작은 episodic memory이며,
다른 사진에도 일반화된다는 뜻은 아니다.

언어·수학 모델 해석도 같은 방식으로 동작한다. Unicode, 대소문자와 연속 공백을 정리한 뒤
동일한 질문이면 최신 검토를 참고한다. 표현만 비슷한 다른 문장은 아직 자동으로 연결하지
않는다.

승인된 텍스트 trace의 역할별 학습 준비도는 다음 명령으로 확인할 수 있다.

```powershell
semop-student-cycle status
```

기본 export는 정규화된 질문·workspace·이미지 digest로 미리 정해진 `train` 80%만 포함한다.
모델의 해석이 달라져도 같은 입력은 같은 역할에 유지된다. `validation` 10%와
`sealed` 10%는 학생 후보 선택과 마지막 평가에 남겨 두므로 학습 파일과 섞이지 않는다.
이미지가 있는 trace와 검토 픽셀은 로컬 journal에 남지만 현재 텍스트 LoRA corpus에는
들어가지 않는다. 질문 문장만 보고 비전 답을 외우는 잘못된 지름길을 막기 위한 현재 범위다.
화면의 준비도에 train과 sealed positive가 모두 표시되면 아래 한 명령으로 export, 후보
학습, 봉인 평가와 통과 후보 stage를 이어서 실행할 수 있다.

```powershell
semop-student-cycle run --confirm --device cuda
```

이 명령의 봉인 평가는 합격 후보를 stage할 뿐 모델 활성화는 하지 않는다.

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

1.46M recurrent compact candidate는 자동으로 기본값이 되지 않는다. 합성 구조 평가 결과를
직접 확인한 뒤 아래처럼 all-pass 보고서를 명시해야 로드된다.

```powershell
python -m semop.beginner_web `
  --controller-evaluation-report artifacts/eval/raw_compact_controller_seed31.json
```

보고서의 `gates.all_passed`, artifact SHA-256, parameter count가 실제 `.npz`와 모두 맞아야
시작된다. 이 로컬 candidate는 언어 3개 trace에서 학습해 held-out 수학·픽셀 expansion을
각각 6에서 2로 줄인 연구 결과이며, 사람 검토 20/100-shot 승격을 대신하지 않는다.

checkpoint의 artifact hash가 다르거나 지원하지 않는 policy 종류면 조용히 무시하지 않고
시작을 중단한다. 잘못된 작은 모델보다 검증 가능한 탐색을 선택하기 위한 동작이다.

## 코딩

통합 채팅에서는 “Python 리스트 컴프리헨션과 for 루프를 예시로 비교해줘” 같은 일반 코딩
질문도 할 수 있다. 이 답은 로컬 모델 후보이므로 `best_effort`다. Python 예시를 요청하면
SemOp이 AST 문법과 요청한 비교 구조를 확인하지만, 아직 그 코드를 실행하거나 테스트 통과를
주장하지 않는다.

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
x² - 5x + 6 = 0
x^2 - 2 = 0
7 < 10
```

지원되는 범위에서는 숫자와 연산을 typed term으로 바꾸고 정확한 계산 연산자를 실행한다.
성공 결과는 전체 계산 프로그램을 처음부터 다시 재생해 확인한다.
이차방정식은 판별식을 계산한 뒤 유리근, 중근, `sqrt(...)` 형태의 정확한 근호 실수해 또는
실수해 없음까지 증명한다. `x²`, `×`, `÷`, 유니코드 빼기 기호도 일반 수식 기호로
정규화한다. 현재는 복소근의 구체적 값, 삼차 이상 방정식, 변수가 분모에 있는 식은 범위
밖이며 그럴듯한 답을 만들지 않고 입력 오류 또는 미증명으로 표시한다.

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
- `best_effort`: 작은 로컬 모델의 대화 답변 또는 미검증 의미 해석이다. typed 해석이 실패해도
  유용한 대화 답은 보존하지만 사실 정확성이나 proof 성공을 뜻하지 않는다.
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
  -> 최근 대화 작업기억 조회(참고 문맥만)
  -> cross-session 및 작업기억 밖 same-session episode 최대 2개 조회(미검증 경험)
  -> 현재 질문과 관련된 명시적 장기기억 최대 3개 조회
  -> 사용자가 선택한 장기 작업 checkpoint 하나 조회
  -> PromptCompiler 또는 BeginnerReasoner 입력 검사
  -> TypedDomainRequest
  -> 언어·수학·비전 adapter
  -> 검증된 sparse controller의 연산자 순위 또는 결정론적 탐색
  -> OperatorKernel만 사실을 도출
  -> proof replay
  -> 한국어 요약과 검증 기록
  -> 전체 대화를 로컬 episodic memory에 저장
  -> 주목할 실패/불확실 사례를 로컬 큐에 저장
  -> 사람 기대 결과 검토
  -> 네 분할 rule gate
  -> 통과한 규칙만 다음 실행에 활성화
```

관련 파일은 다음처럼 나뉜다.

- `src/semop/beginner.py`: 쉬운 입력 검사, typed 요청 변환, 결과 요약
- `src/semop/beginner_learning.py`: 승인 corpus의 규칙 검증과 controller checkpoint 로드
- `src/semop/beginner_web.py`: 로컬 HTTP 서버와 브라우저 화면
- `src/semop/conversation_memory.py`: SQLite 대화 세션, bounded 최근 문맥, 전체-history FTS 후보와 cross/same-session episode 회상, 명시적 장기기억 검색·삭제
- `src/semop/task_memory.py`: append-only 장기 작업 revision, 완료 고정, 채팅용 typed context
- `src/semop/semantic_replay.py`: pending 의미 trace의 bounded·설명 가능한 검토 우선순위
- `src/semop/semantic_memory.py`: exact 검토 기억과 support 3 이상의 text semantic prototype
- `start_semop.bat`: Windows 더블클릭 실행기
- `tests/test_beginner_experience.py`: 세 도메인의 첫 사용 흐름과 HTTP 통합 테스트
- `tests/test_beginner_learning.py`: 승인 사례에서 규칙 발견·저장·재로드·변조 거절 테스트

초보자 기능만 빠르게 검증하려면 다음 명령을 사용한다.

```powershell
python -m pytest tests/test_beginner_experience.py tests/test_beginner_learning.py tests/test_conversation_memory.py tests/test_task_checkpoint_memory.py tests/test_prompt_first_assistant.py -q
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
