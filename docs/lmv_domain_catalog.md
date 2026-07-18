# Language, Math, Vision Domain Catalog

## 목적

SemOp의 장기 목표는 언어, 수학, 비전마다 별도 추론기를 키우는 것이 아니다.
각 입력 adapter는 다른 감각 신호를 typed world로 바꾸되, 이후의 operator 선택, 실행,
proof replay는 하나의 공통 커널과 작은 controller가 담당해야 한다.

이번 리팩터링 전에는 공통 커널이 존재해도 다음 배선이 여러 파일에 하드코딩되어 있었다.

- `UnifiedTypedReasoner`가 도메인별 adapter를 직접 생성
- semantic codec이 `if language / elif math / else vision`으로 분기
- 실행 결과에서 어떤 입력 계약과 능력을 사용했는지 확인하기 어려움
- 새 도메인을 추가할 때 runtime, codec, 평가 코드를 따로 수정해야 함

`DomainCatalog`는 이 배선을 하나의 자가 기술형 계약으로 모은다.

## 현재 구조

```text
raw input
  -> DomainCatalog.require(domain)
  -> DomainSpec.adapter
  -> DomainInstance(registry, state, goals, grounding trace)
  -> shared OperatorKernel
  -> proof replay
  -> UnifiedTypedResult(domain capabilities, input contract, proof)

experience / benchmark artifact
  -> 같은 DomainSpec.semantic_codec
  -> canonical JSON
  -> digest-bound review and split audit
```

주요 타입은 다음과 같다.

- `DomainKind`: `language`, `math`, `vision`, `composed`
- `DomainSpec`: adapter, semantic codec, 입력 계약, 한국어 설명, capability 집합
- `DomainCatalog`: 중복을 거절하는 불변 domain registry
- `FunctionSemanticCodec`: 도메인 payload의 canonical encode/decode 경계
- `TypedDomainRequest`: catalog가 처리하는 raw 요청

기본 catalog는 `create_default_domain_catalog()`로 만든다. `UnifiedTypedReasoner`는
더 이상 도메인별 adapter 클래스를 직접 알지 않고 catalog만 조회한다. 기존
`adapters={...}` override API는 내부적으로 새 catalog를 반환하는 방식으로 유지된다.

## 공통 불변식

언어, 수학, 비전의 기본 `DomainSpec`은 모두 다음 capability를 선언한다.

- `typed_grounding`: raw 입력이 typed fact와 명시적 goal로 변환됨
- `semantic_codec`: 학습·평가 artifact가 canonical JSON으로 왕복됨
- `proof_replay`: 성공 결과가 operator program 재실행을 통과함

runtime은 adapter가 요청과 다른 도메인의 `DomainInstance`를 반환하면 즉시 거절한다.
따라서 잘못 연결된 adapter나 prebuilt instance가 다른 도메인의 사실을 몰래 주입할 수 없다.

domain별 추가 capability는 현재 다음과 같다.

| 도메인 | 추가 capability | 현재 입력 범위 |
|---|---|---|
| 언어 | `controlled_language`, `symbolic_logic` | 명시적 필요조건 문장과 제한된 Horn 논리 |
| 수학 | `exact_arithmetic`, `linear_equation` | 정확한 사칙연산·비교·일변수 일차방정식 |
| 비전 | `deterministic_raster`, `object_centric` | 검증된 symbolic scene과 작은 RGB 연결 요소 |
| 합성 | `cross_domain_composition` | 검증된 비전 측정→수학 조건→언어 결론 |

## 빠른 LMV 계약 게이트

다음 명령은 세 도메인을 한 번에 검사한다.

```powershell
python tools/eval/evaluate_lmv_core_gate.py --require-pass
```

도메인마다 다음 조건을 모두 만족해야 통과한다.

1. positive fixture의 목표가 증명되고 verified임
2. 한 조건만 다른 near-miss negative가 성공으로 보고되지 않음
3. 반환된 proof를 새 `OperatorKernel`로 다시 replay할 수 있음
4. semantic payload encode→decode→encode가 동일함
5. semantic request digest가 왕복 뒤에도 동일함
6. 결과에 catalog의 capability와 입력 계약이 보존됨

이 게이트는 빠른 구조 회귀 검사다. 자유 자연어 이해, 일반 사진 인식, 고등수학 전반의
semantic correctness를 입증하지 않는다. 보고서의 `claim_scope`에도 이 한계를 기록한다.

## 연구 근거

### 하나의 processor와 도메인별 경계

[A Generalist Neural Algorithmic Learner](https://arxiv.org/abs/2209.11142)는
서로 다른 알고리즘이 간단한 task별 encoder/decoder와 하나의 graph processor를 공유할 수
있음을 보였다. SemOp은 이 구도를 neural-only 실행기로 복제하지 않는다. adapter와 codec은
도메인별 경계로 두고, 공통 latent processor에 해당하는 controller는 typed operator action만
순위화한다. 실제 실행 권한은 공통 symbolic kernel에 남긴다.

### 프로그램 라이브러리와 작은 데이터

[DreamCoder](https://arxiv.org/abs/2006.08381)는 검증된 task program에서 반복 조각을
추상화하고, learned library와 recognition model을 wake-sleep 방식으로 함께 개선한다.
SemOp은 일반 PC 범위에 맞게 이를 bounded anti-unification과 MDL gate로 축소한다.
`DomainCatalog`의 공통 capability는 이후 서로 다른 도메인 trace를 같은 프로그램 corpus로
모을 때 어떤 실행 계약이 동일한지 명시하는 기준이다.

### 신경 제안과 symbolic 판정의 분리

[AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5)는 neural model이
보조 구성을 제안하고 symbolic deduction engine이 폐포와 증명을 담당하는 neuro-symbolic
구조를 사용한다. SemOp도 controller나 frontier judge가 사실을 직접 추가하지 못하게 하고,
catalog adapter의 typed grounding과 kernel replay 뒤의 결과만 성공으로 인정한다.

언어에서는 [Neural Symbolic Machines](https://aclanthology.org/P17-1003/)의 neural
programmer/symbolic computer 분리와
[Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers](https://aclanthology.org/2025.acl-industry.34/)의
문법 제약 결과를 따른다. 향후 작은 parser는 자유 텍스트를 직접 답하지 않고 catalog가
선언한 입력 문법과 type에 맞는 후보 program만 제안해야 한다.

### 객체 중심 비전

[Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)은
시각 입력을 교환 가능한 object slot 집합으로 바꾸어 새로운 조합에 일반화하는 방향을
보였다. SemOp의 현재 raster adapter는 학습형 slot model이 아니라 결정론적 연결 요소를
사용하지만, 출력 계약은 이미 객체와 관계 중심이다. 나중에 Slot Attention 계열 producer를
붙이더라도 slot은 먼저 `PROPOSED` grounding으로 들어오고 별도 검증이나 사람 review 없이는
증명 전제가 되지 않는다.

마지막으로 [Obtaining Faithful Interpretations from Compositional Neural Networks](https://arxiv.org/abs/2005.00724)는
모듈 구조 자체가 faithful explanation을 보장하지 않는다는 점을 보였다. 따라서 SemOp의
capability 표시는 neural 중간 출력을 설명이라고 부르는 근거가 아니다. 사용자에게 보여 주는
proof는 오직 typed executor의 실제 action trace에서 생성한다.

## 새 도메인 또는 adapter 추가 절차

1. raw 입력을 `DomainInstance`로 바꾸는 `TypedDomainAdapter`를 구현한다.
2. 학습·평가 저장이 필요하면 canonical `SemanticPayloadCodec`을 구현한다.
3. 입력 범위와 capability를 명시한 `DomainSpec`을 만든다.
4. `DomainCatalog.with_spec()` 또는 별도 catalog factory로 등록한다.
5. positive, near-miss negative, codec roundtrip, proof replay fixture를 추가한다.
6. controller에는 domain 이름 대신 capability, type, predicate, goal 구조를 제공한다.

`DomainKind` 자체는 현재 LMV와 composed MVP에 고정되어 있다. 임의 plugin domain 지원은
catalog의 신뢰 계약과 평가 fixture가 충분히 안정된 뒤 별도 변경으로 진행한다.

## 다음 리팩터링 순서

1. 완료: `controller_feature_profiles.md`의 identity audit와 typed-structure LODO 게이트
2. 언어 parser의 grammar-constrained typed candidate API 추가
3. 비전 neural object producer를 `PROPOSED` grounding 경계 뒤에 연결
4. full recurrent controller의 LMV operator family별 LODO 전이 게이트 강화
5. 검증 trace에서만 macro를 학습하고 catalog capability 조합별 회귀 검사

이 순서는 “세 기능이 한 화면에 있다”가 아니라 “세 기능이 같은 실행 계약과 학습 권한을
공유한다”는 목표를 단계적으로 더 강하게 만든다.
