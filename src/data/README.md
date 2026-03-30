# src/data/ — 데이터 필터링 & 전처리

## 이 폴더가 하는 일
공개 선호도 데이터셋(UltraFeedback, HH-RLHF 등)에서 태스크별 샘플을 로드하고,
SimPO 학습에 맞는 형식 `(prompt, chosen, rejected)`으로 변환한다.

---

## 핵심 개념

### Preference Data (선호도 데이터)란?
SimPO/DPO 같은 선호도 학습에서 쓰는 데이터 형식이다.
```
prompt:   "Can you help me write a professional email?"
chosen:   "Of course! Here's a clear, professional email..."   ← 더 도움이 되는 응답
rejected: "Sure just write what you want lol"                  ← 덜 도움이 되는 응답
```
모델이 `chosen`을 `rejected`보다 더 높은 확률로 생성하도록 학습한다.

### 왜 Preference-Only 태스크인가?
JSON 포맷팅이나 수학처럼 정답이 명확한 태스크는 SFT가 더 직접적이다.
SimPO는 **정답이 없는 태스크**에서 진가를 발휘한다:

- **유용성 (Helpfulness)**: "어느 쪽이 더 도움이 되는가?" — 주관적
- **코딩 스타일 (Coding)**: "어느 쪽 코드 응답이 더 완성도 높은가?" — 선호도 기반
- **무해성 (Harmlessness)**: "어느 쪽이 더 안전한가?" — 맥락 의존적

이런 태스크에서 모델은 생성 품질 자체를 높이는 것이 아니라,
인간이 선호하는 응답 방식을 학습한다.

---

## 현재 사용 데이터셋 (최종 선택)

### 태스크 A: UltraFeedback Helpfulness
- `argilla/ultrafeedback-binarized-preferences-cleaned`
- 64,000개 이상의 일반 유용성 선호도 쌍
- GPT-4 레이블, Llama Instruct 방향성과 일치
- 필터링 없이 전체 사용

### 태스크 B: UltraFeedback Coding
- 동일 데이터셋에서 코딩 관련 프롬프트만 엄격하게 필터링
- 키워드 기준: `write a python`, `write python`, `in python`, `implement a`, `code in python`, `python script`, `python function` 등
- 필터링 결과: 약 2,015개 샘플 (train/test 분리)
- **왜 UltraFeedback을 고른 이유**: 동일한 GPT-4 레이블 → 두 태스크 간 레이블 방향성이 일관됨

---

## 시도했다가 실패한 데이터셋들

### JSON/Math 태스크 (실패 1)
- **방법**: JSON 포맷 맞추기, GSM8K 수학 풀기
- **실패 이유**: 정답이 있는 태스크에 선호도 학습 적용. 110개 샘플로는 task vector 생성 불가
- **교훈**: SFT가 더 적합한 태스크에 SimPO를 강제하면 안 됨

### HH-RLHF harmless split (실패 2)
- **방법**: `Anthropic/hh-rlhf` harmless split 태스크 B로 사용
- **실패 이유**: Anthropic 레이블(safety 우선) vs Llama Instruct(helpfulness 우선) 방향 충돌
- **증거**: accuracy 0.40~0.47 (0.5 이하), margin 음수 → chosen/rejected가 모델 입장에서 뒤집힘
- **교훈**: 레이블러의 가치관이 베이스 모델과 다르면 학습 불가

### PKU-SafeRLHF safety 데이터셋 (실패 3)
- **방법**: PKU-SafeRLHF에서 "safe" 레이블 기준으로 학습
- **실패 이유**: "safe=chosen"으로 레이블됐지만 실제로는 chosen이 유해 방법을 가르치는 경우 존재
- **증거**: epoch 5 accuracy 0.463
- **교훈**: GPT-4 레이블이라도 안전성 기준이 Llama와 다를 수 있음

### HH-RLHF helpful split + Helpfulness 조합 (실패 4)
- **방법**: 태스크 A=UltraFeedback helpfulness, 태스크 B=HH-RLHF helpful
- **실패 이유**: 두 태스크 모두 "유용한 응답" 방향 → task vector가 자연적으로 유사
- **증거**: Orthogonal Loss 적용 후에도 orth vs baseline margin 차이 +0.001 (노이즈 수준)
- **교훈**: Task Arithmetic 실험은 의미적으로 충분히 다른 태스크 조합 필요

---

## filters.py 주요 함수

| 함수 | 역할 |
|---|---|
| `load_ultrafeedback(n_samples)` | UltraFeedback에서 유용성 선호도 쌍 로드 |
| `load_coding(n_samples)` | UltraFeedback에서 코딩 관련 프롬프트만 키워드 필터링 |
| `load_hh_rlhf(split, n_samples)` | HH-RLHF helpful/harmless 쌍 로드 (실험 실패로 현재 미사용) |
| `format_for_simpo(example)` | `(prompt, chosen, rejected)` 형식으로 변환 |
| `train_test_split(dataset, ratio)` | 학습/테스트 분리 |

---

## 데이터 저장 형식
HuggingFace `DatasetDict` 형식으로 저장된다.
```python
DatasetDict({
    "train": Dataset(N개),
    "test":  Dataset(N개)
})
```
`data/processed/{task}/` 폴더에 Arrow 파일로 저장되며, `load_from_disk()`로 불러온다.

예시:
- `data/processed/helpfulness/` — UltraFeedback 유용성 데이터
- `data/processed/coding/` — UltraFeedback 코딩 서브셋
