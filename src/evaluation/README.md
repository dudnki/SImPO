# src/evaluation/ — 평가 지표

## 이 폴더가 하는 일
Preference-only 태스크에서 모델 정렬 품질을 측정한다.
SimPO margin을 보조 지표로, **HumanEval pass@1을 주요 지표**로 사용한다.

---

## 평가 지표 변경 이유

### 처음에는 SimPO Margin을 썼다
학습 objective와 동일한 기준이어서 자연스럽다고 판단:
```
margin = β * (logP(chosen)/|chosen| - logP(rejected)/|rejected|) - γ
```
- margin > 0: 모델이 chosen을 rejected보다 선호 (올바른 방향)
- 평균이 높을수록, 양수 비율이 높을수록 좋은 모델

### 실험 결과: SimPO Margin은 충분히 민감하지 않았다
Task Arithmetic (λ=1.3) 결과:
```
baseline 합성: helpfulness 0.444, coding 0.471
orth 합성:     helpfulness 0.446, coding 0.479
차이:                        +0.002           +0.008
```
→ 차이가 너무 미미해서 Orthogonal Loss의 효과를 구분할 수 없음

**왜 SimPO margin이 민감하지 않은가**:
- 간섭 공식: `||τ_A|| × ||τ_B|| × cos(θ)` — cosine만 줄인다고 효과가 보장되지 않음
- Helpfulness norm 1.11 vs Coding norm 0.65 → helpfulness 벡터가 coding 성능을 지배
- Teacher forcing 기반 log prob는 실제 생성 품질과 다를 수 있음

### 해결책: HumanEval pass@1로 전환
- 코드를 실제로 실행해서 정확도를 측정 (functional correctness)
- 표준 벤치마크 — 외부 비교 가능
- 실제 생성 품질을 직접 측정 → 더 민감하고 해석하기 쉬움

---

## 핵심 지표

### HumanEval pass@1 (주 지표)
164개의 Python 코딩 문제에 대해 모델이 1번의 시도로 통과하는 비율.
```
pass@1 = 정답 통과 문제 수 / 164
```
- 실제 코드 실행으로 테스트 케이스 통과 여부 확인
- 랜덤 샘플링 없이 greedy decoding (temperature=1.0, do_sample=False)

**3가지 비교 조건**:
```
coding_only:      coding task vector만 적용 (λ=1.0)
baseline_merged:  helpfulness + coding baseline 합성 (λ=1.0)
orth_merged:      helpfulness + coding orth 합성 (λ=1.0)
```
coding_only 대비 merged에서 pass@1이 더 높으면 → Task Arithmetic 합성 성공
orth_merged > baseline_merged이면 → Orthogonal Loss 효과 확인

### SimPO Margin (보조 지표)
```
margin = β * (logP(chosen)/|chosen| - logP(rejected)/|rejected|) - γ
```
- 학습 objective와 동일한 기준 → 학습이 제대로 됐는지 확인용
- Task Arithmetic 효과의 방향성 확인에 사용

---

## metrics.py 주요 함수

| 함수 | 역할 |
|---|---|
| `compute_avg_logp(model, tokenizer, prompt, response)` | 응답 토큰들의 평균 log probability 계산 |
| `compute_simpo_margins(model, tokenizer, dataset, beta, gamma)` | SimPO reward margin 일괄 계산 |

### SimPO Margin 계산 방식
```python
logp_c = avg_log_prob(model, prompt, chosen)    # 길이 정규화
logp_r = avg_log_prob(model, prompt, rejected)
margin = beta * (logp_c - logp_r) - gamma
```
SimPO loss = softplus(-margin) = -log σ(margin)

---

## 현재 실험 결과 (Task Arithmetic, λ=1.3)

```
| 조건            | helpfulness margin | coding margin |
|---|---|---|
| 베이스 (λ=0)   | 0.063              | 0.722         |
| baseline 합성  | 0.444              | 0.471         |
| orth 합성      | 0.446              | 0.479         |
```

→ SimPO margin 차이는 미미 (~0.008). HumanEval pass@1로 재측정 중.

**Cosine Similarity (Orthogonal Loss 효과 확인)**:
```
helpfulness vs coding baseline: 0.3018
helpfulness vs coding orth:     0.0817  (73% 감소)
```
→ 기하학적으로 Orthogonal Loss가 작동함은 확인됨. 실제 성능 영향을 HumanEval로 검증 중.

---

## eval_humaneval.py 구성

`scripts/eval_humaneval.py` 참조. 주요 흐름:

1. fp16 베이스 모델 로드 (`unsloth/Llama-3.2-3B-Instruct`)
2. task vector 적용 (λ=0.0 베이스 측정 후 λ=lambda_val 측정)
3. 164개 HumanEval 문제에 대해 greedy decoding
4. `human_eval.evaluation.evaluate_functional_correctness`로 통과율 계산
5. MLflow에 `pass_at_1_base`, `pass_at_1_merged`, `pass_at_1_delta` 기록

**주의**: human_eval 패키지는 miniconda 환경에 설치됨.
`sys.path.append` (insert가 아닌 append)로 venv 패키지 우선순위 유지.
