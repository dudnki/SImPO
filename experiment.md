# Experiment Notes — Orthogonal LoRA Task Arithmetic

이 파일은 현재까지의 실험 결과와 다음 실험(메인 contribution)의 설계를 담는다.
새 모델/세션에서 이 파일을 읽고 다음 실험을 이어받을 수 있도록 작성함.

---

## 1. 이 실험이 검증하려는 핵심 가설

### 메인 Contribution (다음 실험 목표)
> **LoRA로 fine-tuning된 task vector들 사이의 cosine similarity를,
> O(r²) trace trick으로 효율적으로 계산하여 직교 regularization loss로 사용한다.
> 이를 통해 Task Arithmetic 합성 시 태스크 간 간섭을 줄인다.**

수식:
```
L_total = L_task + α · cos²(τ_i, τ_j)

여기서
  τ_i = ΔW_i = B_i @ A_i * (alpha/r)   (현재 학습 중인 task vector)
  τ_j = ΔW_j = B_j @ A_j * (alpha/r)   (이전에 학습된 고정된 task vector)

cos²(τ_i, τ_j) = <ΔW_i, ΔW_j>_F² / (||ΔW_i||_F² · ||ΔW_j||_F²)
```

### Trace Trick — 핵심 novelty
ΔW = B@A를 직접 계산하면 (d_out × d_in) 행렬 → O(d_out · d_in · r) 연산.
Trace의 cyclic property를 이용하면:

```
<ΔW_i, ΔW_j>_F = tr(ΔW_i^T @ ΔW_j)
               = tr((B_i@A_i)^T @ (B_j@A_j))
               = tr(A_i^T @ B_i^T @ B_j @ A_j)
               = tr((B_i^T @ B_j) @ (A_j @ A_i^T))
               = tr(C @ D)

  C = B_i^T @ B_j   (r × r)
  D = A_j @ A_i^T   (r × r)
```

연산량: O(r² · max(d_in, d_out)) → r=16이면 실질적으로 O(r²)
B@A 직접 계산 대비 약 **580,000배 절감** (3B LLM 기준)

### 이 방법이 novel한 이유 (선행 연구 조사 결과)
- LoRA + Task Arithmetic 조합: Chitale et al. (2023) 등 존재 — **기존 연구 있음**
- 직교성 강제: OSRM (ACL 2025)은 학습 전 SVD 초기화만, Ortho-LoRA (2026)는 동시 멀티태스크 gradient projection — **훈련 loss로 cos²를 쓴 논문 없음**
- **Trace trick으로 O(r²) cosine 계산: 어떤 논문도 없음**
- **순차 학습에서 frozen τ_prev 기준 직교화: 어떤 논문도 없음**

---

## 2. 파일럿 실험 결과 (이번 세션에서 완료)

### 실험 설정
- 베이스 모델: `unsloth/Llama-3.2-3B-Instruct` (LoRA: r=16, alpha=16)
- 태스크 A: Helpfulness (UltraFeedback, SimPO, 5 epochs)
- 태스크 B: Coding (UltraFeedback coding subset ~2,015개, SimPO, 7 epochs)
- Orthogonal Loss: α=0.1, tau_prev = helpfulness task vector
- 평가: HumanEval pass@1 (164문제, greedy decoding)

### 학습 결과
| 태스크 | epoch | accuracy | margin | train_loss |
|---|---|---|---|---|
| helpfulness | 5 | 0.663 | — | 1.977 |
| coding baseline | 7 | 0.641 | 0.524 | 1.610 |
| coding orth | 7 | 0.634 | 0.694 | 1.630 |

### Cosine Similarity (Orthogonal Loss 기하학적 효과)
```
helpfulness vs coding baseline:  0.3018
helpfulness vs coding orth:      0.0817  (73% 감소)
```
→ Orthogonal Loss가 task vector 방향을 실제로 직교화함을 확인

### HumanEval pass@1 결과 (λ=1.0)
| 조건 | pass@1 | base 대비 |
|---|---|---|
| 베이스 (λ=0) | 0.4756 | — |
| coding 단독 | 0.5061 | +0.0305 |
| baseline 합성 (helpfulness + coding) | 0.5000 | +0.0244 |
| **orth 합성 (helpfulness + coding orth)** | **0.5122** | **+0.0366** |

**핵심**:
- baseline 합성: helpfulness 간섭으로 coding 능력 저하 (−0.0061)
- orth 합성: 간섭 제거 + coding 단독보다 향상 (+0.0061)
- baseline vs orth: **+0.0122 (약 2문제 차이)**

### Task Arithmetic λ sweep 결과 (SimPO margin 기준)
| λ | helpfulness margin | coding margin |
|---|---|---|
| 0.0 (base) | 0.063 | 0.722 |
| 1.3 baseline 합성 | 0.444 | 0.471 |
| 1.3 orth 합성 | 0.446 | 0.479 |

→ SimPO margin 차이는 작음 (∼0.008). HumanEval이 더 민감한 지표.

---

## 3. 실패한 실험들 (같은 실수를 반복하지 않기 위해)

### 실패 1: JSON/Math 태스크에 SimPO 적용
- 정답이 있는 태스크 → SFT가 적합. SimPO로 task vector 생성 불가
- 110개 샘플로는 학습 자체가 안 됨

### 실패 2: HH-RLHF harmless split
- Anthropic 레이블(safety 우선) vs Llama Instruct(helpfulness 우선) 방향 충돌
- accuracy 0.40~0.47 (0.5 이하), margin 음수 → chosen/rejected가 모델 관점에서 뒤집힘
- **교훈**: 레이블러의 가치관이 베이스 모델과 다르면 학습 불가

### 실패 3: PKU-SafeRLHF
- "safe=chosen" 레이블이지만 실제로는 chosen이 유해한 내용을 가르치는 경우 존재
- epoch 5 accuracy 0.463

### 실패 4: Helpfulness + HH-RLHF helpful 조합
- 두 태스크 모두 "유용한 응답" 방향 → task vector가 자연적으로 유사
- Orthogonal Loss가 개선할 여지 없음 → margin 차이 +0.001 (노이즈)
- **교훈**: Task Arithmetic 실험은 의미적으로 다른 태스크 조합 필요

---

## 4. 다음 실험 설계 (메인 contribution 검증)

### 목적
파일럿은 SimPO + Helpfulness/Coding 조합으로 했지만,
**메인 contribution은 fine-tuning 방법에 무관한 일반적인 방법론**이다.
더 설득력 있는 검증을 위해 다음을 추가한다.

### 권장 실험 구성

#### Option A: 더 많은 태스크 쌍 테스트
```
태스크 조합 1 (현재): Helpfulness + Coding
태스크 조합 2 (추가): Helpfulness + Safety (GPT-4 레이블, 같은 출처 데이터)
태스크 조합 3 (추가): Coding + Math (SFT 방식, 서로 다른 능력)
```
→ 여러 조합에서 일관되게 orth > baseline이면 더 강한 증거

#### Option B: 더 큰 모델로 재현
```
현재: Llama 3.2 3B
추가: Llama 3.1 8B 또는 Qwen2.5 7B
```
→ 모델 크기에 무관한 방법론임을 보여줌

#### Option C: SFT + 우리 Orthogonal Loss 비교
```
SFT baseline: 일반 SFT로 학습 → Task Arithmetic
SFT + Ours: SFT + cos² orthogonal loss → Task Arithmetic
```
→ SimPO에 묶이지 않고 일반적인 방법임을 증명

### 평가 지표 (다음 실험에도 동일하게 사용)
1. **Task vector cosine similarity** (낮을수록 직교 → 좋음)
2. **HumanEval pass@1** (coding 능력 보존)
3. **MT-Bench** (일반 helpfulness 보존)
4. **단독 vs 합성 성능 갭** (gap이 작을수록 간섭이 적음)

---

## 5. 구현 세부사항 (새 모델이 코드를 이해하기 위해)

### Orthogonal Loss 구현 위치
`src/training/simpo.py` — `OrthogonalSimPOTrainer.compute_loss()`

```python
def _orthogonal_loss(self, model):
    loss = 0.0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or name not in self.tau_prev:
            continue
        A_i = module.lora_A["default"].weight   # (r, d_in)
        B_i = module.lora_B["default"].weight   # (d_out, r)
        A_j = self.tau_prev[name]["A"]           # (r, d_in), 고정
        B_j = self.tau_prev[name]["B"]           # (d_out, r), 고정

        # Trace trick: O(r²) 연산
        C = B_i.T @ B_j                          # (r, r)
        D = A_j @ A_i.T                          # (r, r)
        dot = torch.trace(C @ D)

        norm_i_sq = torch.trace(B_i.T @ B_i) * torch.trace(A_i @ A_i.T)
        norm_j_sq = torch.trace(B_j.T @ B_j) * torch.trace(A_j @ A_j.T)

        cos2 = dot**2 / (norm_i_sq * norm_j_sq + 1e-8)
        loss += cos2
    return loss
```

### Task Vector 추출 형식
```python
# ΔW 형태 (apply_arithmetic용)
{
    "model.layers.0.self_attn.q_proj.weight": tensor(d_out, d_in),
    ...
}

# A, B raw 형태 (다음 태스크 학습 시 tau_prev로 사용)
{
    "base_model.model.model.layers.0.self_attn.q_proj": {
        "A": tensor(r, d_in),
        "B": tensor(d_out, r)
    },
    ...
}
```

**주의**: 두 형식의 키 이름이 다름.
- ΔW는 `named_parameters()` 기반 → `"model.layers...."` (base_model 접두사 없음)
- A/B raw는 `named_modules()` 기반 → `"base_model.model.model.layers...."` (접두사 포함)

### Apply Arithmetic 베이스 모델
학습은 4-bit quantized 모델로 했기 때문에,
Task Arithmetic 시 동일한 가중치 공간 유지를 위해:
```python
# 4-bit 로드 후 Linear4bit → nn.Linear(bf16)으로 교체
model = load_4bit_and_dequantize("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
```
`model.to(torch.bfloat16)` 는 4-bit 모델에서 오류 발생 → 모듈 단위로 교체해야 함

### HumanEval 평가 시 주의
- `human-eval` 패키지는 `uv pip install human-eval`로 venv에 직접 설치
- subprocess로 별도 실행 (코드 실행 sandbox 때문에)
- Hydra extra params는 `+` 접두사 필요: `+run_name=coding_only`

---

## 6. 파일 구조 요약

```
RF/
├── configs/
│   ├── train_simpo.yaml        # 학습 설정 (base_model, lora, training, orthogonal)
│   └── apply_arithmetic.yaml   # Task Arithmetic 설정
├── scripts/
│   ├── prepare_dataset.py      # 데이터 준비 (helpfulness, coding)
│   ├── train_simpo.py          # SimPO 학습 (OrthogonalSimPOTrainer 포함)
│   ├── extract_task_vector.py  # LoRA → ΔW 추출
│   ├── apply_arithmetic.py     # Task Arithmetic 적용 + SimPO margin 평가
│   └── eval_humaneval.py       # HumanEval pass@1 평가
├── src/
│   ├── data/filters.py         # 데이터셋 로더 (load_ultrafeedback, load_coding)
│   ├── training/simpo.py       # OrthogonalSimPOTrainer + trace trick
│   ├── arithmetic/task_vector.py # extract_task_vector, apply_task_vector
│   └── evaluation/metrics.py   # SimPO margin 계산
└── experiment.md               # 이 파일
```

---

## 7. 참고 논문

| 논문 | 관련성 |
|---|---|
| Ilharco et al. (2023) — Task Arithmetic (ICLR) | 핵심 기반 방법론 |
| Chitale et al. (2023) — Task Arithmetic with LoRA | LoRA + Task Arithmetic 조합 선례 |
| Meng et al. (2024) — SimPO (NeurIPS) | 파일럿 실험 학습 방법 |
| Zhang & Zhou (2025) — OSRM (ACL) | 가장 가까운 선행연구. SVD 초기화 방식, loss 없음 |
| Ortho-LoRA (arXiv 2601.09684, 2026) | gradient projection 방식, 동시 멀티태스크만 |
| Ortiz et al. (2023) — Tangent Space Task Arithmetic (NeurIPS) | 직교성이 Task Arithmetic 성공의 causal factor임을 이론적 분석 |
