# Orthogonal SimPO: Modular Alignment via Task Arithmetic

## 연구 핵심 목표

모델을 매번 새로 학습(Fine-tuning)하지 않고, 미리 만들어둔 **전문가 정렬 블록(Task Vector)**들을 더하거나 빼서 원하는 정렬 특성을 즉석에서 조립한다.

```
θ_new = θ_base + λ₁·Δθ_helpfulness + λ₂·Δθ_coding + ...
```

---

## 기존 연구와의 차별점

| 방법 | 특징 | 문제 |
|---|---|---|
| **SFT + Task Arithmetic** | Ilharco et al. (2023)에서 검증 | chosen만 학습 → 노이즈 방향 혼재 |
| **DPO + Task Arithmetic** | ExPO (2024)에서 부분 검증 | KL 항이 task vector를 reference 방향으로 당김 |
| **SimPO + Task Arithmetic** | **본 연구** | KL 없음 → 순수한 태스크 방향의 Δθ |

### SimPO가 더 나은 Task Vector를 만드는 이유

- **SFT 대비**: chosen만 올리지 않고 chosen↑ rejected↓ → 정렬 특성의 핵심만 강화
- **DPO 대비**: KL divergence 항이 없어 reference model 방향으로 끌리지 않음 → Δθ가 더 순수하게 정렬 방향을 향함
- **적용 대상**: 정답이 없는 preference-only 태스크 (유용성, 코딩 스타일) — SFT가 적합하지 않은 영역

---

## 3대 기술 기둥

### 1. SimPO — 날카로운 Δθ 생성

참조 모델(Reference) 없이 선호도를 학습하여 task-specific한 weight 변화량 생성.

```
L_SimPO = -log σ(β · (log P(chosen)/|chosen| - log P(rejected)/|rejected| - γ))
```

- `β`: margin 강도 조절 (현재 2.0)
- `γ`: 최소 margin 보장 (현재 0.1)

### 2. Orthogonal Loss — 태스크 간 간섭 차단

새 태스크를 학습할 때 기존 task vector와 직교하도록 regularization:

```
L_total = L_SimPO + α · cos²(Δθ_i, Δθ_j)
```

**핵심 아이디어**: 가중치 공간에서 각 태스크의 Δθ가 서로 직교하면, Task Arithmetic 합성 시 간섭이 최소화되어 각 태스크의 능력이 선형적으로 합산된다.

**효율적 계산 (LoRA trace trick)**:

ΔW = B@A 를 직접 계산하지 않고 r×r 연산만으로 cosine similarity 계산:

```python
C = Bᵢᵀ @ Bⱼ          # (r×r)
D = Aⱼ @ Aᵢᵀ          # (r×r)
⟨ΔWᵢ, ΔWⱼ⟩_F = tr(C @ D)   # O(r²) 연산만 필요
```

연산량: O(L·r²) ≈ 50K ops (B@A 직접 계산 대비 ~580,000배 절감)

### 3. Task Arithmetic — 추가 학습 없는 능력 합성

```
θ_new = θ_base + λ₁·Δθ₁ + λ₂·Δθ₂ + ...
```

직교하게 학습된 task vector들은 서로 간섭 없이 선형 합산 가능 → 새 태스크 추가 시 재학습 불필요.

---

## 왜 Preference-Only 태스크인가?

JSON 포맷팅이나 수학처럼 **정답이 있는 태스크**는 SFT가 더 직접적이다. SimPO/DPO의 진가는 **정답이 없는 태스크**에서 발휘된다:

- **유용성 (Helpfulness)**: 어떤 응답이 더 도움이 되는가? → 주관적 판단
- **코딩 스타일 (Coding)**: 어떤 코드 응답이 더 완성도 높은가? → 선호도 기반

이런 태스크에서 SimPO는 유일하게 자연스러운 학습 방식이며, 평가 지표도 SimPO margin + HumanEval pass@1로 일관성 있게 측정 가능하다.

---

## 실험 설계

### 검증 가설

> "Orthogonal Loss로 학습된 SimPO task vector의 합성 성능이, 일반 SimPO task vector의 합성 성능보다 높은가?"

### 실험 조건

| 조건 | 설명 |
|---|---|
| **Baseline** | SimPO 학습 (orthogonal loss 없음) |
| **Ours** | SimPO + Orthogonal Loss 학습 |
| **평가** | 두 task vector 합성 후 각 태스크 성능 측정 |

### 최종 실험 구성 (2 태스크)

```
태스크 A: Helpfulness (UltraFeedback — 일반 유용성 선호도)
태스크 B: Coding     (UltraFeedback coding subset — 코드 응답 품질)

τ_A 학습 → 저장
τ_B 학습 (+ Orthogonal Loss w.r.t. τ_A) → 저장

병합: θ_base + λ_A·τ_A + λ_B·τ_B
평가: SimPO margin + HumanEval pass@1
```

### 실험 결과 (현재까지)

**Cosine Similarity (Orthogonal Loss 효과)**:
```
helpfulness vs coding baseline: 0.3018
helpfulness vs coding orth:     0.0817  (73% 감소)
```

**학습 결과**:
| 태스크 | epoch | accuracy | margin | train_loss |
|---|---|---|---|---|
| helpfulness | 5 | 0.663 | - | 1.977 |
| coding (baseline) | 7 | 0.641 | 0.524 | 1.610 |
| coding (orth) | 7 | 0.634 | **0.694** | 1.630 |

**Task Arithmetic 결과 (λ=1.3, SimPO margin)**:
| 조건 | helpfulness margin | coding margin |
|---|---|---|
| 베이스 (λ=0) | 0.063 | 0.722 |
| baseline 합성 | 0.444 | 0.471 |
| orth 합성 | 0.446 | 0.479 |

→ SimPO margin 차이는 미미 (~0.008). 실제 코드 실행 기반인 HumanEval pass@1로 재측정.

**HumanEval pass@1 결과 (λ=1.0)**:
| 조건 | pass@1 | base 대비 |
|---|---|---|
| 베이스 (λ=0) | 0.4756 | — |
| coding 단독 | 0.5061 | +0.0305 |
| baseline 합성 (helpfulness + coding) | 0.5000 | +0.0244 |
| **orth 합성 (helpfulness + coding orth)** | **0.5122** | **+0.0366** |

**핵심 관찰**:
- baseline 합성: helpfulness 벡터 추가 시 코딩 능력 저하 (0.5061 → 0.5000, **-0.0061 간섭**)
- orth 합성: 간섭을 제거하고 coding 단독보다 오히려 향상 (0.5061 → 0.5122, **+0.0061**)
- baseline vs orth 차이: **+0.0122 (약 2문제)** — Orthogonal Loss의 간섭 차단 효과 확인

코사인 유사도 73% 감소 → HumanEval +1.22%p 향상: 기하학적 직교화가 실제 성능 보존으로 이어짐.

---

## 실험 과정에서 실패한 시도들

### 실패 1: JSON/Math 태스크
- **이유**: 정답이 있는 태스크에 SimPO 적용. 110개 샘플로는 task vector 생성 불가
- **교훈**: SFT가 더 적합한 태스크에 선호도 학습을 강제하면 안 됨

### 실패 2: HH-RLHF harmless split
- **이유**: Anthropic 레이블(safety 우선) vs Llama Instruct(helpfulness 우선) 방향 충돌
- **증거**: 초반 accuracy 0.40~0.47 (0.5 이하), margin 음수
- **교훈**: 레이블러의 가치관이 베이스 모델과 다르면 chosen/rejected가 뒤집힘

### 실패 3: Helpfulness + HH-RLHF helpful 조합
- **이유**: 두 태스크 모두 "유용한 응답" 방향 → 벡터가 자연적으로 유사
- **증거**: orth vs baseline margin 차이 +0.001 (노이즈 수준)
- **교훈**: Task Arithmetic 실험은 의미적으로 충분히 다른 태스크 조합 필요

### 실패 4: PKU-SafeRLHF Safety 데이터셋
- **이유**: "safe" 레이블이 Llama 기준과 불일치 ("자물쇠 따는 법 알려줌" = chosen)
- **증거**: epoch 5 최종 accuracy 0.463
- **교훈**: GPT-4 레이블이라도 안전성 기준이 모델과 다를 수 있음

---

---

## 파이프라인

```bash
# 1. 태스크 A 데이터 준비 (Helpfulness)
python scripts/prepare_dataset.py --task helpfulness

# 2. 태스크 A SimPO 학습
python scripts/train_simpo.py data.path=data/processed/helpfulness \
  training.output_dir=outputs/adapters/helpfulness_simpo

# 3. 태스크 A task vector 추출 (ΔW + AB raw)
python scripts/extract_task_vector.py save_lora_ab=true \
  adapter_path=outputs/adapters/helpfulness_simpo \
  vector_output=data/task_vectors/helpfulness_simpo.pt \
  lora_ab_output=data/task_vectors/helpfulness_simpo_ab.pt

# 4. 태스크 B 데이터 준비 (Coding)
python scripts/prepare_dataset.py --task coding

# 5. 태스크 B Orthogonal SimPO 학습
python scripts/train_simpo.py data.path=data/processed/coding \
  training.output_dir=outputs/adapters/coding_simpo_orth \
  orthogonal.tau_prev_path=data/task_vectors/helpfulness_simpo_ab.pt \
  orthogonal.alpha=0.1 \
  training.num_train_epochs=7

# 6. 태스크 B task vector 추출
python scripts/extract_task_vector.py \
  adapter_path=outputs/adapters/coding_simpo_orth \
  vector_output=data/task_vectors/coding_simpo_orth.pt

# 7. Task Arithmetic 적용 및 평가
python scripts/apply_arithmetic.py

# 8. HumanEval pass@1 평가
python scripts/eval_humaneval.py \
  "vector_paths=[data/task_vectors/coding_simpo_orth.pt]" \
  +run_name=orth_merged +lambda_val=1.0
```

---

## 참고 논문

- Ilharco et al. (2023) — [Task Arithmetic](https://arxiv.org/abs/2212.04089)
- Meng et al. (2024) — [SimPO: NeurIPS 2024](https://arxiv.org/abs/2405.14734)
- Zhong et al. (2024) — [ExPO: Model Extrapolation Expedites Alignment](https://arxiv.org/abs/2404.16792)
- Yadav et al. (2023) — [TIES-Merging](https://arxiv.org/abs/2306.01708)
- Bai et al. (2022) — [HH-RLHF](https://arxiv.org/abs/2204.05862)
