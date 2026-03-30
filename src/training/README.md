# src/training/ — SimPO 학습 로직

## 이 폴더가 하는 일
베이스 모델에 LoRA 어댑터를 붙이고, SimPO 방식으로 선호도 학습을 진행한다.
**Orthogonal Loss**를 추가해서 태스크 간 task vector 간섭을 최소화한다.
학습된 LoRA 가중치(어댑터)를 `outputs/adapters/`에 저장한다.

---

## 핵심 개념

### LoRA (Low-Rank Adaptation)란?
베이스 모델의 가중치(수십억 개)를 전부 학습하면 메모리가 부족하다.
LoRA는 원래 가중치는 **고정(freeze)** 하고, 작은 보조 행렬 두 개(A, B)만 학습한다.

```
원래 가중치 W (고정)
추가 행렬:  A (r × in_dim)   ← 작음
           B (out_dim × r)  ← 작음

실제 연산: W + B @ A * (alpha / r)
```

- `r` (rank): A, B 행렬의 차원. 작을수록 메모리 절약, 클수록 표현력 증가
- `alpha`: 스케일링 상수. 보통 r과 동일하게 설정

**우리 설정**: r=16, alpha=16 → scaling = 1.0

### SimPO란?
DPO(Direct Preference Optimization)의 개선 버전이다.
- DPO는 참조 모델(Reference Model)이 별도로 필요 → 메모리 2배
- SimPO는 참조 모델 없이 학습 가능 → 메모리 절약
- **KL 항이 없기 때문에** gradient가 reference 방향으로 끌리지 않음 → task vector가 더 순수하게 태스크 방향을 향함

**왜 SimPO인가**:
- Preference-only 태스크(유용성, 코딩 스타일)는 정답이 없어 SFT가 부적합
- DPO 대비: KL 없이 더 직접적인 preference signal → 더 순수한 Δθ
- Task Arithmetic과의 궁합: KL이 없으므로 task vector가 reference 방향으로 오염되지 않음

**손실 함수 핵심**:
```
L_SimPO = -log σ(β * (reward_chosen - reward_rejected - γ))
```
- `β (beta)`: 보상 차이를 얼마나 강하게 강조할지. 높을수록 margin에 민감
- `γ (gamma)`: 목표 margin. chosen이 rejected보다 최소 γ만큼 더 높기를 요구
- **우리 설정**: β=2.0, γ=0.1

### Orthogonal Loss란?
여러 태스크를 Task Arithmetic으로 합성할 때 task vector들이 서로 간섭하는 문제를 해결한다.

```
L_total = L_SimPO + α · cos²(Δθ_i, Δθ_j)
```

- `Δθ_i`: 현재 학습 중인 태스크의 task vector 방향
- `Δθ_j`: 이전 태스크의 task vector 방향 (고정)
- `cos²`: 두 방향이 평행할수록 큰 패널티 (직교하면 0, cos가 아닌 제곱이어야 음수 방향도 패널티)

**Trace Trick으로 효율적 계산**:
cos²을 직접 계산하면 ΔW = B@A 행렬이 필요해서 O(d²) 비용이 든다.
trace의 cyclic property를 이용하면 r×r 연산만으로 동일한 값을 계산할 수 있다.

```
⟨ΔWi, ΔWj⟩_F = tr(Bi^T Bj · Aj Ai^T)
             = tr(C @ D)   where C=(r×r), D=(r×r)
```

연산량: O(L·r²) ≈ 50K ops (B@A 직접 계산 대비 ~580,000배 절감)

**Orthogonal Loss 효과 (실험 결과)**:
```
helpfulness vs coding 코사인 유사도:
  baseline (일반 SimPO): 0.3018
  orth (Orthogonal SimPO): 0.0817  (73% 감소)
```

### Unsloth란?
HuggingFace Transformers + PEFT를 CUDA 레벨에서 최적화한 라이브러리.
- 같은 배치 크기에서 학습 속도 2~3배 빠름
- 메모리 사용량 20~30% 절약
- **주의**: 학습에만 사용. 평가/추론은 표준 PEFT (transformers + peft) 사용

### Gradient Accumulation (그래디언트 누적)이란?
GPU 메모리가 부족해서 batch_size=1밖에 못 쓸 때 사용.
```
gradient_accumulation_steps = 16
→ 실제 배치 크기 효과: 1 * 16 = 16
→ 16번 forward pass 후 한 번 weight 업데이트
```
메모리는 batch=1 수준으로 사용하면서 batch=16 효과를 낸다.

**왜 32가 아닌 16인가**: 32로 올렸을 때 스텝당 gradient 크기가 너무 커지는 현상 확인.
16으로 학습 안정성과 속도 균형.

---

## simpo.py 주요 구성

| 컴포넌트 | 역할 |
|---|---|
| `SimPORunConfig` | 모든 하이퍼파라미터를 담는 데이터클래스 (tau_prev_path, orthogonal_alpha 포함) |
| `OrthogonalSimPOTrainer` | CPOTrainer 상속, Orthogonal Loss 추가 |
| `load_model_and_tokenizer()` | Unsloth로 4-bit 모델 + LoRA 로드 |
| `build_trainer()` | orthogonal_alpha > 0 이면 OrthogonalSimPOTrainer, 아니면 CPOTrainer 반환 |
| `train()` | 학습 실행, 어댑터 저장, MLflow 로그 |

### CPOTrainer / OrthogonalSimPOTrainer란?
TRL 라이브러리의 통합 트레이너. `loss_type="simpo"` 설정으로 SimPO 학습을 수행한다.
`OrthogonalSimPOTrainer`는 `CPOTrainer`를 상속해서 `compute_loss`만 override한다.

### 학습 트레이너 선택 로직
```python
# tau_prev_path와 orthogonal_alpha 둘 다 설정된 경우에만 Orthogonal 학습
use_orthogonal = cfg.orthogonal_alpha > 0.0 and cfg.tau_prev_path is not None

# 기본값(null, 0.0)이면 기존 CPOTrainer 그대로 사용 → 하위 호환성 유지
```

---

## 실험 학습 결과 (현재까지)

### 태스크 A: Helpfulness SimPO
```
epoch 5, accuracy: 0.663, train_loss: 1.977
```

### 태스크 B: Coding SimPO (Baseline, Orthogonal Loss 없음)
```
epoch 7, accuracy: 0.641, margin: 0.524, train_loss: 1.610
```

### 태스크 B: Coding Orthogonal SimPO
```
epoch 7, accuracy: 0.634, margin: 0.694, train_loss: 1.630
```
→ margin이 높은 것이 orthogonal loss가 코딩 선호도 방향을 더 명확하게 학습시킨 결과

---

## 실험 순서

```
태스크 A (Helpfulness) 학습:
  python scripts/train_simpo.py data.path=data/processed/helpfulness \
    training.output_dir=outputs/adapters/helpfulness_simpo
  → outputs/adapters/helpfulness_simpo/

태스크 B (Coding) Baseline 학습 (비교군):
  python scripts/train_simpo.py data.path=data/processed/coding \
    training.output_dir=outputs/adapters/coding_simpo \
    training.num_train_epochs=7
  → outputs/adapters/coding_simpo/

태스크 B (Coding) Orthogonal 학습:
  python scripts/train_simpo.py \
    data.path=data/processed/coding \
    training.output_dir=outputs/adapters/coding_simpo_orth \
    orthogonal.tau_prev_path=data/task_vectors/helpfulness_simpo_ab.pt \
    orthogonal.alpha=0.1 \
    training.num_train_epochs=7
  → outputs/adapters/coding_simpo_orth/
```

## 학습 결과물
```
outputs/adapters/{task}_simpo/
├── adapter_config.json           ← LoRA 설정 (r, alpha, target_modules 등)
├── adapter_model.safetensors     ← 학습된 A, B 행렬 가중치
└── tokenizer 관련 파일들
```
