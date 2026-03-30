# src/arithmetic/ — Task Vector 추출 & 산술

## 이 폴더가 하는 일
LoRA 어댑터에서 "태스크 벡터"를 추출하고, 이를 베이스 모델에 직접 더해서 능력을 합성한다.
여러 태스크의 벡터를 동시에 합산하는 다중 합성이 핵심 연구 목표다.

---

## 핵심 개념

### Task Arithmetic (태스크 산술)이란?
2023년 Ilharco et al. 논문에서 제안된 모델 병합 방법이다.

**기본 아이디어**:
```
θ_fine_tuned = θ_base + Δθ
             ↑ 베이스 모델  ↑ 태스크 벡터 (학습으로 인한 가중치 변화량)
```

여러 태스크를 합성:
```python
θ_new = θ_base + λ1 * τ_helpfulness + λ2 * τ_coding
```

- `λ = 0.0` → 베이스 모델 그대로
- `λ = 1.0` → 학습된 모델과 동일한 효과
- `λ > 1.0` → 태스크 벡터를 과도하게 적용 (ExPO 방식)

### LoRA에서 Task Vector를 뽑는 방법
Full fine-tuning은 `Δθ = θ_ft - θ_base`로 직접 계산할 수 있지만,
LoRA는 베이스 가중치를 안 바꾸고 A, B 행렬만 학습한다.

따라서 LoRA의 태스크 벡터는:
```python
ΔW = B @ A * (lora_alpha / r)
```
이게 곧 해당 레이어의 태스크 벡터가 된다.

**우리 구현에서**:
```python
scaling = lora_alpha / r   # = 16 / 16 = 1.0
delta = B.float() @ A.float() * scaling
```

### 두 가지 저장 형식

| 함수 | 저장 형식 | 용도 |
|---|---|---|
| `extract_task_vector()` | `{param_name: ΔW}` | apply_arithmetic에서 베이스 모델에 더할 때 |
| `extract_lora_ab()` | `{module_name: {"A": ..., "B": ...}}` | OrthogonalSimPO 학습 시 tau_prev로 사용 |

**키 이름이 다른 이유:**
```
extract_task_vector → "model.layers.0.self_attn.q_proj.weight"
                       (named_parameters() 키, base_model 접두사 제거)

extract_lora_ab     → "base_model.model.model.layers.0.self_attn.q_proj"
                       (named_modules() 키, base_model 접두사 유지)
```
`_orthogonal_loss`는 `named_modules()`로 순회하므로 접두사를 유지해야 한다.

### 베이스 모델과 Task Vector의 dtype 문제

**문제**: 학습은 4-bit 양자화 모델로 했지만, Task Arithmetic은 float weight에 직접 더해야 한다.

**해결 방법: 4-bit dequantize**
`apply_arithmetic.py`에서 4-bit 모델을 로드한 후 `bnb.nn.Linear4bit` 모듈을
일반 `nn.Linear(bfloat16)`으로 교체한다:

```python
def load_4bit_and_dequantize(base_model_path):
    # 4-bit로 로드 후...
    for module in model.named_modules():
        if isinstance(child, bnb.nn.Linear4bit):
            dequant = child.weight.dequantize().to(torch.bfloat16).cpu()
            # Linear4bit → nn.Linear(bf16) 교체
```

**왜 이렇게 하는가**:
- 학습 환경(4-bit base + LoRA)과 동일한 base weight 공간에서 Task Arithmetic 수행
- `model.to(torch.bfloat16)` 방식은 4-bit 모델에서 ValueError 발생
- fp16 버전을 따로 로드하면 quantization 차이로 base weight가 달라짐 → 잘못된 task vector 합산

---

## task_vector.py 주요 함수

| 함수 | 역할 |
|---|---|
| `extract_task_vector()` | LoRA safetensors에서 ΔW = B@A*scaling 계산, .pt 저장 |
| `extract_lora_ab()` | A, B 행렬 raw 저장 (tau_prev용) |
| `apply_task_vector()` | 베이스 모델에 λ * task_vector 합산 후 저장 |
| `lambda_sweep()` | 여러 λ에 대해 일괄 병합 모델 생성 |

### safetensors란?
HuggingFace가 만든 안전한 텐서 저장 형식 (`.pt` 대신 사용).
- 임의 코드 실행 불가 (pickle 기반 `.pt`의 보안 취약점 해결)
- 빠른 로딩 지원

---

## 코사인 유사도로 Orthogonal Loss 효과 검증

task vector끼리 얼마나 겹치는지 측정:
```python
cos_sim = sum(dot(τ_A[k], τ_B[k]) for k in layers) / (norm_A * norm_B)
```
- 값이 0에 가까울수록 두 벡터가 직교 → Task Arithmetic 합성 시 간섭 최소
- OOM 방지: 두 벡터를 동시에 로드하지 않고 레이어별 dot product 순차 계산

**실험 결과**:
```
helpfulness vs coding baseline:  0.3018
helpfulness vs coding orth:      0.0817  (73% 감소)
```
→ Orthogonal Loss가 기하학적으로 효과적임을 확인.

---

## 결과물
```
data/task_vectors/
├── helpfulness_simpo.pt        ← ΔW 형태 (apply_arithmetic용)
├── helpfulness_simpo_ab.pt     ← A, B raw 형태 (coding 학습 시 tau_prev용)
├── coding_simpo.pt             ← coding baseline ΔW
├── coding_simpo_orth.pt        ← coding Orthogonal SimPO ΔW
└── ...
```
