# scripts/ — 실행 스크립트 (순서대로 실행)

## 이 폴더가 하는 일
전체 파이프라인을 단계별로 실행하는 진입점(entry point) 스크립트들이 있다.
`src/` 안의 실제 로직을 호출하고, Hydra 설정과 MLflow 로깅을 담당한다.

---

## 현재 실험 파이프라인

```
[태스크 A — Helpfulness]
1. prepare_dataset.py        → UltraFeedback 유용성 데이터 준비
2. train_simpo.py            → Helpfulness SimPO 학습
3. extract_task_vector.py    → Helpfulness task vector 추출 (ΔW + AB raw 둘 다)

[태스크 B — Coding, Orthogonal]
4. prepare_dataset.py        → UltraFeedback coding 서브셋 준비
5. train_simpo.py (baseline) → Coding 일반 SimPO 학습 (비교군)
5. train_simpo.py (orth)     → Coding Orthogonal SimPO 학습 (Helpfulness 벡터 참조)
6. extract_task_vector.py    → Coding task vector 추출 (baseline + orth 둘 다)

[합성 평가]
7. apply_arithmetic.py       → 4-bit dequantize base model에 벡터 합산 + SimPO margin 평가
8. eval_humaneval.py         → HumanEval pass@1 평가 (코딩 능력 보존 측정)
```

---

## 각 스크립트 설명

### 1. prepare_dataset.py
공개 선호도 데이터셋에서 태스크별 데이터를 로드해서 저장한다.

```bash
python scripts/prepare_dataset.py --task helpfulness
python scripts/prepare_dataset.py --task coding
```

**데이터 출처**:
- `helpfulness`: UltraFeedback (`argilla/ultrafeedback-binarized-preferences-cleaned`)
- `coding`: UltraFeedback coding 서브셋 (strict keyword filtering)

**출력**: `data/processed/{task}/`

**과거에 시도했다가 실패한 태스크**:
- `harmlessness`: HH-RLHF harmless split — 레이블 방향 충돌 (accuracy < 0.5)
- `safety`: PKU-SafeRLHF — 레이블 충돌 (accuracy 0.463)
- `hh_helpful`: HH-RLHF helpful — helpfulness와 방향 동일, 의미 없는 Orthogonal Loss

---

### 2. train_simpo.py
LoRA + SimPO로 모델을 학습한다. Orthogonal Loss 옵션 포함.

```bash
# 일반 SimPO (태스크 A — Helpfulness)
python scripts/train_simpo.py data.path=data/processed/helpfulness \
  training.output_dir=outputs/adapters/helpfulness_simpo

# 일반 SimPO (태스크 B — Coding baseline, 비교군)
python scripts/train_simpo.py data.path=data/processed/coding \
  training.output_dir=outputs/adapters/coding_simpo \
  training.num_train_epochs=7

# Orthogonal SimPO (태스크 B — Coding, 태스크 A 참조)
python scripts/train_simpo.py \
  data.path=data/processed/coding \
  training.output_dir=outputs/adapters/coding_simpo_orth \
  orthogonal.tau_prev_path=data/task_vectors/helpfulness_simpo_ab.pt \
  orthogonal.alpha=0.1 \
  training.num_train_epochs=7
```

**출력**: `outputs/adapters/{task}_simpo/`

**학습 로그 해석**:
- `loss`: 낮을수록 좋음 (SimPO loss)
- `rewards/accuracies`: chosen을 rejected보다 높게 평가하는 비율. 0.5 = 랜덤, 1.0 = 완벽
- `rewards/margins`: chosen과 rejected 보상의 차이. 클수록 구분을 잘 함
- `orthogonal_loss`: cos²(Δθ_helpfulness, Δθ_coding). 낮을수록 두 벡터가 직교

---

### 3. extract_task_vector.py
학습된 LoRA 어댑터에서 태스크 벡터를 추출한다.

```bash
# ΔW만 추출 (apply_arithmetic용)
python scripts/extract_task_vector.py \
  adapter_path=outputs/adapters/coding_simpo \
  vector_output=data/task_vectors/coding_simpo.pt

# ΔW + AB raw 둘 다 추출 (다음 태스크 학습의 tau_prev로 사용)
python scripts/extract_task_vector.py save_lora_ab=true \
  adapter_path=outputs/adapters/helpfulness_simpo \
  vector_output=data/task_vectors/helpfulness_simpo.pt \
  lora_ab_output=data/task_vectors/helpfulness_simpo_ab.pt
```

**출력**:
- `data/task_vectors/helpfulness_simpo.pt` — ΔW 형태 (apply_arithmetic용)
- `data/task_vectors/helpfulness_simpo_ab.pt` — A, B raw 형태 (tau_prev용)

---

### 4. apply_arithmetic.py
4-bit dequantize base model에 여러 task vector를 직접 합산해서 SimPO margin을 평가한다.

```bash
python scripts/apply_arithmetic.py
```

**베이스 모델 로드 방식**:
학습 환경과 동일한 가중치 공간 유지를 위해 4-bit 모델을 로드 후 dequantize:
```python
# 4-bit 로드 → Linear4bit 모듈을 nn.Linear(bf16)으로 교체
model = load_4bit_and_dequantize("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
```

**적용 방식**:
```python
# 각 λ마다 base 모델 복원 후 task vector 합산
model.load_state_dict(base_state)
θ_new = θ_base + λ * τ_helpfulness + λ * τ_coding
```

**평가 지표**: SimPO margin (학습 objective와 동일한 기준)

---

### 5. eval_humaneval.py
HumanEval pass@1로 코딩 능력 보존을 평가한다.
실제 코드를 실행해서 정확도를 측정하는 표준 벤치마크.

```bash
# coding 단독 (기준선)
python scripts/eval_humaneval.py \
    "vector_paths=[data/task_vectors/coding_simpo.pt]" \
    +run_name=coding_only +lambda_val=1.0

# helpfulness + coding baseline 합성
python scripts/eval_humaneval.py \
    "vector_paths=[data/task_vectors/helpfulness_simpo.pt,data/task_vectors/coding_simpo.pt]" \
    +run_name=baseline_merged +lambda_val=1.0

# helpfulness + coding orth 합성
python scripts/eval_humaneval.py \
    "vector_paths=[data/task_vectors/helpfulness_simpo.pt,data/task_vectors/coding_simpo_orth.pt]" \
    +run_name=orth_merged +lambda_val=1.0
```

**주의**: Hydra extra params는 `+` 접두사 필요. `run_name=...` (not `+`)은 에러.

**베이스 모델**: fp16 (`unsloth/Llama-3.2-3B-Instruct`) — HumanEval은 inference 환경

---

## 공통 사항

### Hydra 출력 폴더
모든 스크립트는 Hydra를 사용해서 실행할 때마다 자동으로 로그를 저장한다:
```
outputs/
├── 2026-03-19/
│   ├── 15-27-48/
│   │   └── apply_arithmetic.log  ← 이번 실행 로그
```
같은 스크립트를 여러 번 돌려도 결과가 덮어써지지 않는다.

### sys.path 관련 주의사항
```python
# src/ 모듈 import용 (모든 스크립트 공통)
sys.path.insert(0, str(Path(__file__).parent.parent))

# human_eval은 miniconda에 설치 — venv 패키지 우선순위 유지를 위해 append 사용
sys.path.append("/home/lami/miniconda3/envs/deeplearning/lib/python3.10/site-packages")
# ← insert(0, ...)로 하면 miniconda의 transformers가 venv보다 먼저 로드되어 ModuleNotFoundError 발생
```
