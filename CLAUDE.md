# Orthogonal SimPO: Modular Alignment via Task Arithmetic

## 프로젝트 목표
SimPO로 학습된 task vector들이 Orthogonal Loss를 통해 서로 간섭 없이 Task Arithmetic으로 합성 가능한지 검증한다.

핵심 가설: SimPO는 KL 항이 없어 DPO보다 순수한 태스크 방향의 Δθ를 생성하고, Orthogonal Loss는 태스크 간 간섭을 차단하여 선형 합성을 가능케 한다.

**실험 태스크**: Preference-only 태스크 (유용성, 무해성, 스타일) — 정답이 없어 SFT가 아닌 선호도 학습이 유일하게 자연스러운 영역.

## 기술 스택
- **환경**: uv 가상환경
- **학습**: Unsloth + TRL (CPOTrainer 상속 → OrthogonalSimPOTrainer)
- **병합**: 직접 구현 Task Arithmetic (fp16 베이스 모델에 Δθ 직접 합산)
- **추적**: MLflow (`sqlite:///mlflow.db`)
- **프레임워크**: Hydra, PyTorch
- **하드웨어**: RTX 4070 Ti Super 16GB VRAM

## 폴더 구조
```
RF/
├── configs/          # 학습·병합 Hydra YAML 설정
├── data/
│   ├── raw/          # 원본 다운로드 캐시
│   ├── processed/    # 필터링된 선호도 데이터셋 (HF DatasetDict)
│   └── task_vectors/ # 추출된 .pt Task Vector 파일
├── scripts/          # 실행 스크립트 (순서대로 실행)
│   ├── prepare_dataset.py      # 1단계: 데이터 준비
│   ├── train_simpo.py          # 2단계: SimPO 학습 (OrthogonalSimPOTrainer 포함)
│   ├── extract_task_vector.py  # 3단계: Task Vector 추출
│   ├── apply_arithmetic.py     # 4단계: 벡터 산술 적용 (fp16 weight 합산)
│   └── evaluate.py             # 5단계: 평가
├── src/
│   ├── data/         # 필터링·전처리 유틸
│   ├── training/     # OrthogonalSimPOTrainer, Orthogonal Loss
│   ├── arithmetic/   # Task Vector 추출·합성
│   └── evaluation/   # 벤치마크·지표
├── notebooks/        # 실험 분석
├── outputs/
│   ├── adapters/     # 학습된 LoRA 어댑터
│   └── merged/       # Task Arithmetic 결과 모델
└── logs/             # 학습 로그
```

## 실험 파이프라인 (단계별)
```bash
# 태스크 A 준비 및 학습 (Helpfulness)
python scripts/prepare_dataset.py --task helpfulness
python scripts/train_simpo.py data.path=data/processed/helpfulness \
  training.output_dir=outputs/adapters/helpfulness_simpo

# 태스크 A task vector 추출 (ΔW + AB raw)
python scripts/extract_task_vector.py save_lora_ab=true \
  adapter_path=outputs/adapters/helpfulness_simpo \
  vector_output=data/task_vectors/helpfulness_simpo.pt \
  lora_ab_output=data/task_vectors/helpfulness_simpo_ab.pt

# 태스크 B Orthogonal SimPO 학습 (태스크 A 참조)
python scripts/prepare_dataset.py --task harmlessness
python scripts/train_simpo.py data.path=data/processed/harmlessness \
  training.output_dir=outputs/adapters/harmlessness_simpo_orth \
  orthogonal.tau_prev_path=data/task_vectors/helpfulness_simpo_ab.pt \
  orthogonal.alpha=0.1

# 태스크 B task vector 추출
python scripts/extract_task_vector.py \
  adapter_path=outputs/adapters/harmlessness_simpo_orth \
  vector_output=data/task_vectors/harmlessness_simpo_orth.pt

# Task Arithmetic 합성 및 평가
python scripts/apply_arithmetic.py
```

## MLflow UI 실행
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 주의사항
- 학습 시: 4-bit quantization + gradient checkpointing (VRAM 절약)
- Task Vector 추출 시: CPU에서 연산 (GPU OOM 방지)
- apply_arithmetic.py: fp16 베이스 모델에 Δθ 직접 합산 (4-bit 모델에 더하면 shape mismatch)
- Orthogonal Loss: cos²(Δθ_i, Δθ_j) 사용 (cos가 아닌 제곱 — 음수 방향도 패널티)
- LoRA trace trick으로 cosine similarity 계산 (B@A 직접 계산 금지 — 580,000배 비쌈)
- tau_prev의 A, B 행렬은 학습 시 GPU로 올려야 gradient 계산 가능
- 평가 지표: SimPO margin (학습 objective와 동일한 기준) — JSON validity/math accuracy 대신 사용
