# Modular Alignment via SimPO & Task Arithmetic

## 프로젝트 목표
SimPO로 학습된 선호도 지식이 Task Arithmetic(벡터 산술)을 통해 모듈식으로 합성 가능한지 검증한다.

## 기술 스택
- **환경**: uv 가상환경
- **학습**: Unsloth + TRL (CPOTrainer/SimPO)
- **병합**: MergeKit / 직접 구현 Task Arithmetic
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
│   ├── train_simpo.py          # 2단계: SimPO 학습
│   ├── extract_task_vector.py  # 3단계: Task Vector 추출
│   ├── apply_arithmetic.py     # 4단계: 벡터 산술 적용
│   └── evaluate.py             # 5단계: 평가
├── src/
│   ├── data/         # 필터링·전처리 유틸
│   ├── training/     # SimPO 학습 로직
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
python scripts/prepare_dataset.py --task json
python scripts/train_simpo.py --data_path data/processed/json
python scripts/extract_task_vector.py --adapter outputs/adapters/json_simpo
python scripts/apply_arithmetic.py --vector data/task_vectors/json.pt --lambda_val 0.7
python scripts/evaluate.py --model outputs/merged/json_lam0.7
```

## MLflow UI 실행
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 주의사항
- VRAM 절약을 위해 항상 4-bit quantization + gradient checkpointing 사용
- Task Vector 추출 시 반드시 CPU에서 연산 (GPU OOM 방지)
- λ 스윕 범위: 0.0 ~ 1.5 (0.1 단위)
